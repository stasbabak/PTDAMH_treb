"""
PT-MCMC with batched likelihood per step (one chain per temperature),
3-component per-chain proposal mixture, epoch-wise covariance adaptation,
and full collection of proposed points + log-likelihoods.

Key features
------------
- One chain per temperature (no replicas per temperature).
- At every step we propose once per chain, aggregate all proposed points of all
  chains into a (C, D) array, then evaluate log-likelihood in a single batched
  vmap call. Results are redistributed to chains for MH and then adjacent
  parallel-tempering swaps are attempted.
- Each chain uses a 3-component proposal mixture: (full-cov RW, eigen-line RW,
  and pCN/independence-like move). Components and their log q are provided by
  your existing `proposals.build_general_mixture_components_per_chain`.
- After each epoch, per-chain covariance is adapted from a rolling buffer or
  from all seen post-swap states, with shrinkage + jitter for numerical safety.
- We record for every step: proposed points, their log-likelihood, the chosen
  component index, acceptance mask, and swap decisions.

This module expects the proposal helpers in :mod:`ptdamh.proposals`. It does
**not** modify those helpers. If you want to plug in different proposals,
change only the `build_proposals(...)` function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple, Sequence

import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import numpy as np
from jax.scipy.linalg import solve_triangular
from tqdm import trange, tqdm

from .proposals import (
    _build_epoch_components,
    _propose_fullcov,
    _propose_eigenline,
    _propose_student_t,
    _propose_pcn,
    _pcn_logq_delta,
    _propose_stretch,
    _propose_de_two_point
)

from .utilities import (
    temperature_ladder,
    _empirical_cov,
    _shrink_spd,
    _circular_mean,
    _wrapped_diff,
    _empirical_cov_wrapped,
    PTState,
    StepInfo,
    _pt_swap_core,
    _pt_swap_core_parity,
    parallel_tempering_swap,
    _batched_logprob_chunked_fn,
    _fold_params,
    _make_indices_equal_blocks,
    _psi3_uniform_sample,
    _psi3_uniform_logpdf,
    _gather_cols,
    _apply_swaps_vec,
    make_logpost_M23,
    update_z_with_rejuv,
)


# ------------------------- Adaptive driver (m epochs) -------------------------
@dataclass
class AdaptConfig:
    m_epochs: int = 6
    N_steps: int = 500
    target_accept: float = 0.234
    eta: float = 0.05  # Robbins–Monro adaptation rate
    scale_init: float = 1.0
    scale_min: float = 0.1
    scale_max: float = 10.0
    shrink: float = 0.1
    jitter: float = 1e-6
    kappa_line: float = 3.0
    beta_base: float = 0.3
    beta_temp_scale: bool = True
    cov_mode: str = "rolling"  # "rolling" or "all_states"
    window_size: int = 10_000  # rolling window size
    downsample_every: int = 2  # store every k-th post-swap state into buffers


class InfoAccumulator:
    """Accumulate per-epoch arrays into Python lists (kept out of JIT)."""

    def __init__(self):
        self.thetas_prop = []
        self.logprob_prop = []
        self.accepted = []
        self.comp_idx = []
        self.swap_decisions = []
        self.thetas_state = []
        self.logprob_state = []
        self.temperatures = []
        self.scales_per_epoch = []
        self.swap_rate_per_epoch = []
        self.accept_rate_per_epoch = []
        self.covs_per_epoch = []
        ### for product space
        self._z_state = []  # optional
        self._z_hist = []  # optional

    def add(
        self,
        state: PTState,
        info: StepInfo,
        temps,
        scales,
        covs,
        z_hist_step=None,
        z_state_step=None,
    ):
        # stack step-dimension on host
        self.thetas_prop.append(np.asarray(info.thetas_prop))
        self.logprob_prop.append(np.asarray(info.logprob_prop))
        self.accepted.append(np.asarray(info.accepted))
        self.comp_idx.append(np.asarray(info.comp_idx))
        self.swap_decisions.append(np.asarray(info.swap_decisions))
        self.thetas_state.append(np.asarray(state.thetas))
        self.logprob_state.append(np.asarray(state.log_probs))
        self.temperatures.append(np.asarray(temps))
        self.scales_per_epoch.append(np.asarray(scales))
        # swap rate over edges this epoch
        sw = np.asarray(info.swap_decisions)
        self.swap_rate_per_epoch.append(float(sw.mean()))
        acc = np.asarray(info.accepted)
        self.accept_rate_per_epoch.append(np.asarray(acc.mean(axis=0)))
        self.covs_per_epoch.append(np.asarray(covs))
        if hasattr(state, "z"):
            self._z_state.append(np.asarray(z_state_step))
        if z_hist_step is not None:
            self._z_hist.append(np.asarray(z_hist_step))

    def pack(self):
        out = {
            "thetas_prop": (
                None if not self.thetas_prop else np.stack(self.thetas_prop, axis=0)
            ),
            "logprob_prop": (
                None if not self.logprob_prop else np.stack(self.logprob_prop, axis=0)
            ),
            "accepted": None if not self.accepted else np.stack(self.accepted, axis=0),
            "comp_idx": None if not self.comp_idx else np.stack(self.comp_idx, axis=0),
            "swap_decisions": (
                None
                if not self.swap_decisions
                else np.stack(self.swap_decisions, axis=0)
            ),
            "thetas_state": (
                None if not self.thetas_state else np.stack(self.thetas_state, axis=0)
            ),
            "logprob_state": (
                None if not self.logprob_state else np.stack(self.logprob_state, axis=0)
            ),
            "temperatures": (
                None if not self.temperatures else np.stack(self.temperatures, axis=0)
            ),
            "scales_per_epoch": (
                None
                if not self.scales_per_epoch
                else np.stack(self.scales_per_epoch, axis=0)
            ),
            "swap_rate_per_epoch": (
                None
                if not self.swap_rate_per_epoch
                else np.array(self.swap_rate_per_epoch)
            ),
            "accept_rate_per_epoch": (
                None
                if not self.accept_rate_per_epoch
                else np.stack(self.accept_rate_per_epoch, axis=0)
            ),
            "covs_per_epoch": (
                None
                if not self.covs_per_epoch
                else np.stack(self.covs_per_epoch, axis=0)
            ),
        }
        ## save product space evolution if present
        if self._z_state:
            out["z_state"] = np.stack(self._z_state, axis=0)  # (E, C)
        if self._z_hist:
            out["z_hist"] = np.stack(self._z_hist, axis=0)  # (E, T, C)
        # Also provide a flat dataset of proposed points if present
        if out["thetas_prop"] is not None:
            # Shapes: thetas_prop -> (E, T, C, D)
            E, T, C, D = out["thetas_prop"].shape
            X = out["thetas_prop"].reshape(E * T * C, D)
            y = out["logprob_prop"].reshape(E * T * C)
            acc = out["accepted"].reshape(E * T * C).astype(np.int8)
            comp = out["comp_idx"].reshape(E * T * C).astype(np.int16)
            # Derive chain ids and temperatures per sample
            chain = np.tile(np.arange(C), E * T)
            temps_epoch = out["temperatures"]  # (E, C)
            temps_rep = np.repeat(temps_epoch, T, axis=0)  # (E*T, C)
            temperature = temps_rep.reshape(E * T * C)
            dataset = dict(
                X=X,
                y_logprob=y,
                accepted=acc,
                comp_idx=comp,
                chain=chain,
                temperature=temperature,
            )

            # attach z aligned to proposals if available
            if "z_hist" in out:
                dataset["z"] = out["z_hist"].reshape(E * T * C).astype(np.int8)

            out["dataset"] = dataset
        return out

    def finalize(self):
        return self.pack()


@dataclass
class SlimInfo:
    accepted_points: list     # # accepted_points[c][w] -> np.ndarray of shape (N_acc_{c,w}, D)
    accept_rate_per_cw: np.ndarray   # (C, W)  acceptance rate per (chain, walker)
    accept_rate_per_c: np.ndarray    # (C,)    mean over walkers
    swap_rate_per_w_edge: np.ndarray # (W, C-1) swap rate per walker & edge
    swap_rate_per_edge: np.ndarray   # (C-1,)  mean over walkers


@dataclass
class SlimInfoPS:
    # Per (chain=temp, walker), split by model label at proposal time
    accepted_points_M2: list            # [C][W] -> np.ndarray (N2_{c,w}, D)
    accepted_points_M3: list            # [C][W] -> np.ndarray (N3_{c,w}, D)
    accepted_logprob_M2: list           # [C][W] -> np.ndarray (N2_{c,w},)
    accepted_logprob_M3: list           # [C][W] -> np.ndarray (N3_{c,w},)

    # Rates
    accept_rate_per_cw: np.ndarray      # (C, W)
    accept_rate_per_c:  np.ndarray      # (C,)
    swap_rate_per_w_edge: np.ndarray    # (W, C-1)
    swap_rate_per_edge:   np.ndarray    # (C-1,)

    # z diagnostics
    z_final: np.ndarray                 # (C, W) int {0,1}
    z_time_in_M3: np.ndarray            # (C, W) fraction of steps with z==1
    z_switch_count: np.ndarray          # (C, W)




# ===== Device-resident, performance-first variant =====
# - Whole epoch stays on device (jit + lax.scan)
# - No per-chain branching: compute all 3 props, select by comp_idx
# - Component is fixed per chain for the epoch (resampled each epoch from weights)
# - Symmetric proposals -> no logq terms
# - Chunked batched likelihood inside the scan

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax, vmap

# --- helpers ---


# Build proposal components per chain.

################################################################################
# --- core epoch ---


def run_epoch_device_fast(
    key,
    init_state: PTState,
    log_prob_fn_single,  # (D,) -> ()
    temperatures: jnp.ndarray,  # (C,)
    covs: jnp.ndarray,  # (C, D, D)
    scale_small: jnp.ndarray,  # (C,)
    scale_line: jnp.ndarray,  # (C,)
    scale_big: jnp.ndarray,  # (C,)
    comp_idx: jnp.ndarray,  # (C,) fixed for epoch in {0,1,2}
    n_steps: int,
    *,
    lik_chunk: int = 32,
    means: jnp.ndarray,  # <-- (C,D) NEW (per-temperature means)
    fold_mask: jnp.ndarray | None = None,  # (D,) with 0/1; None -> no folding
    period: float = 1.0,
    do_swaps: bool = False,
):
    """
    Fully device-resident epoch with symmetric proposals and fixed component per chain.
    No logq terms; MH ratio uses only tempered log-likelihood difference.
    Returns final PTState and StepInfo (stacked over steps).
    """
    C, D = init_state.thetas.shape
    fullcov, eigenline = _build_epoch_components(
        covs, scale_small, scale_line, scale_big
    )
    batched_lp = _batched_logprob_chunked_fn(log_prob_fn_single, C, D, lik_chunk)

    beta = 0.3
    nu = 5.0

    def body(carry, key_t):
        th, lp = carry  # (C, D), (C,)
        k0, k1, k2, k3, k4, kU, kS = random.split(key_t, 7)

        # 3 components
        # prop0 = _propose_fullcov(k0, th, fullcov["L_chol"], fullcov["scale_small"])   # (C, D)
        prop0 = _propose_student_t(
            k0, k1, th, fullcov["L_chol"], fullcov["scale_small"], nu=nu
        )  # (C, D)
        prop1 = _propose_eigenline(
            k2,
            k3,
            th,
            eigenline["U"],
            eigenline["S"],
            eigenline["scale"],
            axis_logits=None,
        )  # (C, D)
        prop2 = _propose_fullcov(
            k4, th, fullcov["L_chol"], fullcov["scale_big"]
        )  # (C, D)
        # prop2 = _propose_pcn(k4, th, means, fullcov["L_chol"], fullcov["scale_big"], beta=beta)


        props_all = jnp.stack([prop0, prop1, prop2], axis=0)  # (3, C, D)
        proposals = props_all[comp_idx, jnp.arange(C), :]  # (C, D)
        if fold_mask is not None:
            proposals = _fold_params(proposals, fold_mask=fold_mask, period=period)
        
        # batched likelihood
        prop_lp = batched_lp(proposals)

        # Symmetric base (Student-t & eigen-line)
        delta = prop_lp - lp
        log_alpha = delta / temperatures

        # Add Δlogq only for pCN-selected chains
        # is_pcn = (comp_idx == 2)                                                             # (C,)
        # dq = _pcn_logq_delta(th, proposals, means, fullcov["L_chol"], fullcov["scale_big"], beta)
        # log_alpha = log_alpha + jnp.where(is_pcn, dq, 0.0)                                   # (C,)

        # # symmetric MH
        # log_alpha = (prop_lp - lp) / temperatures
        # accept = jnp.log(random.uniform(kU, (C,))) < log_alpha

        # MH
        u_log = jnp.log(random.uniform(kU, (C,)))
        accept = u_log < log_alpha

        th_new = jnp.where(accept[:, None], proposals, th)
        lp_new = jnp.where(accept, prop_lp, lp)

        # print (f'debug, lp = {lp}, prop_lp = {prop_lp}, accept = {accept}, comp_idx = {comp_idx}')

        # ================= DEBUG START =================
        delta = jnp.stack([prop_lp, lp, prop_lp - lp], axis=-1)  # (C,)
        log_alpha = (prop_lp - lp) / temperatures  # (C,)
        u_log = jnp.log(random.uniform(kU, (C,)))  # (C,)
        accept = u_log < log_alpha  # (C,)
        bad_accept = (
            accept  # & ~(u_log < log_alpha)                    # (C,) must be all False
        )
        # ================= DEBUG END ===================

        # th_new = jnp.where(accept[:, None], proposals, th)
        # lp_new = jnp.where(accept,          prop_lp,     lp)

        # PT swap (adjacent)
        if do_swaps:
            _, th_sw, lp_sw, swap_dec = parallel_tempering_swap(
                kS, temperatures, th_new, lp_new
            )
        else:
            th_sw, lp_sw = th_new, lp_new
            swap_dec = jnp.zeros((C - 1,), dtype=bool)

        info_step = (proposals, prop_lp, accept, comp_idx, swap_dec)
        dbg_step = (delta, log_alpha, u_log, bad_accept)
        return (th_sw, lp_sw), (info_step, dbg_step)

    keys = random.split(key, n_steps)
    # (th_f, lp_f), (props, prop_lps, accepts, comp_idxs, swaps, th_history,
    #                deltas, log_alphas, u_logs, bad_acc) = lax.scan(
    #     body, (init_state.thetas, init_state.log_probs), keys
    # )
    (th_f, lp_f), (info_pack, dbg_pack) = lax.scan(
        body, (init_state.thetas, init_state.log_probs), keys
    )
    # unpack step info
    (props, prop_lps, accepts, comp_idxs, swaps) = info_pack
    (deltas, log_alphas, u_logs, bad_accept) = dbg_pack

    final_state = PTState(
        thetas=th_f,
        log_probs=lp_f,
        temperatures=init_state.temperatures,
        n_accepted=init_state.n_accepted + accepts.sum(axis=0).astype(jnp.int32),
        n_swaps=init_state.n_swaps + swaps.sum(axis=0).astype(jnp.int32),
        n_swap_attempts=init_state.n_swap_attempts
        + jnp.full_like(init_state.n_swaps, n_steps),
    )


    info = StepInfo(
        thetas_prop=props,  # (T, C, D)
        logprob_prop=prop_lps,  # (T, C)
        accepted=accepts,  # (T, C)
        comp_idx=comp_idxs,  # (T, C) replicates comp_idx per step
        swap_decisions=swaps,  # (T, C-1)
    )
    debug = {
        "delta": deltas,  # (T, C)
        "log_alpha": log_alphas,  # (T, C)
        "u_log": u_logs,  # (T, C)
        "bad_accept": bad_accept,  # (T, C)
    }
    return final_state, info, debug







# ---------- the one-epoch runner with product space----------
def run_epoch_device_fast_M23(
    key,
    init_state,  # PTState with .thetas (C,D), .log_probs (C,), .z (C,)
    temperatures: jnp.ndarray,  # (C,)
    covs: jnp.ndarray,  # (C,D,D)
    scale_small: jnp.ndarray,  # (C,)
    scale_line: jnp.ndarray,  # (C,)
    scale_big: jnp.ndarray,  # (C,)
    comp_idx: jnp.ndarray,  # (C,) ∈ {0,1,2} fixed this epoch
    n_steps: int,
    *,
    Npar_src: int,  # equal per-signal dimensionality
    loglik_M2_single,
    loglik_M3_single,  # single-point fns
    model_update_stride: int = 5,
    log_prior_z=(0.0, 0.0),
    psi3_sample=_psi3_uniform_sample,
    psi3_logpdf=_psi3_uniform_logpdf,
    fold_mask: jnp.ndarray | None = None,  # (D,) 0/1; None => no fold
    period: float = 1.0,
    do_swaps: bool = True,
    extra_z_flips: int = 2,  # do 2 extra Gibbs flips on hot chains
    T_hot: float = 2.0,  # chains with T > T_hot are considered "hot"
):
    C, D = init_state.thetas.shape
    hot_mask = temperatures > T_hot  # (C,) bool
    hot_mask_b = hot_mask[:, None]  # (C,1) for broadcasting to theta rows

    # indices
    idx1, idx2, idx3, idx_rest = _make_indices_equal_blocks(Npar_src, D)

    # components (chol+eig on device)
    fullcov, eigenline = _build_epoch_components(
        covs, scale_small, scale_line, scale_big
    )

    # product-space logposterior
    logpost = make_logpost_M23(
        loglik_M2_single,
        loglik_M3_single,
        idx1,
        idx2,
        idx3,
        idx_rest,
        log_prior_z=log_prior_z,
        psi3_logpdf=psi3_logpdf,
    )

    f2 = jax.jit(jax.vmap(loglik_M2_single))
    f3 = jax.jit(jax.vmap(loglik_M3_single))

    lp0, lp1 = float(log_prior_z[0]), float(log_prior_z[1])

    def swap_1d(arr, i, j, acc):
        ai, aj = arr[i], arr[j]  # (K,)
        out = arr.at[i].set(jnp.where(acc, aj, ai))
        out = out.at[j].set(jnp.where(acc, ai, aj))
        return out

    # ----- local z-update + θ3 rejuvenation (NO function args) -----
    def update_z_with_rejuv_local(keyZ, X, z):
        k_rej, k_flip = random.split(keyZ)
        # rejuvenate θ3 ~ ψ3 for z==0
        if idx3.size:
            fresh3 = psi3_sample(k_rej, C, int(idx3.size))
            X = X.at[:, idx3].set(jnp.where(z[:, None] == 0, fresh3, X[:, idx3]))

        X1 = _gather_cols(X, idx1)
        X2 = _gather_cols(X, idx2)
        X3 = _gather_cols(X, idx3)
        XR = _gather_cols(X, idx_rest)

        lp2 = f2(jnp.concatenate([X1, X2, XR], axis=1))
        lp3 = f3(jnp.concatenate([X1, X2, X3, XR], axis=1))
        lp_psi3 = psi3_logpdf(X3)

        logits = (lp3 + lp1) - (lp2 + lp0 + lp_psi3)
        beta = 1.0 / temperatures  # (C,) β = 1/T for each chain
        p1 = jax.nn.sigmoid(beta * logits)  # tempered conditional p(z=1|θ)
        # p1 = jax.nn.sigmoid(logits)
        z_new = random.bernoulli(k_flip, p1).astype(jnp.int32)
        lp_new = jnp.where(z_new.astype(bool), lp3 + lp1, lp2 + lp0 + lp_psi3)
        return X, z_new, lp_new, p1

    # stride mask for z-update
    do_model = jnp.arange(n_steps) % int(model_update_stride) == 0

    def body(carry, xs):
        (th, lp, z), (key_t, do_m) = carry, xs
        k0, k1, k2, k3, k4, kU, kS, kZ = random.split(key_t, 8)

        z_at_prop = z
        # --- three full-state proposals (θ₃ moves regardless of z) ---
        prop0 = _propose_student_t(
            k0, k1, th, fullcov["L_chol"], fullcov["scale_small"], nu=5.0
        )
        prop1 = _propose_eigenline(
            k2, k3, th, eigenline["U"], eigenline["S"], eigenline["scale"]
        )
        prop2 = _propose_fullcov(k4, th, fullcov["L_chol"], fullcov["scale_big"])
        props_all = jnp.stack([prop0, prop1, prop2], axis=0)  # (3,C,D)
        proposals = props_all[comp_idx, jnp.arange(C), :]  # (C,D)

        if fold_mask is not None:
            proposals = _fold_params(proposals, fold_mask=fold_mask, period=period)


        # --- MH under current z ---
        prop_lp = logpost(proposals, z_at_prop)  # (C,)
        delta = prop_lp - lp
        log_alpha = delta / temperatures
        u_log = jnp.log(random.uniform(kU, (C,)))
        accept = u_log < log_alpha

        th = jnp.where(accept[:, None], proposals, th)
        lp = jnp.where(accept, prop_lp, lp)
        snap_theta = (th, lp, z_at_prop)  ### after theta proposal (fixed z)

        # --- z-Gibbs (+ rejuvenation) on stride ---
        def _do(args):
            th_, z_, lp_ = args
            th2, z2, lp2, _ = update_z_with_rejuv_local(kZ, th_, z_)
            return (th2, z2, lp2)

        def _skip(args):
            return args

        (th, z, lp) = lax.cond(do_m, _do, _skip, (th, z, lp))
        z_after_gibbs = z

        # ---- EXTRA z flips for HOT chains only (cheap mixer) ----
        # Do them only when we already did a stride update (keeps cost bounded).
        def _extra_hot(args):
            th_, z_, lp_, key_base = args

            def one_flip(i, carry2):
                thc, zc, lpc, kcur = carry2
                kcur, kz2 = random.split(kcur)
                th2, z2, lp2, _ = update_z_with_rejuv_local(kz2, thc, zc)
                # apply only on hot chains
                thc = jnp.where(hot_mask_b, th2, thc)
                zc = jnp.where(hot_mask, z2, zc)
                lpc = jnp.where(hot_mask, lp2, lpc)
                return (thc, zc, lpc, kcur)

            return lax.fori_loop(0, extra_z_flips, one_flip, (th_, z_, lp_, kZ))

        def _no_extra(args):
            return args

        (th, z, lp, _) = lax.cond(do_m, _extra_hot, _no_extra, (th, z, lp, kZ))

        # --- optional PT swap: swap θ, lp, and z coherently ---
        two_swaps = True
        if do_swaps:
            key_s, th_sw, lp_sw, raster, dbg = parallel_tempering_swap(
                kS, temperatures, th, lp, return_debug=True, two_sweeps=two_swaps
            )
            if two_swaps:
                # pass 1 (even):
                i1, j1, acc1 = (
                    dbg["even"]["pairs_i"],
                    dbg["even"]["pairs_j"],
                    dbg["even"]["accept_sel"],
                )
                z1 = swap_1d(z, i1, j1, acc1)
                # pass 2 (odd):
                i2, j2, acc2 = (
                    dbg["odd"]["pairs_i"],
                    dbg["odd"]["pairs_j"],
                    dbg["odd"]["accept_sel"],
                )
                z_sw = swap_1d(z1, i2, j2, acc2)
            else:
                i, j, acc_sel = dbg["pairs_i"], dbg["pairs_j"], dbg["accept_sel"]
                # swap z using the same decisions
                z_sw = swap_1d(
                    z, i, j, acc_sel
                )  ### state after PT swap -> th_sw, lp_sw, z_sw
        else:
            th_sw, lp_sw = th, lp
            raster = jnp.zeros((C - 1,), dtype=bool)
            z_sw = z

        # collect per-step info (keep interface)
        # info_step = (proposals, prop_lp, accept, comp_idx, raster, th_sw, z_sw, z_at_prop)
        info_step = (proposals, prop_lp, accept, comp_idx, raster, z_sw, z_at_prop)
        dbg_step = (
            jnp.stack([prop_lp, lp, delta], axis=-1),
            log_alpha,
            u_log,
            accept & (u_log >= log_alpha),
        )

        return (th_sw, lp_sw, z_sw), (info_step, dbg_step)

    keys = random.split(key, n_steps)
    xs = (keys, do_model)
    (th_f, lp_f, z_f), (info_pack, dbg_pack) = lax.scan(
        body, (init_state.thetas, init_state.log_probs, init_state.z), xs
    )

    # unpack
    # (props, prop_lps, accepts, comp_idxs, swaps, th_history, z_state_history, z_prop_history) = info_pack
    (props, prop_lps, accepts, comp_idxs, swaps, z_state_history, z_prop_history) = (
        info_pack
    )
    (deltas, log_alphas, u_logs, bad_acc) = dbg_pack

    # final state & info
    final_state = type(init_state)(
        thetas=th_f,
        log_probs=lp_f,
        temperatures=init_state.temperatures,
        n_accepted=init_state.n_accepted + accepts.sum(axis=0).astype(jnp.int32),
        n_swaps=init_state.n_swaps + swaps.sum(axis=0).astype(jnp.int32),
        n_swap_attempts=init_state.n_swap_attempts
        + jnp.full_like(init_state.n_swaps, n_steps),
        z=z_f,
    )

    StepInfoCls = StepInfo  # assume your existing dataclass
    info = StepInfoCls(
        thetas_prop=props,  # (T, C, D)
        logprob_prop=prop_lps,  # (T, C)
        accepted=accepts,  # (T, C)
        comp_idx=comp_idxs,  # (T, C)
        swap_decisions=swaps,  # (T, C-1)
    )

    debug = {
        "delta": deltas,  # (T, C, 3) [prop_lp, old_lp, diff]
        "log_alpha": log_alphas,  # (T, C)
        "u_log": u_logs,  # (T, C)
        "bad_accept": bad_acc,  # (T, C) step
    }
    # return final_state, info, th_history, z_state_history, z_prop_history, debug
    return final_state, info, z_state_history, z_prop_history, debug





# --- top-level adaptive runner (single-model OR product-space) ---


def run_adaptive_pt_device_fast(
    key,
    initial_thetas: jnp.ndarray,  # (C, D)
    temperatures: jnp.ndarray,  # (C,)
    log_prob_fn_single,  # (D,) -> ()   (ignored if product_space=True)
    base_cov: np.ndarray,  # (D, D)
    *,
    fold_idx=(),
    period: float = 1.0,
    weights: np.ndarray | None = None,  # (C, up to 3)
    cfg: AdaptConfig = AdaptConfig(),
    lik_chunk: int = 32,
    big_scale_factor: float = 3.0,
    # -------- NEW: product-space options (all optional) --------
    product_space: bool = False,  # set True to use run_epoch_device_fast_M23
    Npar_src: int | None = None,  # required if product_space=True
    loglik_M2_single=None,  # required if product_space=True
    loglik_M3_single=None,  # required if product_space=True
    model_update_stride: int = 5,
    log_prior_z=(0.0, 0.0),
    psi3_sample=None,  # defaults to uniform inside epoch if None
    psi3_logpdf=None,  # defaults to 0 inside [0,1]^d3 if None
    initial_z: np.ndarray | None = None,  # (C,), 0=M2, 1=M3; default all zeros
):
    """
    End-to-end PT where each epoch is a single device loop (scan).
    - Samples comp_idx per chain from weights once per epoch
    - Symmetric proposals (no logq)
    - Updates covariances from accepted proposals of the epoch
    - Records to InfoAccumulator (same interface you use)

    If product_space=False (default): uses run_epoch_device_fast (single model).
    If product_space=True: uses run_epoch_device_fast_M23 (M2/M3 product-space).
    """
    C, D = initial_thetas.shape
    acc_buffers = [np.empty((0, D), dtype=np.float64) for _ in range(C)]
    means = np.asarray(initial_thetas)  # (C,D)
    z_hist = None

    # Initialize covs/eigs
    covs = np.tile(np.asarray(base_cov)[None, :, :], (C, 1, 1))  # host np
    scales_small = jnp.full((C,), float(cfg.scale_init))
    scales_line = jnp.full((C,), float(cfg.scale_init) * cfg.kappa_line)
    scales_big = jnp.full((C,), float(cfg.scale_init) * big_scale_factor)

    # Fold mask
    fold_mask = None
    if len(fold_idx) > 0:
        mask = np.zeros((D,), dtype=np.float32)
        mask[np.asarray(fold_idx, dtype=int)] = 1.0
        fold_mask = jnp.asarray(mask)

    # -------- initial state & initial logp (single vs product-space) --------
    if not product_space:
        batched_init = _batched_logprob_chunked_fn(
            log_prob_fn_single, C, D, max(1, min(C, lik_chunk))
        )
        logp0 = batched_init(initial_thetas)
        state = PTState(
            thetas=initial_thetas,
            log_probs=logp0,
            temperatures=temperatures,
            n_accepted=jnp.zeros((C,), dtype=jnp.int32),
            n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
            n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
        )
    else:
        # --- product-space init ---
        assert (
            (Npar_src is not None)
            and (loglik_M2_single is not None)
            and (loglik_M3_single is not None)
        ), "When product_space=True, provide Npar_src, loglik_M2_single, loglik_M3_single."
        z0 = (
            np.zeros((C,), dtype=np.int32)
            if initial_z is None
            else np.asarray(initial_z, dtype=np.int32)
        )

        # slices for equal-sized blocks
        i1 = slice(0, Npar_src)
        i2 = slice(Npar_src, 2 * Npar_src)
        i3 = slice(2 * Npar_src, 3 * Npar_src)
        iR = slice(min(3 * Npar_src, D), D)

        def _concat2(X):  # M2 args
            return jnp.concatenate([X[:, i1], X[:, i2], X[:, iR]], axis=1)

        def _concat3(X):  # M3 args
            return jnp.concatenate([X[:, i1], X[:, i2], X[:, i3], X[:, iR]], axis=1)

        f2 = jax.jit(jax.vmap(loglik_M2_single))
        f3 = jax.jit(jax.vmap(loglik_M3_single))
        lp2 = f2(_concat2(initial_thetas))
        lp3 = f3(_concat3(initial_thetas))
        lp0, lp1 = float(log_prior_z[0]), float(log_prior_z[1])
        # uniform pseudoprior ⇒ 0; if you pass psi3_logpdf, we can include it here too:
        if psi3_logpdf is not None and (3 * Npar_src) <= D:
            lp_psi = jax.jit(
                lambda X3: jnp.zeros((X3.shape[0],), X3.dtype)
            )  # safe default
            try:
                lp_psi = jax.jit(jax.vmap(psi3_logpdf))
            except Exception:
                pass
            lp_psival = lp_psi(initial_thetas[:, i3])
        else:
            lp_psival = jnp.zeros((C,), dtype=initial_thetas.dtype)

        z0_j = jnp.asarray(z0)
        logp0 = jnp.where(z0_j.astype(bool), lp3 + lp1, lp2 + lp0 + lp_psival)

        # PTState must support .z (C,) int32
        state = PTState(
            thetas=initial_thetas,
            log_probs=logp0,
            temperatures=temperatures,
            n_accepted=jnp.zeros((C,), dtype=jnp.int32),
            n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
            n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
            z=z0_j,
        )

    info_accum = InfoAccumulator()

    # Prepare weights
    if weights is None:
        W = np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float64), (C, 1))
    else:
        W = np.asarray(weights, dtype=np.float64)
        if W.shape[1] < 3:
            W = np.hstack([W, np.zeros((C, 3 - W.shape[1]))])
        W /= np.clip(W.sum(axis=1, keepdims=True), 1e-32, None)

    for epoch in trange(cfg.m_epochs, desc="Adaptive PT (device)", unit="epoch"):
        covs_j = jnp.asarray(covs)  # (C, D, D)

        # component per chain for this epoch
        comp_idx = np.array(
            [np.random.choice(3, p=W[c]) for c in range(C)], dtype=np.int32
        )
        comp_idx_j = jnp.asarray(comp_idx)

        # -------- run one epoch (branch: single vs product-space) --------
        key, subkey = random.split(key)

        if not product_space:
            state, info, debug = run_epoch_device_fast(
                subkey,
                state,
                log_prob_fn_single,
                temperatures,
                covs_j,
                scales_small,
                scales_line,
                scales_big,
                comp_idx_j,
                cfg.N_steps,
                lik_chunk=lik_chunk,
                means=means,
                fold_mask=fold_mask,
                period=period,
                do_swaps=True,
            )
        else:
            # fall back to defaults inside epoch if psi3_* not provided
            ps_sample = psi3_sample if psi3_sample is not None else _psi3_uniform_sample
            ps_logpdf = psi3_logpdf if psi3_logpdf is not None else _psi3_uniform_logpdf

            state, info, z_stat, z_hist, debug = run_epoch_device_fast_M23(
                subkey,
                state,
                temperatures,
                covs_j,
                scales_small,
                scales_line,
                scales_big,
                comp_idx_j,
                cfg.N_steps,
                Npar_src=Npar_src,
                loglik_M2_single=loglik_M2_single,
                loglik_M3_single=loglik_M3_single,
                model_update_stride=model_update_stride,
                log_prior_z=log_prior_z,
                psi3_sample=ps_sample,
                psi3_logpdf=ps_logpdf,
                fold_mask=fold_mask,
                period=period,
                do_swaps=True,
            )

        # -------- accumulate accepted proposals per temperature --------
        props = np.asarray(info.thetas_prop)  # (T_steps, C, D)
        accs = np.asarray(info.accepted, dtype=bool)  # (T_steps, C)
        for c in range(C):
            Pc = props[accs[:, c], c, :]
            if Pc.size:
                acc_buffers[c] = np.vstack([acc_buffers[c], Pc])

        # -------- adapt scales --------
        acc_rate = jnp.asarray(info.accepted).mean(axis=0).astype(jnp.float64)
        scales_small = jnp.clip(
            jnp.exp(jnp.log(scales_small) + cfg.eta * (acc_rate - cfg.target_accept)),
            cfg.scale_min,
            cfg.scale_max,
        )
        scales_line = jnp.clip(scales_line, cfg.scale_min, cfg.scale_max)
        scales_big = jnp.clip(scales_big, cfg.scale_min, cfg.scale_max)

        # -------- adapt covariances (per temperature, all accepted so far) --------
        new_covs = []
        new_means = []
        for c in range(C):
            if acc_buffers[c].shape[0] >= 2:
                cov_hat, mu_hat = _empirical_cov_wrapped(
                    acc_buffers[c], fold_idx=fold_idx, period=period, ddof=1
                )
                cov_c = _shrink_spd(
                    cov_hat,
                    shrink=getattr(cfg, "shrink", 0.1),
                    jitter=getattr(cfg, "jitter", 1e-9),
                )
            else:
                cov_c = covs[c]
                mu_hat = (
                    acc_buffers[c].mean(axis=0) if acc_buffers[c].size else np.zeros(D)
                )
            new_covs.append(cov_c)
            new_means.append(mu_hat)

        covs = np.stack(new_covs, axis=0)  # (C, D, D)
        means = np.stack(new_means, axis=0)

        # -------- record epoch results --------
        # info_accum.add(state, info, np.asarray(temperatures),
        #                np.asarray(scales_small), covs, z_hist_step=np.asarray(z_hist))

        if product_space:
            info_accum.add(
                state,
                info,
                np.asarray(temperatures),
                scales_small,
                covs,
                z_hist_step=np.asarray(z_hist),
                z_state_step=np.asarray(z_stat),
            )
        else:
            info_accum.add(state, info, np.asarray(temperatures), scales_small, covs)

        last_debug = {k: np.asarray(v) for k, v in debug.items()}

        # Optional: progress diagnostics
        if (epoch + 1) % 10 == 0 or epoch == cfg.m_epochs - 1:
            try:
                from tqdm import tqdm as _tqdm

                _tqdm.write(
                    f"Epoch {epoch + 1}/{cfg.m_epochs} | "
                    f"swap_rate={float(np.mean(np.asarray(info.swap_decisions))):.3f} | "
                    f"mean_acc={float(np.mean(np.asarray(info.accepted))):.3f}"
                    + (
                        f" | p(z=1)={float(np.mean(state.z)):.3f}"
                        if product_space
                        else ""
                    )
                )
            except Exception:
                pass

    out = info_accum.finalize()
    out["last_debug"] = last_debug
    return state, out

