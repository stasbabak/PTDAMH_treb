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
    SlimInfo,
    SlimInfoPS,
    AdaptConfig,
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




### with ensemble of walkers

def run_epoch_device_fast_ensemble(
    key,
    init_state: PTState,                 # .thetas: (C, W, D), .log_probs: (C, W)
    log_prob_fn_single,                  # (D,) -> ()
    temperatures: jnp.ndarray,           # (C,)
    covs: jnp.ndarray,                   # (C, D, D)
    scale_small: jnp.ndarray,            # (C,)
    scale_line: jnp.ndarray,             # (C,)
    scale_big: jnp.ndarray,              # (C,)
    comp_idx: jnp.ndarray,               # (C, W) in {0,1,2,3}; fixed this epoch
    n_steps: int,
    *,
    lik_chunk: int = 32,
    means: jnp.ndarray | None = None,    # (C, D) (unused here but kept for parity)
    fold_mask: jnp.ndarray | None = None,# (D,)
    period: float = 1.0,
    do_swaps: bool = True,
    stretch_a: float = 2.0,              # stretch parameter 'a'
    nu: float = 5.0,                     # dof for Student-t component
):
    """
    Ensemble epoch: (C temperatures) × (W walkers) with 5 proposal components:
      0: Student-t (small)
      1: Eigen-line
      2: Full-cov (big)
      3: Stretch move (Goodman–Weare)
      4: Differential Evolution (DE) two-point crossover
    """
    C, W, D = init_state.thetas.shape

    # precompute proposal "components" from covs like you do
    fullcov, eigenline = _build_epoch_components(covs, scale_small, scale_line, scale_big)

    # batched logprob over flattened (C*W, D)
    batched_lp = _batched_logprob_chunked_fn(log_prob_fn_single, C * W, D, lik_chunk)

    # def _fold(X):
    #     if fold_mask is None:
    #         return X
    #     Xf = X.reshape(C * W, D)
    #     Xf = _fold_params(Xf, fold_mask=fold_mask, period=period)
    #     return Xf.reshape(C, W, D)

    # beta = 0.3
    nu = 5.0

    def body(carry, key_t):
        Xth, lp = carry  # (C,W,D), (C,W)
        k0, k1, k2, k3, k4, k5, kU, kS = random.split(key_t, 8)

        prop0 = _propose_student_t(
            k0, k1, Xth, fullcov["L_chol"], fullcov["scale_small"], nu=nu)  # (C, W, D)
        prop1 = _propose_eigenline(
            k2,
            k3,
            Xth,
            eigenline["U"],
            eigenline["S"],
            eigenline["scale"],
            axis_logits=None,)  # (C, D)
        prop2 = _propose_fullcov(
            k4, Xth, fullcov["L_chol"], fullcov["scale_big"])  # (C, W, D)
        # prop2 = _propose_pcn(k4, th, means, fullcov["L_chol"], fullcov["scale_big"], beta=beta)

        # ---------- component 3: Stretch move ----------
        k5p, k5s = random.split(k5)
        prop3, logJ_stretch, partner, has_partner, zfac  = _propose_stretch(k5p, k5s, Xth, a=stretch_a, 
                                                                            z=None)
        # -------- component 4: DE two-point crossover ----------
        k_partner, k_gamma= random.split(kU)
        prop4, idx_y, idx_z, has_pair, mask = _propose_de_two_point(
                    k_partner, k_gamma, Xth, gamma=None, gamma_scale=2.38,
                    crossover_rate=0.8, jitter_scale=1e-6
        )

#     k_partner, k_gamma, Xth, z=z_at_prop,
#     same_z_required=True, gamma=None, gamma_scale=2.38,
#     crossover_rate=0.9, jitter_scale=1e-6
# )

        # We use all proposals but then select the proposed points of a particular proposal: easier
        # stack and pick per-walker component
        props_all = jnp.stack([prop0, prop1, prop2, prop3, prop4], axis=0)  # (5,C,W,D)
        idx = comp_idx[None, :, :, None]                  # (1, C, W, 1)
        proposals = jnp.take_along_axis(
            props_all,                                    # (4, C, W, D)
            jnp.repeat(idx, repeats=props_all.shape[-1], axis=3),  # (1, C, W, D)
            axis=0
        )[0] # (C, W, D)
        # proposals = props_all[comp_idx, jnp.arange(C)[:, None], jnp.arange(W)[None, :], :]
        if fold_mask is not None:
            proposals = _fold_params(proposals, fold_mask=fold_mask, period=period)

        # batched logprob (flatten -> call -> reshape)
        prop_lp = batched_lp(proposals.reshape(C * W, D)).reshape(C, W)

        # base MH term (tempered)
        delta = prop_lp - lp
        log_alpha = delta / temperatures[:, None]

        # add log Jacobian ONLY for stretch (comp==3)
        is_stretch = (comp_idx == 3)
        log_alpha = log_alpha + jnp.where(is_stretch, logJ_stretch, 0.0)

        # accept/reject
        u_log = jnp.log(random.uniform(kU, shape=(C, W)))
        accept = u_log < log_alpha

        X_new = jnp.where(accept[..., None], proposals, Xth)
        lp_new = jnp.where(accept, prop_lp, lp)

        # optional: parallel tempering swaps PER WALKER (vmap over W)
        if do_swaps:
            # split swap keys for W walkers
            ks = random.split(kS, W)

            def _swap_one(k, x_w, lp_w):
                # x_w: (C,D), lp_w: (C,)
                _, x_sw, lp_sw, swap_dec = parallel_tempering_swap(k, temperatures, x_w, lp_w)
                return x_sw, lp_sw, swap_dec

            xw = jnp.swapaxes(X_new, 0, 1)     # (W,C,D)
            lpw = jnp.swapaxes(lp_new, 0, 1)   # (W,C)
            x_sw, lp_sw, swaps = jax.vmap(_swap_one, in_axes=(0, 0, 0))(ks, xw, lpw)
            X_out = jnp.swapaxes(x_sw, 0, 1)   # (C,W,D)
            lp_out = jnp.swapaxes(lp_sw, 0, 1) # (C,W)
            swap_dec = swaps                   # (W, C-1)
        else:
            X_out, lp_out = X_new, lp_new  # (C,W,D), (C,W)
            swap_dec = jnp.zeros((W, C - 1), dtype=bool)

        info_step = (proposals, prop_lp, accept, comp_idx, swap_dec)
        # minimal debug like your single-walker version
        # dbg_step = (jnp.stack([prop_lp, lp, delta], axis=-1),
        #             log_alpha, u_log, accept & (u_log >= log_alpha))
        dbg_step = {}

        return (X_out, lp_out), (info_step, dbg_step)

    keys = random.split(key, n_steps)
    (Xf, lpf), (info_pack, dbg_pack) = lax.scan(body, (init_state.thetas, init_state.log_probs), keys)

    (proposals, prop_lps, accepts, comp_idxs, swaps) = info_pack  # props: (T,C,W,D), swaps: (T,W,C-1)
    # Shapes:
    #   proposals: (T, C, W, D)
    #   prop_lps : (T, C, W)
    #   accepts  : (T, C, W)   (bool)
    #   comp_idxs: (T, C, W)   (your comp_idx replicated per step)
    #   swaps    : (T, W, C-1) (bool)

    # (deltas, log_alphas, u_logs, bad_acc) = dbg_pack

    final_state = PTState(
        thetas=Xf,
        log_probs=lpf,
        temperatures=init_state.temperatures,
        n_accepted=init_state.n_accepted + accepts.sum(axis=(0, 2)).astype(jnp.int32),  # (C,)
        n_swaps=init_state.n_swaps + swaps.sum(axis=(0, 1)).astype(jnp.int32),         # (C-1,)
        n_swap_attempts=init_state.n_swap_attempts + jnp.full_like(init_state.n_swaps, n_steps * W),
    )

    # info = StepInfo(
    #     thetas_prop=props,         # (T, C, W, D)
    #     logprob_prop=prop_lps,     # (T, C, W)
    #     accepted=accepts,          # (T, C, W)
    #     comp_idx=comp_idxs,        # (T, C, W)
    #     swap_decisions=swaps,      # (T, W, C-1)
    # )

    # ----- Convert to host for flexible ragged packing (Ȓ often done host-side) -----
    props_np   = np.asarray(proposals)        # (T,C,W,D)
    accepts_np = np.asarray(accepts)          # (T,C,W)
    swaps_np   = np.asarray(swaps)            # (T,W,C-1)

    T, C, W, D = props_np.shape

    # ----- Per-(chain,walker) accepted points (ragged lists) -----
    accepted_points = [[None for _ in range(W)] for _ in range(C)]
    accept_rate_per_cw = np.zeros((C, W), dtype=np.float64)

    for c in range(C):
        for w in range(W):
            mask = accepts_np[:, c, w]                  # (T,)
            pts  = props_np[mask, c, w, :]              # (N_acc_{c,w}, D)
            accepted_points[c][w] = pts                 # store ragged array
            accept_rate_per_cw[c, w] = float(mask.mean())

    # Chain- and edge-level summaries (optional but handy)
    accept_rate_per_c   = accept_rate_per_cw.mean(axis=1)        # (C,)
    swap_rate_per_w_edge = swaps_np.mean(axis=0)                 # (W, C-1)
    swap_rate_per_edge   = swap_rate_per_w_edge.mean(axis=0)     # (C-1,)


    slim = SlimInfo(
        accepted_points=accepted_points, 
        accept_rate_per_cw=accept_rate_per_cw,
        accept_rate_per_c=accept_rate_per_c,
        swap_rate_per_w_edge=swap_rate_per_w_edge,
        swap_rate_per_edge=swap_rate_per_edge,
        )

    # debug = {}
        # "delta": deltas, "log_alpha": log_alphas, "u_log": u_logs, "bad_accept": bad_acc}

    return final_state, slim


######### one epoch ensemble for product space. ##########

def run_epoch_device_fast_M23_ensemble(
    key,
    init_state,                         # PTState with .thetas (C,W,D), .log_probs (C,W), .z (C,W)
    temperatures: jnp.ndarray,          # (C,)
    covs: jnp.ndarray,                  # (C,D,D)
    scale_small: jnp.ndarray,           # (C,)
    scale_line: jnp.ndarray,            # (C,)
    scale_big: jnp.ndarray,             # (C,)
    comp_idx: jnp.ndarray,              # (C,W) in {0,1,2,3} fixed this epoch
    n_steps: int,
    *,
    Npar_src: int,
    loglik_M2_single,
    loglik_M3_single,
    model_update_stride: int = 5,
    log_prior_z=(0.0, 0.0),
    psi3_sample=_psi3_uniform_sample,
    psi3_logpdf=_psi3_uniform_logpdf,
    fold_mask: jnp.ndarray | None = None,  # (...,D)
    period: float = 1.0,
    do_swaps: bool = True,
    extra_z_flips: int = 2,
    T_hot: float = 2.0,
    nu: float = 5.0,
):
    """
    Product-space PT epoch with an ensemble of walkers:
      - state per step is (C chains) x (W walkers)
      - z is updated per (chain, walker) by Gibbs with optional rejuvenation of θ3
      - PT swaps happen per walker across chains and also swap z coherently.
    """
    C, W, D = init_state.thetas.shape
    hot_mask   = temperatures > T_hot            # (C,)
    hot_mask_b = hot_mask[:, None]               # (C,1)

    # indices for [θ1 | θ2 | θ3 | rest]
    idx1, idx2, idx3, idx_rest = _make_indices_equal_blocks(Npar_src, D)
    d3 = int(idx3.size)

    # components (chol+eig per chain)
    fullcov, eigenline = _build_epoch_components(covs, scale_small, scale_line, scale_big)

    # vectorized single-point likelihoods
    f2 = jax.jit(jax.vmap(loglik_M2_single))  # (N, D_2) -> (N,)
    f3 = jax.jit(jax.vmap(loglik_M3_single))  # (N, D_3) -> (N,)

    lp0, lp1 = float(log_prior_z[0]), float(log_prior_z[1])

    # ---------- helpers ----------
    def _flat(X):   # (...,D) -> (C*W, D)
        return X.reshape(C * W, X.shape[-1])

    def _unflat(v): # (C*W,) -> (C,W)
        return v.reshape(C, W)

    def _gather3D(X, cols):  # X: (C,W,D) -> (C,W,len(cols))
        return X[..., cols]

    # Gibbs on z with θ3 rejuvenation when z==0 (per (C,W))
    def update_z_with_rejuv_local(keyZ, X, z):
        k_rej, k_flip = random.split(keyZ)

        # rejuvenate θ3 ~ ψ3 for z==0
        if d3 > 0:
            fresh3 = psi3_sample(k_rej, C * W, d3).reshape(C, W, d3)   # (C,W,d3)
            X = X.at[:, :, idx3].set(jnp.where(z[..., None] == 0, fresh3, X[:, :, idx3]))

        X1 = _gather3D(X, idx1)   # (C,W,·)
        X2 = _gather3D(X, idx2)
        X3 = _gather3D(X, idx3)
        XR = _gather3D(X, idx_rest)

        # build model-specific argument matrices and flatten to (C*W, ·)
        X2_args = jnp.concatenate([X1, X2, XR], axis=-1)
        X3_args = jnp.concatenate([X1, X2, X3, XR], axis=-1)

        lp2 = _unflat(f2(_flat(X2_args)))       # (C,W)
        lp3 = _unflat(f3(_flat(X3_args)))       # (C,W)
        lp_psi3 = _unflat(psi3_logpdf(_flat(X3))) if d3 > 0 else jnp.zeros((C, W), X.dtype)

        logits = (lp3 + lp1) - (lp2 + lp0 + lp_psi3)    # (C,W)
        beta   = (1.0 / temperatures)[:, None]          # (C,1)
        p1     = jax.nn.sigmoid(beta * logits)          # tempered Bernoulli for z=1

        z_new  = random.bernoulli(k_flip, p1).astype(jnp.int32)   # (C,W)
        lp_new = jnp.where(z_new.astype(bool), lp3 + lp1, lp2 + lp0 + lp_psi3)
        return X, z_new, lp_new, p1

    # stride mask for z updates
    do_model = jnp.arange(n_steps) % int(model_update_stride) == 0

    # swap z with same accept decisions as θ, per walker
    def swap_1d(arr, i, j, acc):
        # arr: (C,), i/j/acc: (C-1,)
        ai, aj = arr[i], arr[j]
        out = arr.at[i].set(jnp.where(acc, aj, ai))
        out = out.at[j].set(jnp.where(acc, ai, aj))
        return out

    stretch_a = 2.0
    def body(carry, xs):
        (th, lp, z), (key_t, do_m) = carry, xs
        k0, k1, k2, k3, k4, k5, kU, kSwap, kZ, _ = random.split(key_t, 10)

        z_at_prop = z  # (C,W)

        # ---- three proposal components (full-state; θ3 moves regardless of z) ----
        prop0 = _propose_student_t(k0, k1, th, fullcov["L_chol"], fullcov["scale_small"], nu=nu)        # (C,W,D)
        prop1 = _propose_eigenline(k2, k3, th, eigenline["U"], eigenline["S"], eigenline["scale"])      # (C,W,D)
        prop2 = _propose_fullcov(k4, th, fullcov["L_chol"], fullcov["scale_big"])   
        # stretch per (c,w) within SAME z label
        kP, kS = jax.random.split(k5)
        prop3_raw, logJ, partner, has_partner, zfac = _propose_stretch(kP, kS, th, a=stretch_a, z=z_at_prop)
        prop3 = jnp.where(has_partner[..., None], prop3_raw, prop1)  # e.g., prop_fallback = prop1
        logJ  = jnp.where(has_partner,           logJ,  0.0)
        k_partner, k_gamma = jax.random.split(kU)
        prop4, idx_y, idx_z, has_pair, mask = _propose_de_two_point(
            k_partner, k_gamma, th, z=z_at_prop,
            same_z_required=True, gamma=None, gamma_scale=2.38,
            crossover_rate=0.8, jitter_scale=1e-6)

        props_all = jnp.stack([prop0, prop1, prop2, prop3, prop4], axis=0)  # (5,C,W,D)

        # select by comp_idx per (C,W)
        idx = comp_idx[None, :, :, None]  # (1,C,W,1)
        proposals = jnp.take_along_axis(
            props_all,
            jnp.repeat(idx, repeats=props_all.shape[-1], axis=3),
            axis=0
        )[0]  # (C,W,D)

        if fold_mask is not None:
            proposals = _fold_params(proposals, fold_mask=fold_mask, period=period)

        # ---- log posterior under current z (piecewise) ----
        P1 = _gather3D(proposals, idx1)
        P2 = _gather3D(proposals, idx2)
        P3 = _gather3D(proposals, idx3)
        PR = _gather3D(proposals, idx_rest)

        P2_args = jnp.concatenate([P1, P2, PR], axis=-1)
        P3_args = jnp.concatenate([P1, P2, P3, PR], axis=-1)

        lp2_prop = _unflat(f2(_flat(P2_args)))                     # (C,W)
        lp3_prop = _unflat(f3(_flat(P3_args)))                     # (C,W)
        lp_psi3_prop = _unflat(psi3_logpdf(_flat(P3))) if d3 > 0 else jnp.zeros((C, W), proposals.dtype)

        prop_lp = jnp.where(z_at_prop.astype(bool), lp3_prop + lp1, lp2_prop + lp0 + lp_psi3_prop)

        # ---- MH accept (tempered) ----
        delta = prop_lp - lp
        log_alpha = delta / temperatures[:, None]
        is_stretch = (comp_idx == 3)
        log_alpha = log_alpha + jnp.where(is_stretch, logJ, 0.0)
        # DE proposal is symmetric, logq=0.


        u_log = jnp.log(random.uniform(kU, (C, W)))
        accept = u_log < log_alpha  # (C,W)

        th = jnp.where(accept[..., None], proposals, th)
        lp = jnp.where(accept,            prop_lp,   lp)

        # ---- z Gibbs on stride (with extra flips on hot chains) ----
        def _do(args):
            th_, z_, lp_ = args
            th2, z2, lp2, _ = update_z_with_rejuv_local(kZ, th_, z_)
            return (th2, z2, lp2)

        (th, z, lp) = lax.cond(do_m, _do, lambda a: a, (th, z, lp))

        def _extra_hot(args):
            th_, z_, lp_, key_base = args

            def one_flip(i, carry2):
                thc, zc, lpc, kcur = carry2
                kcur, kz2 = random.split(kcur)
                th2, z2, lp2, _ = update_z_with_rejuv_local(kz2, thc, zc)
                # apply only on hot chains (all walkers at those chains)
                thc = jnp.where(hot_mask_b[..., None], th2, thc)
                zc  = jnp.where(hot_mask_b,          z2,  zc)
                lpc = jnp.where(hot_mask_b,          lp2, lpc)
                return (thc, zc, lpc, kcur)

            return lax.fori_loop(0, extra_z_flips, one_flip, (th_, z_, lp_, kZ))

        (th, z, lp, _) = lax.cond(do_m, _extra_hot, lambda a: a, (th, z, lp, kZ))

        # ---- PT swap per walker (swap θ, lp, and z coherently) ----
        if do_swaps:
            ks = random.split(kSwap, W)

            def _swap_one(k, x_w, lp_w, z_w):
                # x_w: (C,D), lp_w: (C,), z_w: (C,)
                key_s, x_sw, lp_sw, raster, dbg = parallel_tempering_swap(
                    k, temperatures, x_w, lp_w, return_debug=True, two_sweeps=True
                )
                # even then odd pass swap of z
                z1 = swap_1d(z_w, dbg["even"]["pairs_i"], dbg["even"]["pairs_j"], dbg["even"]["accept_sel"])
                z2 = swap_1d(z1,  dbg["odd"]["pairs_i"],  dbg["odd"]["pairs_j"],  dbg["odd"]["accept_sel"])
                return x_sw, lp_sw, z2, raster

            # vmapped over walkers
            xw  = jnp.swapaxes(th, 0, 1)   # (W,C,D)
            lpw = jnp.swapaxes(lp, 0, 1)   # (W,C)
            zw  = jnp.swapaxes(z,  0, 1)   # (W,C)

            x_sw, lp_sw, z_sw, swaps = jax.vmap(_swap_one, in_axes=(0, 0, 0, 0))(ks, xw, lpw, zw)

            th = jnp.swapaxes(x_sw, 0, 1)  # (C,W,D)
            lp = jnp.swapaxes(lp_sw, 0, 1) # (C,W)
            z  = jnp.swapaxes(z_sw,  0, 1) # (C,W)
            swap_dec = swaps               # (W, C-1)
        else:
            swap_dec = jnp.zeros((W, C - 1), dtype=bool)

        # record minimal info
        info_step = (proposals, prop_lp, accept, comp_idx, swap_dec, z, z_at_prop)
        dbg_step  = (jnp.stack([prop_lp, lp, delta], axis=-1), log_alpha, u_log, accept & (u_log >= log_alpha))
        return (th, lp, z), (info_step, dbg_step)

    keys = random.split(key, n_steps)
    xs = (keys, do_model)
    (th_f, lp_f, z_f), (info_pack, dbg_pack) = lax.scan(
        body, (init_state.thetas, init_state.log_probs, init_state.z), xs
    )

    # unpack info
    (proposals, prop_lps, accepts, comp_idxs, swaps, z_state_hist, z_prop_hist) = info_pack
    # (deltas, log_alphas, u_logs, bad_acc) = dbg_pack
    # proposals:   (T,C,W,D)
    # prop_lps:    (T,C,W)
    # accepts:     (T,C,W)
    # swaps:       (T,W,C-1)
    # z_state_hist:(T,C,W) after step (post PT swap)
    # z_prop_hist:(T,C,W) model label at proposal evaluation

    props_np   = np.asarray(proposals)
    lps_np     = np.asarray(prop_lps)
    accepts_np = np.asarray(accepts)
    swaps_np   = np.asarray(swaps)
    z_hist_np  = np.asarray(z_state_hist)
    zprop_np   = np.asarray(z_prop_hist)

    Tsteps, Cn, Wn, Dn = props_np.shape
    accepted_points_M2 = [[None for _ in range(Wn)] for _ in range(Cn)]
    accepted_points_M3 = [[None for _ in range(Wn)] for _ in range(Cn)]
    accepted_logprob_M2 = [[None for _ in range(Wn)] for _ in range(Cn)]
    accepted_logprob_M3 = [[None for _ in range(Wn)] for _ in range(Cn)]
    accept_rate_per_cw = np.zeros((Cn, Wn), dtype=float)
    for c in range(Cn):
        for w in range(Wn):
            mask = accepts_np[:, c, w]                     # (T,)
            accept_rate_per_cw[c, w] = float(mask.mean())
            if not mask.any():
                accepted_points_M2[c][w]  = np.empty((0, Dn))
                accepted_points_M3[c][w]  = np.empty((0, Dn))
                accepted_logprob_M2[c][w] = np.empty((0,), dtype=float)
                accepted_logprob_M3[c][w] = np.empty((0,), dtype=float)
                continue
            pts  = props_np[mask, c, w, :]                 # (N_acc, D)
            lpa  = lps_np[mask, c, w]                      # (N_acc,)
            z_at = zprop_np[mask, c, w].astype(bool)       # (N_acc,)

            # split by model label at proposal time
            accepted_points_M2[c][w]  = pts[~z_at] if (~z_at).any() else np.empty((0, Dn))
            accepted_points_M3[c][w]  = pts[ z_at] if ( z_at).any() else np.empty((0, Dn))
            accepted_logprob_M2[c][w] = lpa[~z_at] if (~z_at).any() else np.empty((0,), dtype=float)
            accepted_logprob_M3[c][w] = lpa[ z_at] if ( z_at).any() else np.empty((0,), dtype=float)

    accept_rate_per_c    = accept_rate_per_cw.mean(axis=1)     # (C,)
    swap_rate_per_w_edge = swaps_np.mean(axis=0)               # (W, C-1)
    swap_rate_per_edge   = swap_rate_per_w_edge.mean(axis=0)   # (C-1,)

    # z summaries (per temp and walker)
    z_final        = np.asarray(z_f).astype(int)               # (C,W)
    z_time_in_M3   = z_hist_np.mean(axis=0)                    # (C,W)
    z_switch_count = (np.diff(z_hist_np, axis=0) != 0).sum(axis=0)  # (C,W)

    slim = SlimInfoPS(
        accepted_points_M2=accepted_points_M2,
        accepted_points_M3=accepted_points_M3,
        accepted_logprob_M2=accepted_logprob_M2,
        accepted_logprob_M3=accepted_logprob_M3,
        accept_rate_per_cw=accept_rate_per_cw,
        accept_rate_per_c=accept_rate_per_c,
        swap_rate_per_w_edge=swap_rate_per_w_edge,
        swap_rate_per_edge=swap_rate_per_edge,
        z_final=z_final,
        z_time_in_M3=z_time_in_M3,
        z_switch_count=z_switch_count,
    )

    # final state
    final_state = type(init_state)(
        thetas=th_f,
        log_probs=lp_f,
        temperatures=init_state.temperatures,
        n_accepted=init_state.n_accepted + accepts.sum(axis=(0, 1)).astype(jnp.int32),   # sum over T & W per chain
        n_swaps=init_state.n_swaps + swaps.sum(axis=(0, 1)).astype(jnp.int32),          # (C-1,)
        n_swap_attempts=init_state.n_swap_attempts + jnp.full_like(init_state.n_swaps, n_steps * W),
        z=z_f,
    )

    # StepInfo-like container (now ensemble-shaped)
    # info = StepInfo(
    #     thetas_prop=props,          # (T, C, W, D)
    #     logprob_prop=prop_lps,      # (T, C, W)
    #     accepted=accepts,           # (T, C, W)
    #     comp_idx=comp_idxs,         # (T, C, W) (replicates input per step)
    #     swap_decisions=swaps,       # (T, W, C-1)
    # )

    # debug = {
    #     "delta": deltas, "log_alpha": log_alphas, "u_log": u_logs, "bad_accept": bad_acc
    # }
    # return z histories too if you need them for diagnostics
    # return final_state, info, z_state_hist, z_prop_hist, debug
    return final_state, slim




###.  ------ top level runnner ------


def run_adaptive_pt_device_fast(
    key,
    initial_thetas: jnp.ndarray,        # (C, D) or (C, W, D)
    temperatures: jnp.ndarray,          # (C,)
    log_prob_fn_single,                 # (D,) -> ()   (ignored if product_space=True)
    base_cov: np.ndarray,               # (D, D)
    *,
    fold_idx=(),
    period: float = 1.0,
    weights: np.ndarray | None = None,  # (4,), (C,4) or (C,W,4) for 4 components
    cfg: AdaptConfig = AdaptConfig(),
    lik_chunk: int = 32,
    big_scale_factor: float = 3.0,
    # ----- product-space options -----
    product_space: bool = False,        # set True to use *M23_ensemble runner*
    Npar_src: int | None = None,        # required if product_space=True
    loglik_M2_single=None,              # required if product_space=True
    loglik_M3_single=None,              # required if product_space=True
    model_update_stride: int = 5,
    log_prior_z=(0.0, 0.0),
    psi3_sample=None,                   # defaults inside epoch if None
    psi3_logpdf=None,                   # defaults inside epoch if None
    initial_z: np.ndarray | None = None # (C,) or (C,W); default all zeros
):
    """
    Adaptive PT driver with an ensemble of walkers.
    - Each epoch is executed fully on device by your provided single-epoch functions
      (run_epoch_device_fast_ensemble / run_epoch_device_fast_M23_ensemble).
    - Covariances are adapted per-temperature from all accepted points pooled across walkers.
    - Scales are Robbins–Monro adapted from per-chain acceptance (averaged over walkers).
    """
    # ---------- shapes & basic setup ----------
    if initial_thetas.ndim == 2:
        # upgrade to ensemble with W=1
        C, D = initial_thetas.shape
        W = 1
        initial_thetas = initial_thetas[:, None, :]          # (C,1,D)
    else:
        C, W, D = initial_thetas.shape

    means = np.asarray(initial_thetas.mean(axis=1))          # (C, D) (not used by epoch, kept for parity)

    # per-chain covariance & scales
    covs = np.tile(np.asarray(base_cov)[None, :, :], (C, 1, 1))  # (C,D,D) host np
    scales_small = jnp.full((C,), float(cfg.scale_init))
    scales_line  = jnp.full((C,), float(cfg.scale_init) * cfg.kappa_line)
    scales_big   = jnp.full((C,), float(cfg.scale_init) * big_scale_factor)

    # fold mask
    fold_mask = None
    if len(fold_idx) > 0:
        mask = np.zeros((D,), dtype=np.float32)
        mask[np.asarray(fold_idx, dtype=int)] = 1.0
        fold_mask = jnp.asarray(mask)

    # ---------- initial state ----------
    if not product_space:
        # logp for ensemble (C*W batch)
        _batched_init = _batched_logprob_chunked_fn(
            log_prob_fn_single, C * W, D, max(1, min(C * W, lik_chunk))
        )
        logp0 = _batched_init(initial_thetas.reshape(C * W, D)).reshape(C, W)
        state = PTState(
            thetas=initial_thetas,         # (C,W,D)
            log_probs=logp0,               # (C,W)
            temperatures=temperatures,     # (C,)
            n_accepted=jnp.zeros((C,), dtype=jnp.int32),
            n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
            n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
        )
    else:
        # product-space init
        assert (Npar_src is not None) and (loglik_M2_single is not None) and (loglik_M3_single is not None), \
            "When product_space=True, provide Npar_src, loglik_M2_single, loglik_M3_single."

        # z init -> (C,W)
        if initial_z is None:
            z0 = np.zeros((C, W), dtype=np.int32)
        else:
            z0 = np.asarray(initial_z)
            if z0.ndim == 1:
                z0 = np.broadcast_to(z0[:, None], (C, W))
            assert z0.shape == (C, W)

        # build slices
        i1 = slice(0, Npar_src)
        i2 = slice(Npar_src, 2 * Npar_src)
        i3 = slice(2 * Npar_src, 3 * Npar_src)
        iR = slice(min(3 * Npar_src, D), D)

        def _flat(X):   # (C,W,D)->(C*W,D)
            return X.reshape(C * W, D)
        def _unflat(v): # (C*W,)->(C,W)
            return v.reshape(C, W)

        f2 = jax.jit(jax.vmap(loglik_M2_single))
        f3 = jax.jit(jax.vmap(loglik_M3_single))

        X = initial_thetas
        X2_args = jnp.concatenate([X[..., i1], X[..., i2], X[..., iR]], axis=-1)
        X3_args = jnp.concatenate([X[..., i1], X[..., i2], X[..., i3], X[..., iR]], axis=-1)

        lp2 = _unflat(f2(_flat(X2_args)))
        lp3 = _unflat(f3(_flat(X3_args)))

        if psi3_logpdf is None:
            lp_psi = jnp.zeros((C, W), dtype=X.dtype)
        else:
            lp_psi = _unflat(jax.jit(jax.vmap(psi3_logpdf))(_flat(X[..., i3])))

        lp0, lp1 = float(log_prior_z[0]), float(log_prior_z[1])
        z0_j = jnp.asarray(z0)
        logp0 = jnp.where(z0_j.astype(bool), lp3 + lp1, lp2 + lp0 + lp_psi)

        state = PTState(
            thetas=initial_thetas,         # (C,W,D)
            log_probs=logp0,               # (C,W)
            temperatures=temperatures,     # (C,)
            n_accepted=jnp.zeros((C,), dtype=jnp.int32),
            n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
            n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
            z=z0_j                          # (C,W)
        )

    # ---------- component weights -> (C,W,4) ----------
    # We assume 5 components in your epoch code: Student-t, Eigen-line, Fullcov, Stretch, DE.
    def _prep_weights():
        num_comps = 5
        if weights is None:
            Wts = np.ones((C, W, num_comps), dtype=np.float64) / num_comps
        else:
            arr = np.asarray(weights, dtype=np.float64)
            if arr.ndim == 1:
                assert arr.shape[0] == num_comps, f"weights must have {num_comps} components"
                Wts = np.broadcast_to(arr[None, None, :], (C, W, num_comps))
            elif arr.ndim == 2:
                assert arr.shape[0] == C and arr.shape[1] == num_comps
                Wts = np.broadcast_to(arr[:, None, :], (C, W, num_comps))
            else:
                assert arr.shape[:2] == (C, W) and arr.shape[2] == num_comps
                Wts = arr
            # normalize per (c,w)
            Wts = Wts / np.clip(Wts.sum(axis=-1, keepdims=True), 1e-32, None)
        return Wts
    Wts = _prep_weights()  # (C,W,5)

    # ---------- buffers for covariance adaptation (per chain) ----------
    acc_buffers = [np.empty((0, D), dtype=np.float64) for _ in range(C)]

    # ---------- per-epoch outputs ----------
    slim_per_epoch = []           # list of SlimInfo or SlimInfoPS
    scales_per_epoch = []
    covs_per_epoch   = []
    swap_rate_epoch  = []         # scalar (mean over walkers & edges)
    accept_rate_epoch= []         # (C) mean over walkers

    num_of_proposals = 5
    # ---------- main loop ----------
    for epoch in trange(cfg.m_epochs, desc="Adaptive PT (ensemble)", unit="epoch"):
        covs_j = jnp.asarray(covs)  # (C,D,D)

        # per-(C,W) component indices for this epoch (categorical)
        comp_idx = np.empty((C, W), dtype=np.int32)
        # sample with numpy on host (fast, simple)
        for c in range(C):
            for w in range(W):
                comp_idx[c, w] = np.random.choice(num_of_proposals, p=Wts[c, w])
        comp_idx_j = jnp.asarray(comp_idx)  # (C,W)

        key, subkey = random.split(key)

        # ---------- run one epoch ----------
        if not product_space:
            # single-model, ensemble
            state, slim = run_epoch_device_fast_ensemble(
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
                means=jnp.asarray(means),
                fold_mask=fold_mask,
                period=period,
                do_swaps=True,
                stretch_a=2.0,
                nu=5.0,
            )
            # aggregate simple epoch stats
            swap_rate_epoch.append(float(slim.swap_rate_per_edge.mean()))
            accept_rate_epoch.append(np.asarray(slim.accept_rate_per_c))
            # pool accepted points per chain across walkers
            for c in range(C):
                for w in range(W):
                    pts = slim.accepted_points[c][w]
                    if pts.size:
                        acc_buffers[c] = np.vstack([acc_buffers[c], np.asarray(pts)])
        else:
            # product-space, ensemble
            ps_sample = psi3_sample if psi3_sample is not None else _psi3_uniform_sample
            ps_logpdf = psi3_logpdf if psi3_logpdf is not None else _psi3_uniform_logpdf

            state, slim = run_epoch_device_fast_M23_ensemble(
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
                nu=5.0,
            )
            # epoch stats & buffers (pool both M2/M3 accepted)
            swap_rate_epoch.append(float(slim.swap_rate_per_edge.mean()))
            # accept per chain averaged over walkers already in SlimInfoPS
            accept_rate_epoch.append(np.asarray(slim.accept_rate_per_c))
            for c in range(C):
                for w in range(W):
                    p2 = slim.accepted_points_M2[c][w]
                    p3 = slim.accepted_points_M3[c][w]
                    if p2.size:
                        acc_buffers[c] = np.vstack([acc_buffers[c], np.asarray(p2)])
                    if p3.size:
                        acc_buffers[c] = np.vstack([acc_buffers[c], np.asarray(p3)])

        # keep per-epoch artifacts
        slim_per_epoch.append(slim)
        scales_per_epoch.append(np.asarray(scales_small))
        covs_per_epoch.append(np.asarray(covs))

        # ---------- adapt scales (per chain; mean over walkers) ----------
        acc_rate_c = np.asarray(accept_rate_epoch[-1])  # (C,)
        scales_small = jnp.clip(
            jnp.exp(jnp.log(scales_small) + cfg.eta * (acc_rate_c - cfg.target_accept)),
            cfg.scale_min,
            cfg.scale_max,
        )
        # you can also adapt line/big if desired; here we keep them bounded:
        scales_line = jnp.clip(scales_line, cfg.scale_min, cfg.scale_max)
        scales_big  = jnp.clip(scales_big,  cfg.scale_min, cfg.scale_max)

        # ---------- adapt covariances (per chain) ----------
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
                mu_hat = acc_buffers[c].mean(axis=0) if acc_buffers[c].size else np.zeros(D)
            new_covs.append(cov_c)
            new_means.append(mu_hat)
        covs = np.stack(new_covs, axis=0)
        means = np.stack(new_means, axis=0)

        # optional: quiet progress line
        if (epoch + 1) % 10 == 0 or epoch == cfg.m_epochs - 1:
            try:
                from tqdm import tqdm as _tqdm
                _tqdm.write(
                    f"Epoch {epoch + 1}/{cfg.m_epochs} | "
                    f"swap_rate={swap_rate_epoch[-1]:.3f} | "
                    f"mean_acc={float(acc_rate_c.mean()):.3f}"
                    + (f" | p(z=1)={float(state.z.mean()):.3f}" if product_space else "")
                )
            except Exception:
                pass

    # ---------- pack output ----------
    out = {
        "slim_per_epoch": slim_per_epoch,                # list of SlimInfo / SlimInfoPS
        "scales_per_epoch": np.stack(scales_per_epoch),  # (E, C)
        "covs_per_epoch":   np.stack(covs_per_epoch),    # (E, C, D, D)
        "swap_rate_per_epoch": np.array(swap_rate_epoch),
        "accept_rate_per_epoch": np.stack(accept_rate_epoch),  # (E, C)
    }
    return state, out

