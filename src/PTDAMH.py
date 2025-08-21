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
  your existing `Proposals.build_general_mixture_components_per_chain`.
- After each epoch, per-chain covariance is adapted from a rolling buffer or
  from all seen post-swap states, with shrinkage + jitter for numerical safety.
- We record for every step: proposed points, their log-likelihood, the chosen
  component index, acceptance mask, and swap decisions.

This module expects your `Proposals.py` next to it. It does **not** modify
`Proposals.py`. If you want to plug in different proposals, change only the
`build_proposals(...)` function.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.util import info
from typing import Callable, NamedTuple, Tuple, Sequence

import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import numpy as np
from jax.scipy.linalg import solve_triangular
from tqdm import tqdm
from tqdm import trange


import Proposals  # your file with proposal factories

from tqdm import trange, tqdm


import Proposals  # your file with proposal factories


# ------------------------- Utilities -------------------------

def temperature_ladder(
    n_temps: int = 50,
    T_min: float = 1.0,
    T_max: float = 100.0,
    kind: str = "geom",
    cold_dense: bool = False,
    power: float = 2.0,
):
    """Construct a sequence of temperatures between ``T_min`` and ``T_max``.

    Parameters
    ----------
    n_temps:
        Number of temperatures in the ladder.
    T_min:
        Coldest temperature (typically 1.0).
    T_max:
        Hottest temperature.
    kind:
        Ladder scheme. Currently only ``"geom"`` (geometric spacing in ``T``)
        is supported.
    cold_dense:
        If ``True``, densify near ``T_min`` by warping the index.
    power:
        Exponent used for warping when ``cold_dense`` is ``True``. Larger values
        place more points near the cold chain.

    Returns
    -------
    jnp.ndarray
        Array of temperatures of length ``n_temps``.

    Raises
    ------
    ValueError
        If ``kind`` specifies an unsupported ladder scheme.
    """

    if kind != "geom":
        raise ValueError(
            f"Unsupported ladder kind: {kind!r}. Only 'geom' is implemented."
        )

    i = jnp.arange(n_temps, dtype=jnp.float32)
    if cold_dense:
        # warp indices to cluster more temperatures near i=0 (cold end)
        i = ((i / (n_temps - 1)) ** power) * (n_temps - 1)
    r = (T_max / T_min) ** (1.0 / (n_temps - 1))
    T = T_min * (r ** i)
    return T


def _empirical_cov(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[0] <= 1:
        d = x.shape[1]
        return np.eye(d, dtype=x.dtype) * 1e-3
    return np.cov(x, rowvar=False, ddof=ddof)


def _shrink_spd(cov_hat: np.ndarray, shrink: float = 0.1, jitter: float = 1e-6) -> np.ndarray:
    cov_hat = np.asarray(cov_hat)
    d = np.diag(np.diag(cov_hat))
    cov = (1.0 - shrink) * cov_hat + shrink * d
    cov = 0.5 * (cov + cov.T)
    eps = float(max(np.max(np.diag(cov)), 1.0)) * jitter
    cov += eps * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov

def _circular_mean(vals, period=1.0):
    ang = 2.0*np.pi*(vals/period)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    mu_ang = np.arctan2(s, c) % (2*np.pi)
    return mu_ang * (period/(2*np.pi))

def _wrapped_diff(vals, mu, period=1.0):
    d = vals - mu
    return (d + 0.5*period) % period - 0.5*period

def _empirical_cov_wrapped(samples, fold_idx=(), period=1.0, ddof=1):
    X = np.asarray(samples)
    T, D = X.shape
    if T - ddof <= 0:
        return np.eye(D), X.mean(axis=0) if T else np.zeros(D)
    mu = X.mean(axis=0)
    dev = X - mu
    if fold_idx:
        fold_idx = np.asarray(fold_idx, dtype=int)
        for i in fold_idx:
            mu_i = _circular_mean(X[:, i], period=period)
            mu[i] = mu_i
            dev[:, i] = _wrapped_diff(X[:, i], mu_i, period=period)
        # re-center non-periodic with updated mu
        nonper = np.setdiff1d(np.arange(D), fold_idx)
        if nonper.size:
            dev[:, nonper] = X[:, nonper] - mu[nonper]
    C = (dev.T @ dev) / (T - ddof)
    return C, mu


# ------------------------- State & info -------------------------
class PTState(NamedTuple):
    thetas: jnp.ndarray        # (C, D)
    log_probs: jnp.ndarray     # (C,)  (log-target at current thetas)
    temperatures: jnp.ndarray  # (C,)
    n_accepted: jnp.ndarray    # (C,)
    n_swaps: jnp.ndarray       # (C-1,) counters of accepted swaps per edge (optional)
    n_swap_attempts: jnp.ndarray  # (C-1,) total swap attempts per edge


class StepInfo(NamedTuple):
    thetas_prop: jnp.ndarray      # (C, D)
    logprob_prop: jnp.ndarray     # (C,)
    accepted: jnp.ndarray         # (C,) bool
    comp_idx: jnp.ndarray         # (C,) int in [0..K-1]
    swap_decisions: jnp.ndarray   # (C-1,) bool


# ------------------------- Parallel tempering swap -------------------------
# @jax.jit
# def parallel_tempering_swap(key, temperatures: jnp.ndarray,
#                             thetas: jnp.ndarray, log_probs: jnp.ndarray):
#     """Single adjacent-swap sweep (even-odd could be added if desired).

#     Returns updated (thetas, log_probs), plus boolean decisions per edge.
#     """
#     C = temperatures.shape[0]
#     betas = 1.0 / temperatures
#     # Δβ * ΔlogL between neighbors (using current log_probs ~ log target)
#     dlog = (betas[1:] - betas[:-1]) * (log_probs[1:] - log_probs[:-1])

#     key, ukey = random.split(key)
#     u = jnp.log(random.uniform(ukey, shape=(C - 1,)))
#     do_swap = dlog >= u

#     def swap_edge(i, carry):
#         th, lp = carry
#         def yes():
#             th1 = th.at[i].set(th[i + 1])
#             th1 = th1.at[i + 1].set(th[i])
#             lp1 = lp.at[i].set(lp[i + 1])
#             lp1 = lp1.at[i + 1].set(lp[i])
#             return th1, lp1
#         return lax.cond(do_swap[i], yes, lambda: (th, lp))

#     thetas2, logp2 = lax.fori_loop(0, C - 1, swap_edge, (thetas, log_probs))
#     return key, thetas2, logp2, do_swap


def _apply_swaps_vec(arr, i, j, accept):
    """
    arr: (C, ...) values to swap
    i,j: (K,) pair indices
    accept: (K,) booleans
    """
    ai, aj = arr[i], arr[j]
    if arr.ndim == 1:
        arr = arr.at[i].set(jnp.where(accept, aj, ai))
        arr = arr.at[j].set(jnp.where(accept, ai, aj))
    else:
        arr = arr.at[i].set(jnp.where(accept[:, None], aj, ai))
        arr = arr.at[j].set(jnp.where(accept[:, None], ai, aj))
    return arr

@jax.jit
def parallel_tempering_swap(key, temperatures, thetas, log_probs, *, return_debug=False):
    """
    One PT swap sweep (even or odd, chosen at random).
    temperatures: (C,)  *absolute T* (T0=1 is cold).  β = 1/T is computed inside.
    thetas:       (C,D)
    log_probs:    (C,)  untempered log π(θ) (includes prior!), not scaled by T.
    returns: key, thetas_new, log_probs_new, swap_decisions (C-1,) bool [, debug dict]
    """
    C = thetas.shape[0]
    beta = 1.0 / jnp.asarray(temperatures)
    n_edges = C - 1

    # Build even/odd adjacent index sets; pad odd to same static length
    i_even = jnp.arange(0, n_edges, 2, dtype=jnp.int32)   # Ke = ceil(n_edges/2)
    i_odd  = jnp.arange(1, n_edges, 2, dtype=jnp.int32)   # Ko = floor(n_edges/2)
    Ke = i_even.shape[0]
    Ko = i_odd.shape[0]
    pad = Ke - Ko
    i_odd_padded = jnp.concatenate([i_odd, -jnp.ones((pad,), dtype=jnp.int32)], axis=0)
    valid_even = jnp.ones((Ke,), dtype=bool)
    valid_odd  = jnp.arange(Ke) < Ko

    # RNG: advance key and get subkeys
    key, k_par = random.split(key)
    key, k_u   = random.split(key)
    parity = random.bernoulli(k_par)  # False->even, True->odd

    def pick_even():
        return i_even, valid_even
    def pick_odd():
        return i_odd_padded, valid_odd

    i_raw, valid = lax.cond(parity, pick_odd, pick_even)  # both (Ke,)
    # Safe indices for padded slots (map invalid to 0; we’ll mask later)
    i = jnp.where(valid, i_raw, jnp.zeros_like(i_raw))
    j = i + 1

    # Correct MH exponent for swaps: Δ = (β_i - β_j) * (lp_j - lp_i)
    delta = (beta[i] - beta[j]) * (log_probs[j] - log_probs[i])     # (Ke,)
    delta = jnp.where(valid, delta, -jnp.inf)                        # mask padded
    ulog  = jnp.log(random.uniform(k_u, shape=delta.shape))
    accept_sel = ulog < delta                                        # (Ke,)

    # Apply swaps simultaneously on the selected (non-overlapping) pairs
    thetas_new = _apply_swaps_vec(thetas,    i, j, accept_sel)
    logp_new   = _apply_swaps_vec(log_probs, i, j, accept_sel)

    # Raster of decisions over all edges (C-1,)
    raster = jnp.zeros((n_edges,), dtype=bool).at[i].set(jnp.where(valid, accept_sel, False))

    if return_debug:
        dbg = {"delta": delta, "ulog": ulog, "pairs_i": i, "pairs_j": j, "accept_sel": accept_sel, "parity": parity}
        return key, thetas_new, logp_new, raster, dbg
    return key, thetas_new, logp_new, raster


# ------------------------- Adaptive driver (m epochs) -------------------------
@dataclass
class AdaptConfig:
    m_epochs: int = 6
    N_steps: int = 500
    target_accept: float = 0.234
    eta: float = 0.05               # Robbins–Monro adaptation rate
    scale_init: float = 1.0
    scale_min: float = 0.1
    scale_max: float = 10.0
    shrink: float = 0.1
    jitter: float = 1e-6
    kappa_line: float = 3.0
    beta_base: float = 0.3
    beta_temp_scale: bool = True
    cov_mode: str = "rolling"      # "rolling" or "all_states"
    window_size: int = 10_000       # rolling window size
    downsample_every: int = 2       # store every k-th post-swap state into buffers


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

    def add(self, state: PTState, info: StepInfo, temps, scales, covs):
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

    def pack(self):
        out = {
            "thetas_prop": None if not self.thetas_prop else np.stack(self.thetas_prop, axis=0),
            "logprob_prop": None if not self.logprob_prop else np.stack(self.logprob_prop, axis=0),
            "accepted": None if not self.accepted else np.stack(self.accepted, axis=0),
            "comp_idx": None if not self.comp_idx else np.stack(self.comp_idx, axis=0),
            "swap_decisions": None if not self.swap_decisions else np.stack(self.swap_decisions, axis=0),
            "thetas_state": None if not self.thetas_state else np.stack(self.thetas_state, axis=0),
            "logprob_state": None if not self.logprob_state else np.stack(self.logprob_state, axis=0),
            "temperatures": None if not self.temperatures else np.stack(self.temperatures, axis=0),
            "scales_per_epoch": None if not self.scales_per_epoch else np.stack(self.scales_per_epoch, axis=0),
            "swap_rate_per_epoch": None if not self.swap_rate_per_epoch else np.array(self.swap_rate_per_epoch),
            "accept_rate_per_epoch": None if not self.accept_rate_per_epoch else np.stack(self.accept_rate_per_epoch, axis=0),
            "covs_per_epoch": None if not self.covs_per_epoch else np.stack(self.covs_per_epoch, axis=0),
        }
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
            out["dataset"] = dict(X=X, y_logprob=y, accepted=acc, comp_idx=comp,
                                   chain=chain, temperature=temperature)
        return out

    def finalize(self):
        return self.pack()



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

def _batched_logprob_chunked_fn(log_prob_fn_single, C: int, D: int, chunk: int):
    """Return jitted (X:(C,D)->(C,)) using pad+reshape+vmap with given chunk size."""
    chunk = max(1, min(int(chunk), C))
    n_chunks = (C + chunk - 1) // chunk
    C_pad = n_chunks * chunk
    pad_rows = C_pad - C
    f1 = jax.vmap(log_prob_fn_single)
    f2 = jax.vmap(f1)

    @jax.jit
    def run(X):  # (C, D)
        if pad_rows:
            pad = jnp.zeros((pad_rows, D), dtype=X.dtype)
            Xp = jnp.concatenate([X, pad], axis=0)
        else:
            Xp = X
        Yp = f2(Xp.reshape((n_chunks, chunk, D))).reshape((C_pad,))
        return Yp[:C]
    return run

def _fold_params(x: jnp.ndarray, fold_mask: jnp.ndarray | None, period: float) -> jnp.ndarray:
    """
    Fold selected dimensions into [0, period). fold_mask: (D,) with {0,1}; None -> no-op.
    Works inside jit without Python loops.
    """
    if fold_mask is None:
        return x
    x_mod = jnp.mod(x, period)
    # broadcast mask to (C,D)
    m = fold_mask[None, :]
    return x * (1.0 - m) + x_mod * m

########## Move to Proposals later #################

def _build_epoch_components(
    covs: jnp.ndarray,       # (C, D, D)
    scale_small: jnp.ndarray,# (C,)
    scale_line: jnp.ndarray, # (C,)
    scale_big: jnp.ndarray,  # (C,)
    jitter: float = 1e-9,
):
    # symmetrize + jitter
    C, D, _ = covs.shape
    I = jnp.eye(D)[None, :, :]
    sym = 0.5 * (covs + jnp.swapaxes(covs, -1, -2)) + jitter * I

    # per-chain Cholesky (batched)
    L_chol = jnp.linalg.cholesky(sym)                     # (C, D, D), lower

    # per-chain eigen (batched) for eigen-line
    S, U = jnp.linalg.eigh(sym)                           # (C, D), (C, D, D)
    S = jnp.clip(S, 1e-12, None)
    axis_logits = jnp.log(jnp.sqrt(S) + 1e-12)            # (C, D)

    fullcov = {
        "L_chol": L_chol,
        "scale_small": scale_small,
        "scale_big":   scale_big,
    }
    eigenline = {
        "U": U, "S": S, "axis_logits": axis_logits, "scale": scale_line
    }
    return fullcov, eigenline

def _propose_fullcov(key, x, L, scale):
    # x: (C,D), L: (C,D,D), scale: (C,)
    C, D = x.shape
    cd_const = 2.38 / jnp.sqrt(D) * 0.5 ## last one is empirical
    z = random.normal(key, x.shape)
    step = jnp.einsum('cij,cj->ci', L, z)        # (C, D)
    # return x + scale[:, None] * step
    return x + cd_const * step

def _propose_eigenline(key_axis, key_noise, x, U, S,  scale, axis_logits=None):
    """
    Propose a step along a single, randomly chosen eigenvector for each chain.
    The eigenvector can be chosen uniformly or weighted by logits.
    """
    C, D = x.shape
    if axis_logits is None:
        # Uniformly choose an axis for each chain if no logits are provided
        axes = random.randint(key_axis, shape=(C,), minval=0, maxval=D)
    else:
        # Choose axis based on provided logits (e.g., proportional to sqrt(eigenvalue))
        axes = vmap(lambda lg, k: random.categorical(k, lg))(axis_logits, random.split(key_axis, C))  # (C,)

    Usel = jnp.take_along_axis(U, axes[:, None, None], axis=2).squeeze(-1)   # (C, D)
    Ssel = jnp.take_along_axis(S, axes[:, None], axis=1).squeeze(-1)         # (C,)
    r = random.normal(key_noise, (C,))
    # step = (scale * jnp.sqrt(Ssel) * r)[:, None] * Usel
    step = (jnp.sqrt(Ssel) * r)[:, None] * Usel
    return x + step

def _propose_student_t(key_norm, key_gamma, x, L, scale, nu=5.0):
    """
    x:     (C, D)
    L:     (C, D, D)   Cholesky of per-chain covariance
    nu:    (C,) or scalar  degrees of freedom
    scale: (C,)         step scale per chain
    returns: (C, D)
    """
    C, D = x.shape
    cd_const = 2.38 / jnp.sqrt(D) * 0.5 ## last one is empirical
    z = random.normal(key_norm, (C, D))                               # N(0, I)
    g = random.gamma(key_gamma, a=(nu if jnp.ndim(nu)==0 else nu[:,None]) / 2.0,
                     shape=(C, 1)) * (2.0 / (nu if jnp.ndim(nu)==0 else nu[:,None]))
    t = z / jnp.sqrt(g)                                               # iid t_nu
    # step = cd_const * (scale[:, None]) * jnp.einsum('cij,cj->ci', L, t)
    step = cd_const * jnp.einsum('cij,cj->ci', L, t)
    return x + step

def _propose_pcn(key, x, mu, L, scale, beta=0.3):
    """
    x, mu:   (C, D)
    L:       (C, D, D)   Cholesky of cov
    beta:    (C,) or scalar
    scale:   (C,)            runtime tuning (multiplies beta)
    returns: (C, D)
    """
    # b will have shape (C,) due to broadcasting of scalar beta with scale
    # b = beta * scale
    # b = beta
    C, D = x.shape
    b = (beta if jnp.ndim(beta)==0 else beta) * (scale if jnp.ndim(scale)==0 else scale)  # (C,)
    alpha = jnp.sqrt(1.0 - b**2)

    # Generate correlated noise N(0, Σ) for each chain
    # z ~ N(0, I), eps = L @ z
    eps = jnp.einsum('cij,cj->ci', L, random.normal(key, x.shape))

    # pCN update: x_new = mu + alpha*(x - mu) + b*eps
    # The implementation is an equivalent rearrangement:
    # x_new = alpha*x + b*eps + (1-alpha)*mu
    return alpha[:, None] * x + b[:, None] * eps + (1.0 - alpha)[:, None] * mu

def _pcn_logq_delta(x, y, mu, L, scale, beta=0.3):
    """
    Δlogq = log q(x|y) - log q(y|x)  for pCN with cov = (b^2) Σ, mean = a·state + (1-a)·mu.
    x,y,mu: (C,D); L: (C,D,D); beta,scale: (C,) or scalars
    returns: (C,)
    """
    b = (beta if jnp.ndim(beta)==0 else beta) * (scale if jnp.ndim(scale)==0 else scale)  # (C,)
    b2 = jnp.clip(b*b, 1e-12, 1.0 - 1e-12)
    a  = jnp.sqrt(1.0 - b2)    # (C,)                                                               

    m_x = a[:, None]*x + (1.0 - a)[:, None]*mu
    m_y = a[:, None]*y + (1.0 - a)[:, None]*mu

    # whitened residuals via triangular solve per chain
    def maha_sq(Lc, v):
        w = solve_triangular(Lc, v, lower=True)
        return jnp.sum(w*w)

    maha_y_given_x = vmap(maha_sq)(L, (y - m_x))   # (C,)
    maha_x_given_y = vmap(maha_sq)(L, (x - m_y))   # (C,)

    return -0.5 * (maha_x_given_y - maha_y_given_x) / (b*b + 1e-32)

################################################################################
# --- core epoch ---

def run_epoch_device_fast(
    key,
    init_state: PTState,
    log_prob_fn_single,              # (D,) -> ()
    temperatures: jnp.ndarray,       # (C,)
    covs: jnp.ndarray,               # (C, D, D)
    # U: jnp.ndarray,                  # (C, D, D)
    # S: jnp.ndarray,                  # (C, D)
    scale_small: jnp.ndarray,        # (C,)
    scale_line: jnp.ndarray,         # (C,)
    scale_big: jnp.ndarray,          # (C,)
    comp_idx: jnp.ndarray,           # (C,) fixed for epoch in {0,1,2}
    n_steps: int,
    *,
    lik_chunk: int = 32, 
    means: jnp.ndarray,              # <-- (C,D) NEW (per-temperature means) 
    fold_mask: jnp.ndarray | None = None,  # (D,) with 0/1; None -> no folding
    period: float = 1.0,
    do_swaps: bool = False
):
    """
    Fully device-resident epoch with symmetric proposals and fixed component per chain.
    No logq terms; MH ratio uses only tempered log-likelihood difference.
    Returns final PTState and StepInfo (stacked over steps).
    """
    C, D = init_state.thetas.shape
    fullcov, eigenline = _build_epoch_components(covs, scale_small, scale_line, scale_big)
    batched_lp = _batched_logprob_chunked_fn(log_prob_fn_single, C, D, lik_chunk)

    beta = 0.3
    nu = 5.0
    def body(carry, key_t):
        th, lp = carry  # (C, D), (C,)
        k0, k1, k2, k3, k4, kU, kS = random.split(key_t, 7)


        # 3 components
        # prop0 = _propose_fullcov(k0, th, fullcov["L_chol"], fullcov["scale_small"])   # (C, D)
        prop0 = _propose_student_t(k0, k1, th, fullcov["L_chol"], fullcov["scale_small"], nu=nu)  # (C, D)
        prop1 = _propose_eigenline(k2, k3, th, eigenline["U"], eigenline["S"],
                                   eigenline["scale"], axis_logits=None) # (C, D)
        prop2 = _propose_fullcov(k4, th, fullcov["L_chol"], fullcov["scale_big"])     # (C, D)
        # prop2 = _propose_pcn(k4, th, means, fullcov["L_chol"], fullcov["scale_big"], beta=beta)


        props_all = jnp.stack([prop0, prop1, prop2], axis=0)                      # (3, C, D)
        proposals = props_all[comp_idx, jnp.arange(C), :]                         # (C, D)
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
        lp_new = jnp.where(accept,          prop_lp,     lp)

        # print (f'debug, lp = {lp}, prop_lp = {prop_lp}, accept = {accept}, comp_idx = {comp_idx}')

         # ================= DEBUG START =================
        delta = jnp.stack([prop_lp, lp, prop_lp - lp], axis=-1)                                          # (C,)
        log_alpha = (prop_lp - lp) / temperatures                              # (C,)
        u_log = jnp.log(random.uniform(kU, (C,)))                     # (C,)
        accept = u_log < log_alpha                                    # (C,)
        bad_accept = accept #& ~(u_log < log_alpha)                    # (C,) must be all False
        # ================= DEBUG END ===================

        # th_new = jnp.where(accept[:, None], proposals, th)
        # lp_new = jnp.where(accept,          prop_lp,     lp)

        # PT swap (adjacent)
        # _, th_sw, lp_sw, swap_dec = parallel_tempering_swap(kS, temperatures, th_new, lp_new)
        if do_swaps:
            _, th_sw, lp_sw, swap_dec = parallel_tempering_swap(kS, temperatures, th_new, lp_new)
        else:
            th_sw, lp_sw = th_new, lp_new
            swap_dec = jnp.zeros((C - 1,), dtype=bool)

        info_step = (proposals, prop_lp, accept, comp_idx, swap_dec, th_sw,
                     delta, log_alpha, u_log, bad_accept)
        return (th_sw, lp_sw), info_step

    keys = random.split(key, n_steps)
    # (th_f, lp_f), (props, prop_lps, accepts, comp_idxs, swaps) = lax.scan(
    #     body, (init_state.thetas, init_state.log_probs), keys
    # )
    (th_f, lp_f), (props, prop_lps, accepts, comp_idxs, swaps, th_history,
                   deltas, log_alphas, u_logs, bad_acc) = lax.scan(
        body, (init_state.thetas, init_state.log_probs), keys
    )

    final_state = PTState(
        thetas=th_f,
        log_probs=lp_f,
        temperatures=init_state.temperatures,
        n_accepted=init_state.n_accepted + accepts.sum(axis=0).astype(jnp.int32),
        n_swaps=init_state.n_swaps + swaps.sum(axis=0).astype(jnp.int32),
        n_swap_attempts=init_state.n_swap_attempts + jnp.full_like(init_state.n_swaps, n_steps),
    )

    info = StepInfo(
        thetas_prop=props,            # (T, C, D)
        logprob_prop=prop_lps,        # (T, C)
        accepted=accepts,             # (T, C)
        comp_idx=comp_idxs,           # (T, C) replicates comp_idx per step
        swap_decisions=swaps,         # (T, C-1)
    )
    debug = {
        "delta": deltas,              # (T, C)
        "log_alpha": log_alphas,      # (T, C)
        "u_log": u_logs,              # (T, C)
        "bad_accept": bad_acc,        # (T, C)
    }
    return final_state, info, th_history, debug

# --- top-level adaptive runner ---

def run_adaptive_pt_device_fast(
    key,
    initial_thetas: jnp.ndarray,      # (C, D)
    temperatures: jnp.ndarray,        # (C,)
    log_prob_fn_single,               # (D,) -> ()
    base_cov: np.ndarray,             # (D, D)
    *,
    fold_idx=(),
    period: float = 1.0,
    weights: np.ndarray | None = None,  # (C, up to 3)
    cfg: AdaptConfig = AdaptConfig(),
    lik_chunk: int = 32,
    big_scale_factor: float = 3.0,
):
    """
    End-to-end PT where each epoch is a single device loop (scan).
    - Samples comp_idx per chain from weights once per epoch
    - Symmetric proposals (no logq)
    - Updates covariances from accepted proposals of the epoch
    - Records to InfoAccumulator (same interface you use)
    """
    C, D = initial_thetas.shape
    acc_buffers = [np.empty((0, D), dtype=np.float64) for _ in range(C)] 
    means = np.asarray(initial_thetas)               # (C,D)


    # Initialize covs/eigs
    covs = np.tile(np.asarray(base_cov)[None, :, :], (C, 1, 1))  # host np
    scales_small = jnp.full((C,), float(cfg.scale_init))
    scales_line  = jnp.full((C,), float(cfg.scale_init) * cfg.kappa_line)
    scales_big   = jnp.full((C,), float(cfg.scale_init) * big_scale_factor)

    # Fold mask
    fold_mask = None
    if len(fold_idx) > 0:
        mask = np.zeros((D,), dtype=np.float32)
        mask[np.asarray(fold_idx, dtype=int)] = 1.0
        fold_mask = jnp.asarray(mask)

    # Initial state
    batched_init = _batched_logprob_chunked_fn(log_prob_fn_single, C, D, max(1, min(C, lik_chunk)))
    logp0 = batched_init(initial_thetas)
    state = PTState(
        thetas=initial_thetas,
        log_probs=logp0,
        temperatures=temperatures,
        n_accepted=jnp.zeros((C,), dtype=jnp.int32),
        n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
        n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
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

    for epoch in trange(cfg.m_epochs, desc='Adaptive PT (device)', unit='epoch'):
        # Eig on host (stable), then to device
        # U_list, S_list = [], []
        # for c in range(C):
        #     Sc, Uc = np.linalg.eigh(covs[c])
        #     Sc = np.clip(Sc, 1e-12, None)
        #     U_list.append(Uc)
        #     S_list.append(Sc)
        # U_j = jnp.asarray(np.stack(U_list, axis=0))   # (C, D, D)
        # S_j = jnp.asarray(np.stack(S_list, axis=0))   # (C, D)
        covs_j = jnp.asarray(covs)                    # (C, D, D)

        # Sample component per chain for this epoch (host)
        comp_idx = np.array([np.random.choice(3, p=W[c]) for c in range(C)], dtype=np.int32)
        comp_idx_j = jnp.asarray(comp_idx)

        # Run one device-resident epoch
        key, subkey = random.split(key)
        state, info, th_history, debug = run_epoch_device_fast(
            subkey, state, log_prob_fn_single, temperatures,
            covs_j, scales_small, scales_line, scales_big,
            comp_idx_j, cfg.N_steps, lik_chunk=lik_chunk, means=means,
            fold_mask=fold_mask, period=period, do_swaps= True
        )



        props = np.asarray(info.thetas_prop)              # (T_steps, C, D)
        accs  = np.asarray(info.accepted, dtype=bool)     # (T_steps, C)

        for c in range(C):
            Pc = props[accs[:, c], c, :]                 # all accepted at temperature slot c
            if Pc.size:                                  # append across epochs (ALL accepted)
                acc_buffers[c] = np.vstack([acc_buffers[c], Pc])

        # Adapt scales by per-chain acceptance this epoch
        acc_rate = jnp.asarray(info.accepted).mean(axis=0).astype(jnp.float64)
        scales_small = jnp.clip(
            jnp.exp(jnp.log(scales_small) + cfg.eta * (acc_rate - cfg.target_accept)),
            cfg.scale_min, cfg.scale_max,
        )
        # keep line/big fixed or clip
        scales_line = jnp.clip(scales_line, cfg.scale_min, cfg.scale_max)
        scales_big  = jnp.clip(scales_big,  cfg.scale_min, cfg.scale_max)
        # scales_line = 1.0
        # scales_small = 1.0
        # scales_big = 1.0

        # Covariance update from accepted proposals (host)
        states_hist = np.asarray(th_history)   # (T, C, D)
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
                cov_c = covs[c]                            # keep previous (or base_cov)
                mu_hat = acc_buffers[c].mean(axis=0) if acc_buffers[c].size else np.zeros(D)
            new_covs.append(cov_c)
            new_means.append(mu_hat)

        covs = np.stack(new_covs, axis=0)                  # (C, D, D)
        means = np.stack(new_means, axis=0)      # if you need it elsewhere   

        # Record epoch results
        info_accum.add(state, info, np.asarray(temperatures), np.asarray(scales_small), covs)

        last_debug = {k: np.asarray(v) for k, v in debug.items()}

        # Optional: progress diagnostics, print every 10 epochs or on the last one
        if (epoch + 1) % 10 == 0 or epoch == cfg.m_epochs - 1:
            try:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"Epoch {epoch + 1}/{cfg.m_epochs} | "
                            f"swap_rate={float(np.mean(np.asarray(info.swap_decisions))):.3f} | "
                            f"mean_acc={float(np.mean(np.asarray(info.accepted))):.3f}")
            except Exception:
                pass
    out = info_accum.finalize()
    out["last_debug"] = last_debug
    return state, out

###################################################
############ updates for the product space ########
##################################################


# --- indices for the 3 equal-sized signal blocks ---
def make_signal_indices(Npar_src, D, with_rest=True):
    i1 = np.arange(0, Npar_src, dtype=np.int32)
    i2 = np.arange(Npar_src, 2*Npar_src, dtype=np.int32)
    i3 = np.arange(2*Npar_src, 3*Npar_src, dtype=np.int32)
    if with_rest and D > 3*Npar_src:
        rest = np.setdiff1d(np.arange(D, dtype=np.int32), np.concatenate([i1,i2,i3]))
    else:
        rest = np.empty((0,), dtype=np.int32)
    return i1, i2, i3, rest

def _gather_cols(X, idx):  # X:(C,D), idx:(k,) -> (C,k)
    idx = jnp.asarray(idx, jnp.int32)
    return jnp.take(X, idx, axis=1) if idx.size else jnp.zeros((X.shape[0], 0), X.dtype)

# Provide your single-point log-liks for the two models:
# loglik_M2_single(theta12[, rest]) and loglik_M3_single(theta123[, rest])
# Below we just vmaps them and select per-chain by z.
def make_logpost_M23(loglik_M2_single, loglik_M3_single,
                     idx1, idx2, idx3, idx_rest=(),
                     log_prior_z=(0.0, 0.0)):
    f2 = jax.jit(jax.vmap(loglik_M2_single))
    f3 = jax.jit(jax.vmap(loglik_M3_single))
    lp0, lp1 = map(float, log_prior_z)

    @jax.jit
    def logpost(X, z):
        X1 = _gather_cols(X, idx1)
        X2 = _gather_cols(X, idx2)
        X3 = _gather_cols(X, idx3)
        R  = _gather_cols(X, idx_rest)
        lp2 = f2(jnp.concatenate([X1, X2, R], axis=1))
        lp3 = f3(jnp.concatenate([X1, X2, X3, R], axis=1))
        return jnp.where(z.astype(bool), lp3 + lp1, lp2 + lp0)  # (C,)
    return logpost


@jax.jit
def update_z_with_rejuv(key, X, z,
                        loglik_M2_single, loglik_M3_single,
                        idx1, idx2, idx3, idx_rest=(),
                        log_prior_z=(0.0,0.0)):
    C, D = X.shape
    k_rejuv, k_flip = random.split(key)

    # Rejuvenate θ3 ~ Uniform[0,1] when z==0 (cheap, normalized ψ on your box)
    if len(idx3):
        U = random.uniform(k_rejuv, (C, len(idx3)))
        X = X.at[:, jnp.asarray(idx3)].set(jnp.where(z[:,None]==0, U, X[:, jnp.asarray(idx3)]))

    # Compute both model log-liks
    f2 = jax.vmap(loglik_M2_single); f3 = jax.vmap(loglik_M3_single)
    X1 = _gather_cols(X, idx1); X2 = _gather_cols(X, idx2); X3 = _gather_cols(X, idx3); R=_gather_cols(X, idx_rest)
    lp2 = f2(jnp.concatenate([X1, X2, R], axis=1))
    lp3 = f3(jnp.concatenate([X1, X2, X3, R], axis=1))
    lp0, lp1 = map(float, log_prior_z)

    logits = (lp3 + lp1) - (lp2 + lp0)         # logit p(z=1 | θ)
    p1 = jax.nn.sigmoid(logits)
    z_new = random.bernoulli(k_flip, p1).astype(jnp.int32)

    # Consistent logpost with the new z
    lp_new = jnp.where(z_new.astype(bool), lp3 + lp1, lp2 + lp0)
    return X, z_new, lp_new, p1

# helper from your swap impl
def _apply_swaps_vec(arr, i, j, accept):
    ai, aj = arr[i], arr[j]
    return arr.at[i].set(jnp.where(accept, aj, ai)).at[j].set(jnp.where(accept, ai, aj))

def run_epoch_device_fast_M23(
    key, init_state,                            # PTState must include .z (C,) int32
    temperatures,                               # (C,)
    loglik_M2_single, loglik_M3_single,         # model-specific single-point fns
    idx1, idx2, idx3, idx_rest=(),
    # ... all your existing args for proposals ...
    model_update_stride: int = 5,               # e.g., update z every 5 steps
    log_prior_z=(0.0, 0.0),
    # ...
):
    C, D = init_state.thetas.shape
    logpost = make_logpost_M23(loglik_M2_single, loglik_M3_single, idx1, idx2, idx3, idx_rest, log_prior_z)

    # precompute stride mask (static under jit if you pass it as xs)
    do_model = jnp.arange(n_steps) % model_update_stride == 0  # (T,)

    def body(carry, xs):
        (th, lp, z), (key_t, comp_idx_t, do_m) = carry, xs
        k0,k1,k2,k3,k4,kU,kS,kZ = random.split(key_t, 8)

        # --- propose as you already do ---
        # prop0/1/2 = student-t / eigen-line / pCN ...
        props_all = jnp.stack([prop0, prop1, prop2], axis=0)        # (3,C,D)
        proposals = props_all[comp_idx_t, jnp.arange(C), :]
        proposals = _fold_params(proposals, fold_mask=fold_mask, period=period)

        # --- MH under current z ---
        prop_lp   = logpost(proposals, z)
        log_alpha = (prop_lp - lp) / temperatures
        ulog  = jnp.log(random.uniform(kU, (C,)))
        accept = ulog < log_alpha

        th = jnp.where(accept[:,None], proposals, th)
        lp = jnp.where(accept,        prop_lp,   lp)

        # --- Carlin–Chib z update on stride ---
        def _do(args):
            th_, z_, lp_ = args
            th2, z2, lp2, p1 = update_z_with_rejuv(kZ, th_, z_, loglik_M2_single, loglik_M3_single,
                                                   idx1, idx2, idx3, idx_rest, log_prior_z)
            return (th2, z2, lp2), p1
        def _skip(args):
            th_, z_, lp_ = args
            return (th_, z_, lp_), jnp.zeros((C,), th.dtype)

        (th, z, lp), p1_dbg = jax.lax.cond(do_m, _do, _skip, (th, z, lp))

        # --- PT swap (need to swap z as well) ---
        key_s, th_sw, lp_sw, raster, dbg = parallel_tempering_swap(kS, temperatures, th, lp, return_debug=True)
        i, j, acc_sel = dbg["pairs_i"], dbg["pairs_j"], dbg["accept_sel"]
        z_sw = _apply_swaps_vec(z, i, j, acc_sel)

        # collect what you need
        info_step = (proposals, prop_lp, accept, comp_idx_t, raster, th_sw, z_sw)  # etc.
        return (th_sw, lp_sw, z_sw), info_step

    keys = random.split(key, n_steps)
    comp_seq = ...  # your per-step (T,C) component choices, or keep per-epoch fixed like before
    xs = (keys, comp_seq, do_model)
    (th_f, lp_f, z_f), outs = lax.scan(body, (init_state.thetas, init_state.log_probs, init_state.z), xs)

    # unpack outs (props, prop_lps, accepts, comp_idxs, swaps, th_hist, z_hist) = outs
    # return final state + info; record z_hist in InfoAccumulator (see below)



# Add one optional field to your accumulator (non-breaking):

# In InfoAccumulator.__init__: self.model_indicator_states = []

# In InfoAccumulator.add(...): append z_history from the epoch scan.

# In pack(): out["model_indicator_state"] = np.concatenate(self.model_indicator_states, axis=0) # (ΣT, C)

# Then BF from the cold slot:

# z_state = np.asarray(run_info["model_indicator_state"])  # (ΣT, C)
# cold_z  = z_state[:, 0].astype(int)
# burn = int(0.3 * len(cold_z))
# m = cold_z[burn:]          # post burn-in
# n1 = m.sum(); n0 = m.size - n1
# post_odds = (n1 + 0.5) / (n0 + 0.5)   # Jeffreys
# BF_31 = post_odds                     # prior odds 1 for 50/50 prior
# print("log BF (M3:M2) =", np.log(BF_31))






# # ------------------------- Build proposals (external from your Proposals.py) -------------------------
# # We now rely on your existing function:
# # Proposals.build_general_mixture_components_per_chain(...)
# # The local helper previously here is no longer used.
# # Kept comment header for readability.

# def _unused_local_build_proposals(
#     covs: Sequence[np.ndarray],
#     eig_U: Sequence[np.ndarray],
#     eig_S: Sequence[np.ndarray],
#     means_for_indep: Sequence[np.ndarray],
#     temperatures: jnp.ndarray,
#     scales: np.ndarray,
#     *,
#     fold_idx=(),
#     period=1.0,
#     beta_base: float = 0.3,
#     beta_temp_scale: bool = True,
#     kappa_line: float = 3.0,
#     weights: np.ndarray | None = None,
# ):
#     """Wire up (sample_fn, logq_fn) tuples per chain using your helpers.

#     Each chain gets a 3-component mixture: full-cov RW, eigen-line RW, pCN.
#     Returns (proposal_fns_per_chain, logq_fns_per_chain, weights_arr).
#     """
#     C = len(covs)
#     if weights is None:
#         W = np.tile(np.array([0.6, 0.3, 0.1], dtype=np.float64), (C, 1))
#     else:
#         W = np.asarray(weights, dtype=np.float64)
#         assert W.shape == (C, 3)
#         # normalize
#         W /= np.clip(W.sum(axis=1, keepdims=True), 1e-32, None)

#     samp, logq = [], []

#     for c in range(C):
#         # (1) full-cov RW
#         f_s, f_q = Proposals.make_fullcov_proposal(jnp.asarray(covs[c]), fold_idx=fold_idx, period=period)
#         comp1_s = (lambda key, x, n, sf=1.0, f=f_s: f(key, x, n, sf))
#         comp1_q = (lambda newB, oldB, sf=1.0, q=f_q: q(newB, oldB, sf))

#         # (2) eigen-line RW
#         e_s, e_q = Proposals.make_eigenline_proposal(jnp.asarray(eig_U[c]), jnp.asarray(eig_S[c]),
#                                                      fold_idx=fold_idx, period=period,
#                                                      axis_probs=None)  # or pass probs
#         comp2_s = (lambda key, x, n, sf=1.0, f=e_s: f(key, x, n, sf))
#         comp2_q = (lambda newB, oldB, sf=1.0, q=e_q: q(newB, oldB, sf))

#         # (3) pCN / independence using running mean & cov
#         beta_c = float(beta_base / temperatures[c] if beta_temp_scale else beta_base)
#         g_s, g_q = Proposals.make_pcn_proposal(jnp.asarray(means_for_indep[c]),
#                                                jnp.asarray(covs[c]),
#                                                fold_idx=fold_idx, period=period, beta=beta_c)
#         comp3_s = (lambda key, x, n, sf=1.0, f=g_s: f(key, x, n, sf))
#         comp3_q = (lambda newB, oldB, sf=1.0, q=g_q: q(newB, oldB, sf))

#         samp.append((comp1_s, comp2_s, comp3_s))
#         logq.append((comp1_q, comp2_q, comp3_q))

#     return tuple(samp), tuple(logq), jnp.asarray(W)


# # ------------------------- One epoch (batched log-lik each step) -------------------------
# class EpochOut(NamedTuple):
#     states: PTState
#     infos: StepInfo  # stacked over steps via lax.scan


# def _mk_switch_branches_per_chain(proposal_fns_per_chain, logq_fns_per_chain):
#     """Return functions usable under JIT/scan that switch on chain & component.

#     propose_for_chain(chain_idx, comp_idx, key, theta) -> (dim,)
#     logq_for_chain(chain_idx, comp_idx, new, old) -> () scalar
#     """
#     C = len(proposal_fns_per_chain)
#     K = len(proposal_fns_per_chain[0])

#     # sample branches
#     def _make_sample_branch(cc: int):
#         branches = [
#             (lambda key, x, fn=proposal_fns_per_chain[cc][kk]: fn(key, x, 1)[0])
#             for kk in range(K)
#         ]
#         return lambda key, x, k: lax.switch(k, branches, key, x)

#     sample_chain_branches = [_make_sample_branch(c) for c in range(C)]

#     def propose_for_chain(chain_idx, comp_idx, key, theta):
#         return lax.switch(chain_idx, sample_chain_branches, key, theta, comp_idx)

#     # logq branches
#     def _make_logq_branch(cc: int):
#         branches = [
#             (lambda new, old, fn=logq_fns_per_chain[cc][kk]: jnp.squeeze(fn(new[None, :], old[None, :])))
#             for kk in range(K)
#         ]
#         return lambda new, old, k: lax.switch(k, branches, new, old)

#     logq_chain_branches = [_make_logq_branch(c) for c in range(C)]

#     def logq_for_chain(chain_idx, comp_idx, new, old):
#         return lax.switch(chain_idx, logq_chain_branches, new, old, comp_idx)

#     return propose_for_chain, logq_for_chain


# def run_epoch(
#     key,
#     init_state: PTState,
#     log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],  # maps (D,) -> ()
#     proposal_fns_per_chain: Tuple[Tuple, ...],
#     logq_fns_per_chain: Tuple[Tuple, ...],
#     weights: jnp.ndarray,  # (C, K)
#     n_steps: int,
# ) -> Tuple[PTState, StepInfo]:
#     """Run one epoch (n_steps). We compute all proposals' log-lik in a batched call
#     per step: (C, D) -> (C,). We also collect per-step info for later analysis.
#     """
#     C, D = init_state.thetas.shape
#     K = weights.shape[1]
    

#     propose_for_chain, logq_for_chain = _mk_switch_branches_per_chain(
#         proposal_fns_per_chain, logq_fns_per_chain
#     )

#     def step(state: PTState, key):
#         key, k_comp, k_prop, k_u, k_swap = random.split(key, 5)

#         # component choice per chain (vectorized categorical)
#         w = weights / jnp.clip(jnp.sum(weights, axis=1, keepdims=True), 1e-32, None)
#         logits = jnp.where(w > 0, jnp.log(w), -jnp.inf)
#         comp_idx = vmap(lambda lg, k: random.categorical(k, lg))(logits, random.split(k_comp, C))

#         # proposals for all chains (C, D)
#         idxs = jnp.arange(C)
#         prop_keys = random.split(k_prop, C)
#         proposals = vmap(propose_for_chain)(idxs, comp_idx, prop_keys, state.thetas)

#         # log q(new|old) and q(old|new) per chain (C,)
#         logq_fwd = vmap(logq_for_chain)(idxs, comp_idx, proposals, state.thetas)
#         logq_rev = vmap(logq_for_chain)(idxs, comp_idx, state.thetas, proposals)

#         # *** batched likelihood of ALL proposed points across chains ***
#         prop_lp = vmap(log_prob_fn)(proposals)  # (C,)

#         # MH accept per chain (tempered)
#         log_alpha = (prop_lp - state.log_probs) / state.temperatures + (logq_rev - logq_fwd)
#         accept = jnp.log(random.uniform(k_u, (C,))) < log_alpha

#         new_thetas = jnp.where(accept[:, None], proposals, state.thetas)
#         new_logps = jnp.where(accept, prop_lp, state.log_probs)
#         new_nacc = state.n_accepted + accept.astype(jnp.int32)

#         # PT adjacent swap
#         key2, t_sw, lp_sw, swap_dec = parallel_tempering_swap(k_swap, state.temperatures, new_thetas, new_logps)
#         n_swaps = state.n_swaps + swap_dec.astype(jnp.int32)
#         n_swap_attempts = state.n_swap_attempts + jnp.ones_like(swap_dec, dtype=jnp.int32)

#         state2 = PTState(
#             thetas=t_sw,
#             log_probs=lp_sw,
#             temperatures=state.temperatures,
#             n_accepted=new_nacc,
#             n_swaps=n_swaps,
#             n_swap_attempts=n_swap_attempts,
#         )

#         info = StepInfo(
#             thetas_prop=proposals,
#             logprob_prop=prop_lp,
#             accepted=accept,
#             comp_idx=comp_idx,
#             swap_decisions=swap_dec,
#         )
#         return state2, info

#     keys = random.split(key, n_steps)
#     final_state, infos = lax.scan(step, init_state, keys)
#     return final_state, infos





# # ------------------------- Top-level runner -------------------------

# # == Host-driven, performance-first variant ==
# # - Fixes proposal component per chain for the whole epoch (resampled each epoch)
# # - Drives the sampler from Python/NumPy; only the batched likelihood is JITed
# # - Uses chunked vmap for the likelihood to avoid kernel/cache kinks
# # - Avoids chain-index lax.switch branching entirely
# # - Records into the same InfoAccumulator interface



# def _make_batched_logprob_chunked(log_prob_fn_single, C: int, D: int, chunk: int):
#     """Return a function (X:(C,D)->(C,)) using pad+reshape+vmap with chunk size.
#     Compiled once per epoch (C may change across runs; here constant inside epoch).
#     """
#     chunk = max(1, min(int(chunk), C))
#     n_chunks = (C + chunk - 1) // chunk
#     C_pad = n_chunks * chunk
#     pad_rows = C_pad - C

#     v1 = jax.vmap(log_prob_fn_single)         # (chunk, D)->(chunk,)
#     v2 = jax.vmap(v1)                          # (n_chunks, chunk, D)->(n_chunks, chunk)

#     @jax.jit
#     def f(X):  # X: (C, D)
#         if pad_rows:
#             pad = jnp.zeros((pad_rows, D), dtype=X.dtype)
#             Xp = jnp.concatenate([X, pad], axis=0)
#         else:
#             Xp = X
#         blocks = Xp.reshape((n_chunks, chunk, D))
#         Yp = v2(blocks).reshape((C_pad,))
#         return Yp[:C]

#     return f


# def _pt_adjacent_swap_np(rng: np.random.Generator, temperatures: np.ndarray,
#                          thetas: np.ndarray, logp: np.ndarray):
#     """Single adjacent swap sweep on host (NumPy). Returns swap decisions (C-1,).
#     thetas: (C, D), logp: (C,) in-place updated copies are returned.
#     """
#     C = temperatures.shape[0]
#     betas = 1.0 / temperatures
#     do_swap = np.zeros((C - 1,), dtype=bool)

#     for i in range(C - 1):
#         dlog = (betas[i + 1] - betas[i]) * (logp[i + 1] - logp[i])
#         if np.log(rng.uniform()) < dlog:
#             do_swap[i] = True
#             # swap
#             thetas[[i, i + 1]] = thetas[[i + 1, i]]
#             logp[[i, i + 1]] = logp[[i + 1, i]]
#     return thetas, logp, do_swap


# def run_epoch_hostdriven(
#     key,
#     state: PTState,                       # uses jnp for arrays but fine to mix
#     log_prob_fn_single,                   # (D,)->()
#     proposal_fns_per_chain,               # tuple of C tuples (K=3) sample_fns
#     logq_fns_per_chain,                   # tuple of C tuples (K=3) logq_fns
#     weights: np.ndarray | jnp.ndarray,    # (C, K)
#     n_steps: int,
#     *,
#     lik_chunk: int = 32,
#     downsample_every: int = 2,
# ):
#     """Host-driven epoch: pick one component per chain for the *whole epoch*,
#     drive proposals/MH/PT on CPU, call a single jitted batched likelihood.
#     """
#     # Shapes & host copies
#     C, D = int(state.thetas.shape[0]), int(state.thetas.shape[1])
#     K = int(np.asarray(weights).shape[1])

#     # Prepare RNG: both JAX key (for proposal fns) and NumPy RNG (for swaps)
#     key = jax.random.PRNGKey(int(jax.random.randint(key, (), 0, 2**31 - 1))) if isinstance(key, np.random.Generator) else key
#     rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

#     # Fix component per chain for this epoch
#     W = np.asarray(weights)
#     W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-32, None)
#     comp_idx = np.array([rng.choice(K, p=W[c]) for c in range(C)], dtype=np.int32)  # (C,)

#     # Build jitted batched likelihood
#     batched_lp = _make_batched_logprob_chunked(log_prob_fn_single, C, D, lik_chunk)

#     # Host buffers for per-step info (to feed InfoAccumulator at epoch end)
#     thetas_prop_steps = np.empty((n_steps, C, D), dtype=np.float32)
#     logprob_prop_steps = np.empty((n_steps, C), dtype=np.float32)
#     accepted_steps = np.empty((n_steps, C), dtype=bool)
#     comp_idx_steps = np.tile(comp_idx[None, :], (n_steps, 1))  # (n_steps, C)
#     swap_steps = np.empty((n_steps, C - 1), dtype=bool)

#     # State (host-side copies)
#     th = np.array(state.thetas, dtype=np.float32)
#     lp = np.array(state.log_probs, dtype=np.float32)
#     temps = np.array(state.temperatures, dtype=np.float32)
#     n_acc = np.array(state.n_accepted, dtype=np.int32)
#     n_swaps = np.array(state.n_swaps, dtype=np.int32)
#     n_swap_att = np.array(state.n_swap_attempts, dtype=np.int32)

#     # Pre-split JAX keys for proposals (n_steps * C)
#     step_keys = jax.random.split(key, n_steps)
#     prop_keys = jax.random.split(key, n_steps * C).reshape(n_steps, C, 2)  # we'll split again

#     # Downsample buffers for covariance adaptation
#     ds_buffers = [np.empty((0, D), dtype=np.float32) for _ in range(C)]

#     for t in range(n_steps):
#         # Propose for each chain using its fixed component comp_idx[c]
#         prop = np.empty((C, D), dtype=np.float32)
#         logq_fwd = np.empty((C,), dtype=np.float32)
#         logq_rev = np.empty((C,), dtype=np.float32)

#         for c in range(C):
#             k = int(comp_idx[c])
#             key_t = step_keys[t]
#             # derive per-chain key
#             k1, k2 = jax.random.split(jax.random.fold_in(key_t, c))

#             # sample
#             sample_fn = proposal_fns_per_chain[c][k]
#             new_c = sample_fn(k1, jnp.asarray(th[c]), 1)[0]     # (D,)
#             prop[c, :] = np.asarray(new_c, dtype=np.float32)

#             # log q terms
#             lq_fn = logq_fns_per_chain[c][k]
#             logq_fwd[c] = float(lq_fn(jnp.asarray(new_c[None, :]), jnp.asarray(th[c][None, :])).squeeze())
#             logq_rev[c] = float(lq_fn(jnp.asarray(th[c][None, :]), jnp.asarray(new_c[None, :])).squeeze())

#         # Batched log-likelihood on device (chunked)
#         prop_lp = np.asarray(batched_lp(jnp.asarray(prop)))  # (C,)

#         # Tempered MH
#         log_alpha = (prop_lp - lp) / temps + (logq_rev - logq_fwd)
#         u = np.log(rng.random(C))
#         accept = u < log_alpha

#         # Apply accepts
#         th = np.where(accept[:, None], prop, th)
#         lp = np.where(accept, prop_lp, lp)
#         n_acc += accept.astype(np.int32)

#         # PT swaps (adjacent)
#         th, lp, swap_dec = _pt_adjacent_swap_np(rng, temps, th, lp)
#         n_swaps += swap_dec.astype(np.int32)
#         n_swap_att += np.ones_like(swap_dec, dtype=np.int32)

#         # Record step info
#         thetas_prop_steps[t] = prop
#         logprob_prop_steps[t] = prop_lp
#         accepted_steps[t] = accept
#         swap_steps[t] = swap_dec

#         # Downsample states into buffers for covariance adaptation
#         if (t % downsample_every) == 0:
#             for c in range(C):
#                 ds_buffers[c] = np.vstack([ds_buffers[c], th[c][None, :]])

#     # Build new PTState (convert back to jnp for consistency)
#     new_state = PTState(
#         thetas=jnp.asarray(th),
#         log_probs=jnp.asarray(lp),
#         temperatures=jnp.asarray(temps),
#         n_accepted=jnp.asarray(n_acc),
#         n_swaps=jnp.asarray(n_swaps),
#         n_swap_attempts=jnp.asarray(n_swap_att),
#     )

#     # Wrap StepInfo-like numpy into jnp for downstream uniformity
#     info = StepInfo(
#         thetas_prop=jnp.asarray(thetas_prop_steps),
#         logprob_prop=jnp.asarray(logprob_prop_steps),
#         accepted=jnp.asarray(accepted_steps),
#         comp_idx=jnp.asarray(comp_idx_steps[:, :, 0] if comp_idx_steps.ndim == 3 else comp_idx_steps),
#         swap_decisions=jnp.asarray(swap_steps),
#     )

#     return new_state, info, ds_buffers, comp_idx


# def run_adaptive_pt_hostdriven(
#     key,
#     initial_thetas: jnp.ndarray,      # (C, D)
#     temperatures: jnp.ndarray,        # (C,)
#     log_prob_fn_single,               # (D,) -> ()
#     base_cov: np.ndarray,             # (D, D)
#     *,
#     fold_idx=(),
#     period=1.0,
#     weights: np.ndarray | None = None,  # (C, 3)
#     cfg: AdaptConfig = AdaptConfig(),
#     lik_chunk: int = 32,
# ):
#     """Performance-first PT with host-driven control and epoch-fixed components."""
#     C, D = initial_thetas.shape
#     acc_buffers = [np.empty((0, D), dtype=np.float64) for _ in range(C)]  # <- NEW

#     # init per-chain covariance, eigen, means
#     covs = [np.array(base_cov, copy=True) for _ in range(C)]
#     means = [np.array(initial_thetas[c]) for c in range(C)]
#     eig_U = [np.eye(D) for _ in range(C)]
#     eig_S = [np.ones(D) for _ in range(C)]

#     scales = np.full((C,), float(cfg.scale_init), dtype=np.float64)

#     # initial state (evaluate logp once per chain)
#     batched_init = _make_batched_logprob_chunked(log_prob_fn_single, C, D, max(1, min(C, lik_chunk)))
#     logp0 = np.asarray(batched_init(initial_thetas))
#     state = PTState(
#         thetas=initial_thetas,
#         log_probs=jnp.asarray(logp0),
#         temperatures=temperatures,
#         n_accepted=jnp.zeros((C,), dtype=jnp.int32),
#         n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
#         n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
#     )

#     info_accum = InfoAccumulator()

#     # host RNG
#     rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

#     for epoch in trange(cfg.m_epochs, desc='Adaptive PT (host)', unit='epoch'):
#         # Build proposals for this epoch (uses updated covs/eigs/means/scales)
#         proposal_fns, logq_fns, W = Proposals.build_general_mixture_components_per_chain(
#             covs, eig_U, eig_S, means, temperatures, scales,
#             fold_idx=fold_idx, period=period,
#             beta_base=cfg.beta_base, beta_temp_scale=cfg.beta_temp_scale,
#             kappa_line=cfg.kappa_line, weights=weights,
#         )

#         # Run epoch on host
#         state, info, ds_buffers, comp_idx = run_epoch_hostdriven(
#             key, state, log_prob_fn_single, proposal_fns, logq_fns, W,
#             n_steps=cfg.N_steps, lik_chunk=lik_chunk, downsample_every=cfg.downsample_every,
#         )

#         # Acceptance per chain this epoch
#         acc_rate = np.asarray(info.accepted).mean(axis=0)

#         # Robbins–Monro scale adaptation
#         for c in range(C):
#             scales[c] = float(
#                 np.clip(
#                     np.exp(np.log(scales[c]) + cfg.eta * (acc_rate[c] - cfg.target_accept)),
#                     cfg.scale_min, cfg.scale_max,
#                 )
#             )

#         # Covariance + eigen update per chain from downsampled buffers
#         new_covs, new_U, new_S = [], [], []
#         for c in range(C):
#             buf = ds_buffers[c]
#             if buf.shape[0] >= 2:
#                 cov_hat = _empirical_cov(buf, ddof=1)
#             else:
#                 cov_hat = base_cov
#             cov_c = _shrink_spd(cov_hat, shrink=cfg.shrink, jitter=cfg.jitter)
#             S, U = np.linalg.eigh(cov_c)
#             S = np.clip(S, 1e-12, None)
#             new_covs.append(cov_c)
#             new_U.append(U)
#             new_S.append(S)
#             # update running mean for independence/pCN
#             means[c] = 0.9 * means[c] + 0.1 * np.asarray(state.thetas[c])

#         covs, eig_U, eig_S = new_covs, new_U, new_S

#         # record epoch results
#         info_accum.add(state, info, np.asarray(temperatures), scales, covs)

#         # tqdm diagnostics
#         swap_rate = float(np.mean(np.asarray(info.swap_decisions)))
#         mean_acc = float(np.mean(np.asarray(info.accepted)))
#         tqdm.write(f"Epoch {len(info_accum.scales_per_epoch)}/{cfg.m_epochs} | swap_rate={swap_rate:.3f} | mean_acc={mean_acc:.3f}")

#     return state, info_accum.finalize()


# def run_adaptive_pt(
#     key,
#     initial_thetas: jnp.ndarray,      # (C, D)
#     temperatures: jnp.ndarray,        # (C,)
#     log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
#     base_cov: np.ndarray,             # (D, D)
#     *,
#     fold_idx=(),
#     period=1.0,
#     init_weights: np.ndarray | None = None,  # (C, 3)
#     cfg: AdaptConfig = AdaptConfig(),
# ):
#     C, D = initial_thetas.shape

#     # init per-chain covariance, eigen-decomp, means
#     covs = [np.array(base_cov, copy=True) for _ in range(C)]
#     means = [np.array(initial_thetas[c]) for c in range(C)]
#     eig_U = [np.eye(D) for _ in range(C)]
#     eig_S = [np.ones(D) for _ in range(C)]

#     scales = np.full((C,), float(cfg.scale_init), dtype=np.float64)
#     weights = init_weights

#     # initialize current state
#     logp0 = vmap(log_prob_fn)(initial_thetas)
#     state = PTState(
#         thetas=initial_thetas,
#         log_probs=logp0,
#         temperatures=temperatures,
#         n_accepted=jnp.zeros((C,), dtype=jnp.int32),
#         n_swaps=jnp.zeros((C - 1,), dtype=jnp.int32),
#         n_swap_attempts=jnp.zeros((C - 1,), dtype=jnp.int32),
#     )

#     # buffers for covariance adaptation
#     buffers: list[list[np.ndarray]] | list[np.ndarray] = (
#         [[] for _ in range(C)] if cfg.cov_mode == "all_states" else [np.empty((0, D)) for _ in range(C)]
#     )

#     info_accum = InfoAccumulator()


#     for epoch in trange(cfg.m_epochs, desc='Adaptive PT', unit='epoch'):
#         # (re)build proposals for this epoch
#         proposal_fns, logq_fns, W = Proposals.build_general_mixture_components_per_chain(
#             covs, eig_U, eig_S, means, temperatures, scales,
#             fold_idx=fold_idx, period=period,
#             beta_base=cfg.beta_base, beta_temp_scale=cfg.beta_temp_scale,
#             kappa_line=cfg.kappa_line, weights=weights,
#         )

#         key, subkey = random.split(key)
#         state, info = run_epoch(
#             subkey, state, log_prob_fn, proposal_fns, logq_fns, W, cfg.N_steps
#         )

#         # host-side arrays for adaptation
#         thetas_end = np.asarray(state.thetas)
#         logp_end = np.asarray(state.log_probs)
#         temps = np.asarray(temperatures)
#         acc_mask = np.asarray(info.accepted)  # (T, C)

#         # update per-chain running means (for pCN/indep)
#         means = [0.9 * means[c] + 0.1 * thetas_end[c] for c in range(C)]

#         # downsample post-swap states into buffers
#         step_states = np.asarray(state.thetas)  # already last state; also keep intermediate if desired
#         # If you want intermediate post-swap states, track them inside scan & collect.
#         if cfg.cov_mode == "rolling":
#             # append and truncate per chain
#             for c in range(C):
#                 buf = buffers[c]
#                 buf = np.vstack([buf, thetas_end[c][None, :]]) if buf.size else thetas_end[c][None, :]
#                 if buf.shape[0] > cfg.window_size:
#                     buf = buf[-cfg.window_size :, :]
#                 buffers[c] = buf
#         else:
#             # accumulate blocks
#             blocks = np.asarray(state.thetas)  # last only; replace with a log of states across steps if needed
#             for c in range(C):
#                 buffers[c].append(thetas_end[c][None, :])

#         # acceptance rate per chain this epoch (over steps)
#         acc_rate = acc_mask.mean(axis=0)

#         # Robbins–Monro scale adaptation per chain
#         for c in range(C):
#             scales[c] = float(
#                 np.clip(
#                     np.exp(np.log(scales[c]) + cfg.eta * (acc_rate[c] - cfg.target_accept)),
#                     cfg.scale_min, cfg.scale_max,
#                 )
#             )

#         # covariance + eigen update per chain
#         new_covs = []
#         new_U = []
#         new_S = []
#         for c in range(C):
#             if cfg.cov_mode == "rolling":
#                 buf = np.asarray(buffers[c])
#             else:
#                 blk = buffers[c]
#                 buf = np.vstack(blk) if len(blk) else np.empty((0, D))
#             cov_hat = _empirical_cov(buf, ddof=1) if buf.shape[0] >= 2 else base_cov
#             cov_c = _shrink_spd(cov_hat, shrink=cfg.shrink, jitter=cfg.jitter)
#             # eigh for symmetric covariance (stable in float64)
#             S, U = np.linalg.eigh(cov_c)
#             S = np.clip(S, 1e-12, None)
#             new_covs.append(cov_c)
#             new_U.append(U)
#             new_S.append(S)

#         covs, eig_U, eig_S = new_covs, new_U, new_S

#         # record epoch results
#         info_accum.add(state, info, temps, scales, covs)

#         # tqdm diagnostics
#         swap_rate = float(np.mean(np.asarray(info.swap_decisions)))
#         acc_rate = float(np.mean(np.asarray(info.accepted)))
#         try:
#             from tqdm import tqdm
#             tqdm.write(f"Epoch {epoch+1}/{cfg.m_epochs} | swap_rate={swap_rate:.3f} | mean_acc={acc_rate:.3f}")
#         except Exception:
#             pass

#     return state, info_accum.finalize()


# class InfoAccumulator:
#     """
#     Collects per-epoch outputs from PTwarmup_collect_mixture and packs them
#     into a single dict at the end. Uses NumPy lists to keep JAX happy.
#     """
#     def __init__(self):
#         # per-step/chain tensors
#         self.thetas_prop   = []   # list of (T, C, D)
#         self.logprob_prop  = []   # list of (T, C)
#         self.accepted      = []   # list of (T, C) bool
#         self.comp_idx      = []   # list of (T, C) int
#         self.swap_decisions= []   # list of (T, C-1) bool

#         # post-swap state traces (optional but useful)
#         self.thetas_state  = []   # list of (T, C, D)
#         self.logprob_state = []   # list of (T, C)
#         self.replica_id    = []   # list of (T, C) int  (requires you added replica_id to state)

#         # temperatures per epoch (so you know which T per chain)
#         self.temperatures  = []   # list of (C,)

#         # history (per epoch)
#         self.scales_per_epoch      = []  # (C,)
#         self.weights_per_epoch     = []  # (C, K)
#         self.swap_rate_per_epoch   = []  # float
#         self.accept_rate_per_epoch = []  # (C,)
#         self.covs_per_epoch = [] # list of (C, D, D)

#     def add_epoch_core(self, states, infos, temperatures):
#         self.thetas_prop.append(   np.asarray(infos["thetas_prop"]))
#         self.logprob_prop.append(  np.asarray(infos["logprob_prop"]))
#         self.accepted.append(      np.asarray(infos["accepted"]))
#         self.comp_idx.append(      np.asarray(infos["comp_idx"]))
#         self.swap_decisions.append(np.asarray(infos["swap_decisions"]))

#         self.thetas_state.append(  np.asarray(states.thetas))
#         self.logprob_state.append( np.asarray(states.log_probs))
#         if hasattr(states, "replica_id"):
#             self.replica_id.append(np.asarray(states.replica_id))

#         self.temperatures.append(np.asarray(temperatures))

#     def add_epoch_meta(self, scales, weights, swap_rate, acc_rates, covs):
#         self.scales_per_epoch.append(np.asarray(scales))
#         self.weights_per_epoch.append(np.asarray(weights))
#         self.swap_rate_per_epoch.append(float(swap_rate))
#         self.accept_rate_per_epoch.append(np.asarray(acc_rates))
#         self.covs_per_epoch.append(np.stack(covs, axis=0))  # (C, D, D)

#     def _cat(self, lst): return np.concatenate(lst, axis=0) if lst else None

#     def finalize(self):
#         out = {
#             # proposals (pre-swap)
#             "thetas_prop":    self._cat(self.thetas_prop),     # (ΣT, C, D)
#             "logprob_prop":   self._cat(self.logprob_prop),    # (ΣT, C)
#             "accepted":       self._cat(self.accepted),        # (ΣT, C)
#             "comp_idx":       self._cat(self.comp_idx),        # (ΣT, C)
#             "swap_decisions": self._cat(self.swap_decisions),  # (ΣT, C-1)

#             # post-swap states
#             "thetas_state":   self._cat(self.thetas_state),    # (ΣT, C, D)
#             "logprob_state":  self._cat(self.logprob_state),   # (ΣT, C)
#             "replica_id":     self._cat(self.replica_id),      # (ΣT, C) or None

#             # temperatures per epoch (list of (C,))
#             "temperatures_per_epoch": self.temperatures,

#             # history (lists per epoch)
#             "history": {
#                 "scale":      self.scales_per_epoch,      # list[(C,)]
#                 "weights":    self.weights_per_epoch,     # list[(C,K)]
#                 "swap_rate":  self.swap_rate_per_epoch,   # list[float]
#                 "accept_rate":self.accept_rate_per_epoch, # list[(C,)]
#                 "covs_per_epoch": self.covs_per_epoch  # list[(C, D, D)]
#             }
#         }

#         # Flattened dataset for NN training (all proposals)
#         if out["thetas_prop"] is not None:
#             Ttot, C, D = out["thetas_prop"].shape
#             X     = out["thetas_prop"].reshape(Ttot*C, D)
#             y     = out["logprob_prop"].reshape(Ttot*C)
#             acc   = out["accepted"].reshape(Ttot*C).astype(np.int8)
#             comp  = out["comp_idx"].reshape(Ttot*C).astype(np.int16)
#             chain = np.tile(np.arange(C, dtype=np.int32), Ttot)
#             temps0 = self.temperatures[0]  # assumes fixed ladder
#             temp_flat = np.repeat(temps0, Ttot).reshape(Ttot, C).ravel()
#             out["dataset"] = dict(
#                 X=X, y_logprob=y, accepted=acc,
#                 chain=chain, temperature=temp_flat, comp_idx=comp
#             )
#         return out




