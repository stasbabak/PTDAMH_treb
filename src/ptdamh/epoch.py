# src/ptdamh/epoch.py
from __future__ import annotations
from dataclasses import dataclass
from logging import log
from typing import Callable, NamedTuple, Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random, lax

# ---- import proposals ----
from .proposals import (
    redblue_mask,
    _propose_stretch_redblue,
    _propose_fullcov,
    _propose_eigenline,
    _propose_student_t,
    _propose_de_two_point,
)

from .utilities import _fold_params


class PTState(NamedTuple):
     thetas: jnp.ndarray    # (C, W, D)
     log_probs: jnp.ndarray # (C, W)


class PTTrace(NamedTuple): ## records the chain (for each temperature, each walker across epoch)
    thetas: jnp.ndarray       # (T+1, C, W, D)
    log_probs: jnp.ndarray    # (T+1, C, W)


# Proposal IDs for logging (int8-friendly)
PROPOSAL_IDS: Dict[str, int] = {
    "stretch": 0,          # (combined red+blue)
    "rw_fullcov": 1,
    "rw_eigenline": 2,
    "rw_student_t": 3,
    "de_two_point": 4,
    "pt_swap": 5,          # (combined even+odd)
    "skipped": 255,
}


class EventLog(NamedTuple): 
    ids: jnp.ndarray        # (T, S) int8           <-- only once per slot/step, S=1,2,3,4 porposal slot
    attempted: jnp.ndarray  # (T, S, C, W) bool
    accepted: jnp.ndarray   # (T, S, C, W) bool

def make_empty_event_log(n_steps, C, W, dtype=jnp.float32):
    S = 4
    ids       = jnp.full((n_steps, S), PROPOSAL_IDS["skipped"], dtype=jnp.int8)  # (T,S)
    attempted = jnp.zeros((n_steps, S, C, W), dtype=jnp.bool_)
    accepted  = jnp.zeros((n_steps, S, C, W), dtype=jnp.bool_)
    return EventLog(ids=ids, attempted=attempted, accepted=accepted)

### aux
def cholesky_from_covs_safe(
    covs: jnp.ndarray,                 # (C, D, D)
    *,
    evals: jnp.ndarray | None = None,  # optional (C, D)
    evecs: jnp.ndarray | None = None,  # optional (C, D, D)
    min_eig_rel: float = 1e-7,
    min_eig_abs: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (L, evals_pd, evecs), where L is Cholesky of a PD-projected version of covs.
    If evals/evecs are provided, they are used; otherwise we compute eigh.
    Eigenvalues are floored to max(min_eig_abs, min_eig_rel * scale) per matrix.
    """
    # Symmetrize to kill tiny asymmetry
    sym = 0.5 * (covs + jnp.swapaxes(covs, -1, -2))

    if (evals is None) or (evecs is None):
        evals, evecs = jnp.linalg.eigh(sym)  # (C,D), (C,D,D)

    # Per-matrix scale (use max diag or max eig — both are fine)
    # Using max eig keeps the floor in the same basis we’re clamping.
    scale = jnp.maximum(jnp.max(evals, axis=-1, keepdims=True), 1.0)

    floor = jnp.maximum(min_eig_abs, min_eig_rel * scale)  # (C,1)
    evals_pd = jnp.maximum(evals, floor)                   # (C,D)

    # Rebuild SPD covariance and factor
    covs_pd = evecs @ (evals_pd[..., None] * jnp.swapaxes(evecs, -1, -2))
    L = jnp.linalg.cholesky(covs_pd)
    return L, evals_pd, evecs


# ---- Likelihood batching ----
def batched_log_prob(
    log_prob_fn_single: Callable[[jnp.ndarray], jnp.ndarray],
    xs_flat: jnp.ndarray,       # (B, D)
    chunk: int = 8192
) -> jnp.ndarray:
    """
    JIT-safe batching: pad to a multiple of `chunk` so dynamic_slice sizes are static.
    """
    B, D = xs_flat.shape
    f = jax.vmap(log_prob_fn_single)  # (N,D) -> (N,)

    if (chunk is None) or (B <= chunk):
        return f(xs_flat)

    # number of chunks & padding to make B a multiple of chunk
    n_chunks = (B + chunk - 1) // chunk
    pad = n_chunks * chunk - B

    # pad by repeating the last row (keeps inputs valid for your logprob)
    xs_pad = jnp.pad(xs_flat, ((0, pad), (0, 0)), mode="edge")  # (n_chunks*chunk, D)
    xs_blocks = xs_pad.reshape((n_chunks, chunk, D))            # (n_chunks, chunk, D)

    # evaluate each fixed-size block
    def eval_block(block):  # block: (chunk, D)
        return f(block)     # -> (chunk,)

    ys_blocks = jax.vmap(eval_block, in_axes=0)(xs_blocks)      # (n_chunks, chunk)
    y_flat = ys_blocks.reshape((n_chunks * chunk,))[:B]         # trim padding
    return y_flat

# ---- MH accept (tempered) ----
def mh_accept_masked(
    key: jax.random.PRNGKey,
    current_lp: jnp.ndarray,      # (C, W)
    proposed_lp: jnp.ndarray,     # (C, W)
    betas: jnp.ndarray,           # (C,)
    log_qcorr: jnp.ndarray,       # (C, W)
    move_mask: jnp.ndarray,       # (C, W) boolean: who actually proposed a move
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    delta = (proposed_lp - current_lp) * betas[:, None] + log_qcorr
    log_u = jnp.log(random.uniform(key, shape=current_lp.shape))
    accept = (log_u < jnp.minimum(0.0, delta)) & move_mask
    return accept, delta

# ---- PT swap pass ----
def pt_swap_pass(
    key: jax.random.PRNGKey,
    state: PTState,
    betas: jnp.ndarray,
    even_pass: bool,
) -> Tuple[PTState, jnp.ndarray, jnp.ndarray]:
    thetas, lps = state.thetas, state.log_probs
    C, W, D = thetas.shape
    start = 0 if even_pass else 1
    idx_low  = jnp.arange(start, C-1, 2)
    idx_high = idx_low + 1

    bi, bj = betas[idx_low], betas[idx_high]     # (P,), (P,)
    li = lps[idx_low, :]                         # (P,W)
    lj = lps[idx_high, :]                        # (P,W)
    delta = (bi - bj)[:, None] * (lj - li)       # (P,W)
    log_u = jnp.log(random.uniform(key, shape=delta.shape))
    accept_pairs = log_u < jnp.minimum(0.0, delta)

    def swap_pair(carry, p):
        th, lp = carry
        i = idx_low[p]; j = idx_high[p]
        mask = accept_pairs[p]            # (W,)
        mask_wd = mask[:, None]
        thi, thj = th[i], th[j]
        lpi, lpj = lp[i], lp[j]
        new_i = jnp.where(mask_wd, thj, thi)
        new_j = jnp.where(mask_wd, thi, thj)
        new_lpi = jnp.where(mask, lpj, lpi)
        new_lpj = jnp.where(mask, lpi, lpj)
        th = th.at[i].set(new_i)
        th = th.at[j].set(new_j)
        lp = lp.at[i].set(new_lpi)
        lp = lp.at[j].set(new_lpj)
        return (th, lp), None

    (thetas_new, lps_new), _ = lax.scan(swap_pair, (thetas, lps), jnp.arange(idx_low.shape[0]))

    acc_mask = jnp.zeros((C, W), dtype=jnp.bool_)
    def mark_pair(acc, p):
        i = idx_low[p]; j = idx_high[p]
        m = accept_pairs[p]
        acc = acc.at[i].set(m | acc[i])
        acc = acc.at[j].set(m | acc[j])
        return acc, None
    acc_mask, _ = lax.scan(mark_pair, acc_mask, jnp.arange(idx_low.shape[0]))
    
    # attempted mask: mark both temps of every considered pair True (independent of acceptance)
    att_mask = jnp.zeros((C, W), dtype=jnp.bool_)                                    # <<< ADD
    def mark_attempt(acc, p):                                                        # <<< ADD
        i = idx_low[p]; j = idx_high[p]
        m = jnp.ones((W,), dtype=jnp.bool_)
        acc = acc.at[i].set(m | acc[i])
        acc = acc.at[j].set(m | acc[j])
        return acc, None
    att_mask, _ = lax.scan(mark_attempt, att_mask, jnp.arange(idx_low.shape[0]))     # <<< ADD
    return PTState(thetas=thetas_new, log_probs=lps_new), acc_mask, att_mask 

# # ---- RW and DE adapters (unchanged, abbreviated signatures) ----
# def propose_rw_all(
#     key: jax.random.PRNGKey,
#     X: jnp.ndarray,                 # (C,W,D)
#     Ls: jnp.ndarray,                # (C,D,D)  <<< Cholesky factors
#     U: Optional[jnp.ndarray],       # (C,D,D)
#     S: Optional[jnp.ndarray],       # (C,D)
#     scales: Dict[str, jnp.ndarray], # {"small": (C,), "line": (C,), "big": (C,)}
#     kind: str,
#     df: float = 7.0
# ):
#     from .proposals import _propose_fullcov, _propose_eigenline, _propose_student_t
#     C = X.shape[0]
#     keys = random.split(key, C)

#     if kind == "fullcov":
#         # _propose_fullcov(key, x, L, scale=None)
#         def one(k, Xc, Lc, scl): 
#             return _propose_fullcov(k, Xc, Lc, scale=scl)
#         prop = jax.vmap(one, in_axes=(0,0,0,0))(keys, X, Ls, scales["small"])
#         log_qcorr = jnp.zeros(X.shape[:2]); meta = {}

#     elif kind == "eigenline":
#         # _propose_eigenline(key, x, U, S, scale=None)
#         def one(k, Xc, Uc, Sc, scl): 
#             return _propose_eigenline(k, Xc, Uc, Sc, scale=scl)
#         prop = jax.vmap(one, in_axes=(0,0,0,0,0))(keys, X, U, S, scales["line"])
#         log_qcorr = jnp.zeros(X.shape[:2]); meta = {}

#     elif kind == "student_t":
#         # assuming your student-t also takes L (common pattern)
#         # _propose_student_t(key, x, L, df=..., scale=None)
#         def one(k, Xc, Lc, scl): 
#             return _propose_student_t(k, Xc, Lc, df=df, scale=scl)
#         prop = jax.vmap(one, in_axes=(0,0,0,0))(keys, X, Ls, scales["big"])
#         log_qcorr = jnp.zeros(X.shape[:2]); meta = {}

#     else:
#         raise ValueError(f"Unknown RW kind: {kind}")

#     return prop, log_qcorr, meta

# def propose_de_all(
#     key_partner: jax.random.PRNGKey,
#     key_gamma: jax.random.PRNGKey,
#     X: jnp.ndarray,
#     gamma: Optional[float] = None,
#     gamma_scale: float = 2.38,
#     crossover_rate: float = 0.7,
#     jitter_scale: float = 1e-6,
# ):
#     from .proposals import _propose_de_two_point
#     C = X.shape[0]
#     keys_p = random.split(key_partner, C)
#     keys_g = random.split(key_gamma, C)
#     def one(kp, kg, Xc):
#         return _propose_de_two_point(kp, kg, Xc, gamma=gamma, gamma_scale=gamma_scale,
#                                      crossover_rate=crossover_rate, jitter_scale=jitter_scale)
#     prop, idx_y, idx_z, has_pair, mask = jax.vmap(one, in_axes=(0,0,0))(keys_p, keys_g, X)
#     meta = {"idx_y": idx_y, "idx_z": idx_z, "has_pair": has_pair, "mask": mask}
#     log_qcorr = jnp.zeros(X.shape[:2])
#     return prop, log_qcorr, meta

# ---- Runner with red-blue stretch ----
def run_epoch(
    key: jax.random.PRNGKey,
    init_state: PTState,                # thetas: (C,W,D), log_probs: (C,W)
    log_prob_fn_single: Callable[[jnp.ndarray], jnp.ndarray],
    temperatures: jnp.ndarray,          # (C,)
    covs: jnp.ndarray,                  # (C,D,D)
    *,
    n_steps: int,
    W_sm: float,                        # prob to trigger stretch (per iteration)
    rw_weights: Tuple[float, float, float],  # (w1,w2,w3) for (fullcov,eigenline,student_t)
    W_de: float,                        # prob to trigger DE (if W>=2)
    fold_mask: jnp.ndarray | None = None,# (D,)
    period: float = 1.0,
    scale_small: jnp.ndarray | None = None,           # (C,)
    scale_line: jnp.ndarray | None = None,            # (C,)
    scale_big: jnp.ndarray | None = None,             # (C,)
    stretch_a: float = 1.3,
    cross_rate: float = 0.6,
    gamma_de: float = 2.38,
    student_df: float = 5.0,
    lik_chunk: int = 64,
) -> Tuple[PTState, PTTrace, EventLog]:

    
    thetas0, lps0 = init_state.thetas, init_state.log_probs

    C, W, D = thetas0.shape
    if fold_mask is not None:
        fm = jnp.asarray(fold_mask)
        if fm.ndim == 1 and fm.size == D and fm.dtype == jnp.bool_:
            periodic_mask = fm
        elif fm.ndim == 1 and jnp.issubdtype(fm.dtype, jnp.integer):
            # treat as list/array/tuple of periodic indices
            idx = fm.astype(jnp.int32)
            periodic_mask = jnp.zeros((D,), dtype=jnp.bool_).at[idx].set(True)
        elif fm.ndim == 1 and fm.size == D:
            periodic_mask = fm.astype(jnp.bool_)
        else:
            raise ValueError(f"fold_mask must be length-{D} bool mask or index list; got shape {fm.shape}, dtype {fm.dtype}")
        fold_mask = periodic_mask

    def _fold(x):
        return _fold_params(x, fold_mask, period) if fold_mask is not None else x

    thetas0 = _fold(thetas0)
    flat0 = thetas0.reshape((C*W, D))
    lps0  = batched_log_prob(log_prob_fn_single, flat0, chunk=lik_chunk).reshape((C, W))

    betas = 1.0 / temperatures  # <- your log_prob is untempered

    evals, evecs = jnp.linalg.eigh(covs)  # (C,D), (C,D,D)
    evals = jnp.clip(evals, a_min=1e-12)
    scales = {"small": scale_small, "line": scale_line, "big": scale_big}
    
    # Make SPD via eigenvalue floor (more robust than a fixed diagonal jitter)
    min_eig_abs = 1e-12
    min_eig_rel = 1e-7
    scale = jnp.maximum(jnp.max(evals, axis=-1, keepdims=True), 1.0)
    floor = jnp.maximum(min_eig_abs, min_eig_rel * scale)
    evals_pd = jnp.maximum(evals, floor)

    # Rebuild SPD covariance (optional; we only need L below)
    covs_pd = evecs @ (evals_pd[..., None] * jnp.swapaxes(evecs, -1, -2))

    # Per-temperature Cholesky for RW(fullcov) / RW(student_t)
    Ls = jnp.linalg.cholesky(covs_pd)     # (C,D,D)

    
    w1, w2, w3 = rw_weights
    rw_probs = jnp.array([w1, w2, w3], dtype=jnp.float32)
    rw_probs = rw_probs / jnp.sum(rw_probs)

    log = make_empty_event_log(n_steps, C, W, dtype=lps0.dtype)

    def one_step(carry, t):

        # --- AFTER: same, we’ll also RETURN per-step chain in the scan’s ys ---
        key, state, log = carry
        th, lp = state.thetas, state.log_probs
        k = key
        k, k_rbmask, k_st_p1, k_st_z1, k_st_p2, k_st_z2, k_gate = random.split(k, 7)
        k, k_rw, k_el, k_de_p, k_de_g, k_swap_e, k_swap_o = random.split(k, 7)

        # utility: apply MH with masking (for subset moves)
        def apply_mh_masked(prop_thetas, log_qcorr, slot_id, prop_id, rng_key, move_mask):
            prop_thetas_f = _fold(prop_thetas)
            flat = prop_thetas_f.reshape((C*W, D))
            prop_lp_flat = batched_log_prob(log_prob_fn_single, flat, chunk=lik_chunk)
            prop_lp = prop_lp_flat.reshape((C, W))
            accept, _ = mh_accept_masked(rng_key, lp, prop_lp, betas, log_qcorr, move_mask)
            th_new = jnp.where(accept[:, :, None], prop_thetas_f, th)
            lp_new = jnp.where(accept, prop_lp, lp)
            # ---------- LOGGING ----------
            ids_arr = log.ids.at[t, slot_id].set(jnp.int8(prop_id))                     # (T,S)
            att_arr = log.attempted.at[t, slot_id].set(move_mask)                       # (T,S,C,W)
            acc_arr = log.accepted.at[t, slot_id].set(accept)                           # (T,S,C,W)
            log2 = EventLog(ids=ids_arr, attempted=att_arr, accepted=acc_arr)           # <<< FIX
            # ----------------------------------
            return PTState(th_new, lp_new), log2
        

        # ---- 0) RED-BLUE stretch (combined slot) ----
        slot = 0
        prop_id_stretch = PROPOSAL_IDS["stretch"]

        do_stretch = random.bernoulli(k_gate, W_sm)
        def _stretch_yes(args):
            _k, st_local, _log = args
            th_local, lp_local = st_local.thetas, st_local.log_probs

            # keys_rb = random.split(k_rbmask, C)          # (C, 2) 
            red, blue = redblue_mask(k_rbmask, C, W)      # (C, W), (C, W)

            # accumulators for the combined stretch slot
            acc_slot = jnp.zeros((C, W), dtype=jnp.bool_)                                              # <<< NEW
            att_slot = jnp.zeros((C, W), dtype=jnp.bool_)                                              # <<< NEW

            # Half A: RED moves (using BLUE pool)
            prop1, logJ1, partner1, has_partner1, zfac1 = _propose_stretch_redblue(
                k_st_p1, k_st_z1, th_local, subset_mask=red, a=stretch_a, z=None
            )
            move1 = red & has_partner1
            prop1 = _fold(prop1)                                                                   # <<< FIX
            prop_lp1 = batched_log_prob(log_prob_fn_single, prop1.reshape((C*W, D)), chunk=lik_chunk).reshape((C, W))
            acc1, _ = mh_accept_masked(k_st_p1, lp_local, prop_lp1, betas, logJ1, move1)
            th_local = jnp.where(acc1[..., None], prop1, th_local)
            lp_local = jnp.where(acc1,           prop_lp1, lp_local)
            att_slot = att_slot | move1                                                                     # <<< NEW
            acc_slot = acc_slot | acc1                                                                      # <<< NEW

            # Half B: BLUE moves (using UPDATED RED pool)
            prop2, logJ2, partner2, has_partner2, zfac2 = _propose_stretch_redblue(
                k_st_p2, k_st_z2,
                th_local,  # <<< FIX: propose from UPDATED thetas, not st_local.thetas
                subset_mask=blue, a=stretch_a, z=None
            )
            prop2 = _fold(prop2)                                                                   # <<< FIX
            move2 = blue & has_partner2
            prop_lp2 = batched_log_prob(log_prob_fn_single, prop2.reshape((C*W, D)), chunk=lik_chunk).reshape((C, W))
            acc2, _ = mh_accept_masked(k_st_p2, lp_local, prop_lp2, betas, logJ2, move2)
            th_local = jnp.where(acc2[..., None], prop2, th_local)
            lp_local = jnp.where(acc2,           prop_lp2, lp_local)
            att_slot = att_slot | move2                                                                     # <<< NEW
            acc_slot = acc_slot | acc2                                                                      # <<< NEW

            # write combined stretch slot once
            ids_arr = _log.ids.at[t, slot].set(jnp.int8(prop_id_stretch))
            att_arr = _log.attempted.at[t, slot].set(att_slot)
            acc_arr = _log.accepted.at[t, slot].set(acc_slot)
            log_out = EventLog(ids=ids_arr, attempted=att_arr, accepted=acc_arr)         # <<< FIX

            return PTState(th_local, lp_local), log_out

        def _stretch_no(args):
            _k, st_local, log_local = args
            ids_arr = log_local.ids.at[t, slot].set(jnp.int8(PROPOSAL_IDS["skipped"]))                      # <<< CHANGED
            log_out = EventLog(ids=ids_arr, attempted=log_local.attempted, accepted=log_local.accepted) 
            return st_local, log_out

        state, log = lax.cond(do_stretch, _stretch_yes, _stretch_no, (k, state, log))
        th, lp = state.thetas, state.log_probs

        # ---- 1) RW: pick one of 3 ----
        slot_rw = 1
        rw_kind_idx = random.categorical(k_rw, jnp.log(rw_probs))

        def _apply(prop, prop_id):
            # everyone proposes
            move_mask = jnp.ones((C, W), dtype=jnp.bool_)
            log_qcorr = jnp.zeros((C, W), dtype=lp.dtype)
            return apply_mh_masked(prop, log_qcorr, slot_rw, prop_id, k_rw, move_mask)

        def _rw_full(_):
            # _propose_fullcov(key, x, L, scale=None) — x:(C,W,D), L:(C,D,D), scale:(C,) or None
            prop = _propose_fullcov(k_rw, state.thetas, Ls, scale=scales["small"])
            return _apply(prop, PROPOSAL_IDS["rw_fullcov"])

        def _rw_eig(_):
            # _propose_eigenline(key, x, U, S, scale=None) — U:(C,D,D), S:(C,D)
            prop = _propose_eigenline(k_rw, k_el, state.thetas, evecs, evals_pd, scale=scales["line"])
            return _apply(prop, PROPOSAL_IDS["rw_eigenline"])

        def _rw_t(_):
            # _propose_student_t(key, x, L, df, scale=None) — uses L like fullcov
            prop = _propose_student_t(k_rw, k_el, state.thetas, Ls, scale=scales["big"])
            return _apply(prop, PROPOSAL_IDS["rw_student_t"])

        state, log = lax.switch(rw_kind_idx, [_rw_full, _rw_eig, _rw_t], operand=None)
        th, lp = state.thetas, state.log_probs

        # ---- 2) DE two-point (prob W_de and W>=3) ----
        slot_de = 2
        do_de = jnp.logical_and(W >= 3, random.bernoulli(k_de_p, W_de))      # <<< FIX: W>=3

        def _de_yes(args):
            _k, st_local, _log = args
            # Call proposals directly on the full ensemble (C, W, D)
            prop, idx_y, idx_z, has_pair, xover_mask = _propose_de_two_point(   # <<< CALL DIRECTLY
                    k_de_p, k_de_g, st_local.thetas,
                    gamma=None,                 # or pass your fixed gamma here
                    gamma_scale=gamma_de,           # tune as desired
                    crossover_rate=cross_rate,
                    jitter_scale=1e-6,
            )
            move_mask = has_pair                                               # (C, W)
            log_qcorr = jnp.zeros((C, W), dtype=st_local.log_probs.dtype)      # DE is symmetric
            return apply_mh_masked(prop, log_qcorr, slot_de,
                           PROPOSAL_IDS["de_two_point"], k_de_g, move_mask)

        def _de_no(args):
            _k, st_local, log_local = args
            ids_arr = log_local.ids.at[t, slot_de].set(jnp.int8(PROPOSAL_IDS["skipped"]))
            log_out = EventLog(ids=ids_arr, attempted=log_local.attempted, accepted=log_local.accepted)
            return st_local, log_out

        state, log = lax.cond(do_de, _de_yes, _de_no, (k, state, log))
        th, lp = state.thetas, state.log_probs

        # ---- 3) PT swap (combined even+odd) ----
        slot_pt = 3
        # NOTE: pt_swap_pass should return (state, acc_mask, att_mask)
        state, acc_e, att_e = pt_swap_pass(k_swap_e, state, betas, even_pass=True)                          # <<< CHANGED
        state, acc_o, att_o = pt_swap_pass(k_swap_o, state, betas, even_pass=False)                         # <<< CHANGED
        acc_pt = acc_e | acc_o                                                                              # <<< ADDED
        att_pt = att_e | att_o                                                                              # <<< ADDED
        ids_arr = log.ids.at[t, slot_pt].set(jnp.int8(PROPOSAL_IDS["pt_swap"]))                             # <<< CHANGED
        att_arr = log.attempted.at[t, slot_pt].set(att_pt)                                                  # <<< ADDED
        acc_arr = log.accepted.at[t, slot_pt].set(acc_pt)                                                   # <<< CHANGED
        log = EventLog(ids=ids_arr, attempted=att_arr, accepted=acc_arr)                                   # <<< CHANGED

        # ---------- RETURN per-step chain so we can build PTTrace ----------
        return (k, state, log), (state.thetas, state.log_probs)                                             # <<< CHANGED

    # ---------- SCAN & TRACE RECORDING ----------
    carry0 = (key, PTState(thetas0, lps0), log)
    # ys = (th_t, lp_t) per step
    (key_out, final_state, final_log), (th_steps, lp_steps) = lax.scan(one_step, carry0, jnp.arange(n_steps))   # <<< CHANGED

    trace = PTTrace(
        thetas=jnp.concatenate([thetas0[None, ...], th_steps], axis=0),                                       # <<< ADDED
        log_probs=jnp.concatenate([lps0[None, ...], lp_steps], axis=0),                                       # <<< ADDED
    )
    return final_state, trace, final_log



#### On-Off rate calculations 
