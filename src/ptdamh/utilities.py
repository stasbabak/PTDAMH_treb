"""Utility helpers and data containers for PTDAMH."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random


# ------------------------- Temperature/covariance helpers -------------------------


def temperature_ladder(
    n_temps=50, T_min=1.0, T_max=100.0, kind="geom", cold_dense=False, power=2.0
):
    """
    kind: "geom" = geometric in T
    cold_dense: if True, densify near T=1 by warping the index
    power: >1 puts more points near the cold chain
    """
    i = jnp.arange(n_temps, dtype=jnp.float32)
    if cold_dense:
        # warp indices to cluster more temperatures near i=0 (cold end)
        i = ((i / (n_temps - 1)) ** power) * (n_temps - 1)
    r = (T_max / T_min) ** (1.0 / (n_temps - 1))
    T = T_min * (r**i)
    return T


def _empirical_cov(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[0] <= 1:
        d = x.shape[1]
        return np.eye(d, dtype=x.dtype) * 1e-3
    return np.cov(x, rowvar=False, ddof=ddof)


def _shrink_spd(
    cov_hat: np.ndarray, shrink: float = 0.1, jitter: float = 1e-6
) -> np.ndarray:
    cov_hat = np.asarray(cov_hat)
    d = np.diag(np.diag(cov_hat))
    cov = (1.0 - shrink) * cov_hat + shrink * d
    cov = 0.5 * (cov + cov.T)
    eps = float(max(np.max(np.diag(cov)), 1.0)) * jitter
    cov += eps * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov


def _circular_mean(vals, period=1.0):
    ang = 2.0 * np.pi * (vals / period)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    mu_ang = np.arctan2(s, c) % (2 * np.pi)
    return mu_ang * (period / (2 * np.pi))


def _wrapped_diff(vals, mu, period=1.0):
    d = vals - mu
    return (d + 0.5 * period) % period - 0.5 * period


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


# ------------------------- Data containers -------------------------


class PTState(NamedTuple):
    thetas: jnp.ndarray  # (C, D)
    log_probs: jnp.ndarray  # (C,)  (log-target at current thetas)
    temperatures: jnp.ndarray  # (C,)
    n_accepted: jnp.ndarray  # (C,)
    n_swaps: jnp.ndarray  # (C-1,) counters of accepted swaps per edge (optional)
    n_swap_attempts: jnp.ndarray  # (C-1,) total swap attempts per edge
    z: jnp.ndarray | None = None  # (C,) int32, 0=M2, 1=M3


class StepInfo(NamedTuple):
    thetas_prop: jnp.ndarray  # (C, D)
    logprob_prop: jnp.ndarray  # (C,)
    accepted: jnp.ndarray  # (C,) bool
    comp_idx: jnp.ndarray  # (C,) int in [0..K-1]
    swap_decisions: jnp.ndarray  # (C-1,) bool


# ------------------------- Parallel tempering swap -------------------------


@jax.jit
def _pt_swap_core(key, temperatures, thetas, log_probs):
    C = thetas.shape[0]
    beta = 1.0 / jnp.asarray(temperatures)
    n_edges = C - 1

    i_even = jnp.arange(0, n_edges, 2, dtype=jnp.int32)  # Ke
    i_odd = jnp.arange(1, n_edges, 2, dtype=jnp.int32)  # Ko
    Ke = i_even.shape[0]
    Ko = i_odd.shape[0]
    pad = Ke - Ko
    i_odd_padded = jnp.concatenate([i_odd, -jnp.ones((pad,), dtype=jnp.int32)], axis=0)
    valid_even = jnp.ones((Ke,), dtype=bool)
    valid_odd = jnp.arange(Ke) < Ko

    key, k_par = random.split(key)
    key, k_u = random.split(key)
    parity = random.bernoulli(k_par)

    i_raw, valid = jax.lax.cond(
        parity, lambda: (i_odd_padded, valid_odd), lambda: (i_even, valid_even)
    )  # (Ke,)

    i = jnp.where(valid, i_raw, jnp.zeros_like(i_raw)).astype(jnp.int32)
    j = (i + 1).astype(jnp.int32)

    # Δ = (β_i - β_j) * (lp_j - lp_i)
    delta = (beta[i] - beta[j]) * (log_probs[j] - log_probs[i])  # (Ke,)
    delta = jnp.where(valid, delta, -jnp.inf)
    ulog = jnp.log(random.uniform(k_u, shape=delta.shape))
    accept_sel = (ulog < delta) & valid  # (Ke,) bool

    def _swap_rows(arr):
        ai, aj = arr[i], arr[j]
        if arr.ndim == 1:
            acc = accept_sel
            arr = arr.at[i].set(jnp.where(acc, aj, ai))
            arr = arr.at[j].set(jnp.where(acc, ai, aj))
        else:
            acc_b = accept_sel.reshape(accept_sel.shape + (1,) * (ai.ndim - 1))
            arr = arr.at[i].set(jnp.where(acc_b, aj, ai))
            arr = arr.at[j].set(jnp.where(acc_b, ai, aj))
        return arr

    thetas_new = _swap_rows(thetas)  # (C,D)
    logp_new = _swap_rows(log_probs)  # (C,)

    raster = (
        jnp.zeros((n_edges,), dtype=bool).at[i].set(jnp.where(valid, accept_sel, False))
    )
    # Always return the raw debug arrays (no dicts inside jit)
    return key, thetas_new, logp_new, raster, i, j, accept_sel, delta, ulog, parity


@jax.jit
def _pt_swap_core_parity(key, temperatures, thetas, log_probs, parity: jnp.bool_):
    C = thetas.shape[0]
    beta = 1.0 / jnp.asarray(temperatures)
    n_edges = C - 1

    i_even = jnp.arange(0, n_edges, 2, dtype=jnp.int32)  # Ke
    i_odd = jnp.arange(1, n_edges, 2, dtype=jnp.int32)  # Ko
    Ke = i_even.shape[0]
    Ko = i_odd.shape[0]
    pad = Ke - Ko
    i_odd_padded = jnp.concatenate([i_odd, -jnp.ones((pad,), dtype=jnp.int32)], axis=0)
    valid_even = jnp.ones((Ke,), dtype=bool)
    valid_odd = jnp.arange(Ke) < Ko

    # choose pair set deterministically via parity
    i_raw, valid = jax.lax.cond(
        parity, lambda: (i_odd_padded, valid_odd), lambda: (i_even, valid_even)
    )  # (Ke,)

    i = jnp.where(valid, i_raw, jnp.zeros_like(i_raw)).astype(jnp.int32)
    j = (i + 1).astype(jnp.int32)

    # draw uniforms for this pass
    key, k_u = random.split(key)

    # Δ = (β_i - β_j) * (lp_j - lp_i)
    delta = (beta[i] - beta[j]) * (log_probs[j] - log_probs[i])
    delta = jnp.where(valid, delta, -jnp.inf)
    ulog = jnp.log(random.uniform(k_u, shape=delta.shape))
    accept_sel = (ulog < delta) & valid

    def _swap_rows(arr):
        ai, aj = arr[i], arr[j]
        if arr.ndim == 1:
            acc = accept_sel
            arr = arr.at[i].set(jnp.where(acc, aj, ai))
            arr = arr.at[j].set(jnp.where(acc, ai, aj))
        else:
            acc_b = accept_sel.reshape(accept_sel.shape + (1,) * (ai.ndim - 1))
            arr = arr.at[i].set(jnp.where(acc_b, aj, ai))
            arr = arr.at[j].set(jnp.where(acc_b, ai, aj))
        return arr

    thetas_new = _swap_rows(thetas)
    logp_new = _swap_rows(log_probs)

    raster = (
        jnp.zeros((n_edges,), dtype=bool).at[i].set(jnp.where(valid, accept_sel, False))
    )
    return key, thetas_new, logp_new, raster, i, j, accept_sel, delta, ulog


def parallel_tempering_swap(
    key,
    temperatures,
    thetas,
    log_probs,
    *,
    return_debug=False,
    two_sweeps=True,
):
    if not two_sweeps:
        # original single-sweep behavior
        key2, th2, lp2, raster, i, j, acc_sel, delta, ulog, parity = _pt_swap_core(
            key, temperatures, thetas, log_probs
        )
        if return_debug:
            dbg = {
                "pairs_i": i,
                "pairs_j": j,
                "accept_sel": acc_sel,
                "delta": delta,
                "ulog": ulog,
                "parity": parity,
            }
            return key2, th2, lp2, raster, dbg
        return key2, th2, lp2, raster

    # -------- two sequential sweeps: EVEN then ODD --------
    key, k1, k2 = random.split(key, 3)

    # pass 1: EVEN (parity=False)
    k1, th1, lp1, ras1, i1, j1, acc1, d1, u1 = _pt_swap_core_parity(
        k1, temperatures, thetas, log_probs, parity=jnp.array(False)
    )
    # pass 2: ODD (parity=True), on the *permuted* arrays (no model calls)
    k2, th2, lp2, ras2, i2, j2, acc2, d2, u2 = _pt_swap_core_parity(
        k2, temperatures, th1, lp1, parity=jnp.array(True)
    )

    raster = ras1 | ras2
    if return_debug:
        dbg = {
            "even": {
                "pairs_i": i1,
                "pairs_j": j1,
                "accept_sel": acc1,
                "delta": d1,
                "ulog": u1,
            },
            "odd": {
                "pairs_i": i2,
                "pairs_j": j2,
                "accept_sel": acc2,
                "delta": d2,
                "ulog": u2,
            },
        }
        return k2, th2, lp2, raster, dbg
    return k2, th2, lp2, raster


# ------------------------- VMAP and parameter utilities -------------------------


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


def _fold_params(
    x: jnp.ndarray, fold_mask: jnp.ndarray | None, period: float
) -> jnp.ndarray:
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


# --- indices and pseudo-priors ---


def _make_indices_equal_blocks(Npar_src: int, D: int):
    i1 = jnp.arange(0, Npar_src, dtype=jnp.int32)
    i2 = jnp.arange(Npar_src, 2 * Npar_src, dtype=jnp.int32)
    i3 = jnp.arange(2 * Npar_src, 3 * Npar_src, dtype=jnp.int32)
    # any remaining dims are "rest" (shared by both models)
    if 3 * Npar_src < D:
        rest = jnp.arange(3 * Npar_src, D, dtype=jnp.int32)
    else:
        rest = jnp.empty((0,), dtype=jnp.int32)
    return i1, i2, i3, rest


def _psi3_uniform_sample(key, C, d3):
    return random.uniform(key, (C, d3))


def _psi3_uniform_logpdf(theta3):
    # constant 0 inside [0,1]; you already fold/box -> we return zeros
    return jnp.zeros((theta3.shape[0],), dtype=theta3.dtype)


# ---------- product-space logposterior & z-Gibbs (+rejuvenation) ----------


def _gather_cols(X, idx):  # X:(C,D), idx:(k,) -> (C,k) (works with k=0)
    return jnp.take(X, idx, axis=1) if idx.size else jnp.zeros((X.shape[0], 0), X.dtype)


def make_logpost_M23(
    loglik_M2_single,
    loglik_M3_single,
    idx1,
    idx2,
    idx3,
    idx_rest,
    log_prior_z=(0.0, 0.0),
    psi3_logpdf=_psi3_uniform_logpdf,
):
    f2 = jax.jit(jax.vmap(loglik_M2_single))
    f3 = jax.jit(jax.vmap(loglik_M3_single))
    lp0, lp1 = map(float, log_prior_z)

    @jax.jit
    def _logpost(X, z):
        X1 = _gather_cols(X, idx1)
        X2 = _gather_cols(X, idx2)
        X3 = _gather_cols(X, idx3)
        R = _gather_cols(X, idx_rest)
        lp2 = f2(jnp.concatenate([X1, X2, R], axis=1))
        lp3 = f3(jnp.concatenate([X1, X2, X3, R], axis=1))
        lp_psi3 = psi3_logpdf(X3)
        return jnp.where(z.astype(bool), lp3 + lp1, lp2 + lp0 + lp_psi3)

    return _logpost


@jax.jit
def update_z_with_rejuv(
    key,
    X,
    z,
    loglik_M2_single,
    loglik_M3_single,
    idx1,
    idx2,
    idx3,
    idx_rest,
    log_prior_z=(0.0, 0.0),
    psi3_sample=_psi3_uniform_sample,
    psi3_logpdf=_psi3_uniform_logpdf,
):
    C, D = X.shape
    k_rejuv, k_flip = random.split(key)
    # rejuvenate θ3 ~ ψ when z==0
    if idx3.size:
        d3 = int(idx3.size)
        fresh = psi3_sample(k_rejuv, C, d3)
        X = X.at[:, idx3].set(jnp.where(z[:, None] == 0, fresh, X[:, idx3]))

    f2 = jax.vmap(loglik_M2_single)
    f3 = jax.vmap(loglik_M3_single)
    X1 = _gather_cols(X, idx1)
    X2 = _gather_cols(X, idx2)
    X3 = _gather_cols(X, idx3)
    R = _gather_cols(X, idx_rest)

    lp2 = f2(jnp.concatenate([X1, X2, R], axis=1))
    lp3 = f3(jnp.concatenate([X1, X2, X3, R], axis=1))
    lp_psi3 = psi3_logpdf(X3)
    lp0, lp1 = map(float, log_prior_z)

    logits = (lp3 + lp1) - (lp2 + lp0 + lp_psi3)
    p1 = jax.nn.sigmoid(logits)
    z_new = random.bernoulli(k_flip, p1).astype(jnp.int32)
    lp_new = jnp.where(z_new.astype(bool), lp3 + lp1, lp2 + lp0 + lp_psi3)
    return X, z_new, lp_new, p1


def _apply_swaps_vec(arr, i, j, accept):
    ai, aj = arr[i], arr[j]
    acc = accept.astype(bool)
    if arr.ndim == 1:
        out = arr.at[i].set(jnp.where(acc, aj, ai))
        out = out.at[j].set(jnp.where(acc, ai, aj))
        return out
    else:
        acc_b = acc.reshape(acc.shape + (1,) * (ai.ndim - 1))
        out = arr.at[i].set(jnp.where(acc_b, aj, ai))
        out = out.at[j].set(jnp.where(acc_b, ai, aj))
        return out


__all__ = [
    "temperature_ladder",
    "_empirical_cov",
    "_shrink_spd",
    "_circular_mean",
    "_wrapped_diff",
    "_empirical_cov_wrapped",
    "PTState",
    "StepInfo",
    "_pt_swap_core",
    "_pt_swap_core_parity",
    "parallel_tempering_swap",
    "_batched_logprob_chunked_fn",
    "_fold_params",
    "_make_indices_equal_blocks",
    "_psi3_uniform_sample",
    "_psi3_uniform_logpdf",
    "_gather_cols",
    "_apply_swaps_vec",
    "make_logpost_M23",
    "update_z_with_rejuv",
]
