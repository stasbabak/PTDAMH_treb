# Utilities for product-space model switching within PTDAMH.
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax

__all__ = [
    "make_signal_indices",
    "make_logpost_M23",
    "update_z_with_rejuv",
    "run_epoch_device_fast_M23",
]


def make_signal_indices(Npar_src, D, with_rest=True):
    """Return index blocks for a three-source product space."""
    i1 = np.arange(0, Npar_src, dtype=np.int32)
    i2 = np.arange(Npar_src, 2 * Npar_src, dtype=np.int32)
    i3 = np.arange(2 * Npar_src, 3 * Npar_src, dtype=np.int32)
    if with_rest and D > 3 * Npar_src:
        rest = np.setdiff1d(
            np.arange(D, dtype=np.int32), np.concatenate([i1, i2, i3])
        )
    else:
        rest = np.empty((0,), dtype=np.int32)
    return i1, i2, i3, rest


def _gather_cols(X, idx):
    """Gather the selected columns from ``X`` in a JIT-friendly manner."""
    idx = jnp.asarray(idx, jnp.int32)
    return jnp.take(X, idx, axis=1) if idx.size else jnp.zeros((X.shape[0], 0), X.dtype)


def make_logpost_M23(
    loglik_M2_single,
    loglik_M3_single,
    idx1,
    idx2,
    idx3,
    idx_rest=(),
    log_prior_z=(0.0, 0.0),
):
    """Build a batched log-posterior for models M2 and M3."""
    f2 = jax.jit(jax.vmap(loglik_M2_single))
    f3 = jax.jit(jax.vmap(loglik_M3_single))
    lp0, lp1 = map(float, log_prior_z)

    @jax.jit
    def logpost(X, z):
        X1 = _gather_cols(X, idx1)
        X2 = _gather_cols(X, idx2)
        X3 = _gather_cols(X, idx3)
        R = _gather_cols(X, idx_rest)
        lp2 = f2(jnp.concatenate([X1, X2, R], axis=1))
        lp3 = f3(jnp.concatenate([X1, X2, X3, R], axis=1))
        return jnp.where(z.astype(bool), lp3 + lp1, lp2 + lp0)

    return logpost


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
    idx_rest=(),
    log_prior_z=(0.0, 0.0),
):
    """Update the model indicator ``z`` with simple rejuvenation."""
    C, _ = X.shape
    k_rejuv, k_flip = random.split(key)

    if len(idx3):
        U = random.uniform(k_rejuv, (C, len(idx3)))
        X = X.at[:, jnp.asarray(idx3)].set(jnp.where(z[:, None] == 0, U, X[:, jnp.asarray(idx3)]))

    f2 = jax.vmap(loglik_M2_single)
    f3 = jax.vmap(loglik_M3_single)
    X1 = _gather_cols(X, idx1)
    X2 = _gather_cols(X, idx2)
    X3 = _gather_cols(X, idx3)
    R = _gather_cols(X, idx_rest)
    lp2 = f2(jnp.concatenate([X1, X2, R], axis=1))
    lp3 = f3(jnp.concatenate([X1, X2, X3, R], axis=1))
    lp0, lp1 = map(float, log_prior_z)

    logits = (lp3 + lp1) - (lp2 + lp0)
    p1 = jax.nn.sigmoid(logits)
    z_new = random.bernoulli(k_flip, p1).astype(jnp.int32)
    lp_new = jnp.where(z_new.astype(bool), lp3 + lp1, lp2 + lp0)
    return X, z_new, lp_new, p1


def _apply_swaps_vec(arr, i, j, accept):
    ai, aj = arr[i], arr[j]
    return arr.at[i].set(jnp.where(accept, aj, ai)).at[j].set(jnp.where(accept, ai, aj))


def run_epoch_device_fast_M23(
    key,
    init_state,
    temperatures,
    loglik_M2_single,
    loglik_M3_single,
    idx1,
    idx2,
    idx3,
    idx_rest=(),
    # ... all your existing args for proposals ...
    model_update_stride: int = 5,
    log_prior_z=(0.0, 0.0),
    swap_fn=None,
    # ...
):
    """Version of ``run_epoch_device_fast`` with model indicator updates."""
    if swap_fn is None:
        from PTDAMH import parallel_tempering_swap as swap_fn  # lazy import

    C, _ = init_state.thetas.shape
    logpost = make_logpost_M23(
        loglik_M2_single, loglik_M3_single, idx1, idx2, idx3, idx_rest, log_prior_z
    )

    do_model = jnp.arange(n_steps) % model_update_stride == 0

    def body(carry, xs):
        (th, lp, z), (key_t, comp_idx_t, do_m) = carry, xs
        k0, k1, k2, k3, k4, kU, kS, kZ = random.split(key_t, 8)

        # --- propose as you already do ---
        # prop0/1/2 = student-t / eigen-line / pCN ...
        props_all = jnp.stack([prop0, prop1, prop2], axis=0)
        proposals = props_all[comp_idx_t, jnp.arange(C), :]
        proposals = _fold_params(proposals, fold_mask=fold_mask, period=period)

        # --- MH under current z ---
        prop_lp = logpost(proposals, z)
        log_alpha = (prop_lp - lp) / temperatures
        ulog = jnp.log(random.uniform(kU, (C,)))
        accept = ulog < log_alpha

        th = jnp.where(accept[:, None], proposals, th)
        lp = jnp.where(accept, prop_lp, lp)

        # --- Carlin–Chib z update on stride ---
        def _do(args):
            th_, z_, lp_ = args
            th2, z2, lp2, p1 = update_z_with_rejuv(
                kZ,
                th_,
                z_,
                loglik_M2_single,
                loglik_M3_single,
                idx1,
                idx2,
                idx3,
                idx_rest,
                log_prior_z,
            )
            return (th2, z2, lp2), p1

        def _skip(args):
            th_, z_, lp_ = args
            return (th_, z_, lp_), jnp.zeros((C,), th.dtype)

        (th, z, lp), p1_dbg = jax.lax.cond(do_m, _do, _skip, (th, z, lp))

        # --- PT swap (need to swap z as well) ---
        key_s, th_sw, lp_sw, raster, dbg = swap_fn(
            kS, temperatures, th, lp, return_debug=True
        )
        i, j, acc_sel = dbg["pairs_i"], dbg["pairs_j"], dbg["accept_sel"]
        z_sw = _apply_swaps_vec(z, i, j, acc_sel)

        info_step = (proposals, prop_lp, accept, comp_idx_t, raster, th_sw, z_sw)
        return (th_sw, lp_sw, z_sw), info_step

    keys = random.split(key, n_steps)
    comp_seq = ...  # your per-step (T,C) component choices, or keep per-epoch fixed like before
    xs = (keys, comp_seq, do_model)
    (th_f, lp_f, z_f), outs = lax.scan(
        body, (init_state.thetas, init_state.log_probs, init_state.z), xs
    )

    # unpack outs (props, prop_lps, accepts, comp_idxs, swaps, th_hist, z_hist) = outs
    # return final state + info; record z_hist in InfoAccumulator (see below)

    return th_f, lp_f, z_f, outs
