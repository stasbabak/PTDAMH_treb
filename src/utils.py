"""Utility functions for parallel tempering swaps."""

import jax
import jax.numpy as jnp
from jax import lax, random


def _apply_swaps_vec(arr, i, j, accept):
    """Apply a set of non-overlapping swaps to ``arr``.

    Parameters
    ----------
    arr : jnp.ndarray
        Array of shape ``(C, ...)`` holding values to swap.
    i, j : jnp.ndarray
        Arrays of shape ``(K,)`` with the left and right indices of each swap.
    accept : jnp.ndarray
        Boolean array of shape ``(K,)`` indicating which swaps to perform.
    """

    ai, aj = arr[i], arr[j]
    return arr.at[i].set(jnp.where(accept, aj, ai)).at[j].set(jnp.where(accept, ai, aj))


@jax.jit
def parallel_tempering_swap(key, temperatures, thetas, log_probs, *, return_debug=False):
    """Perform a single sweep of parallel-tempering swaps.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key.
    temperatures : jnp.ndarray
        Array of shape ``(C,)`` with absolute temperatures ``T``.
    thetas : jnp.ndarray
        Current states of shape ``(C, D)``.
    log_probs : jnp.ndarray
        Log-probabilities at each state, shape ``(C,)``.
    return_debug : bool, optional
        If ``True`` return a debug dictionary with intermediate values.
    """

    C = thetas.shape[0]
    beta = 1.0 / jnp.asarray(temperatures)
    n_edges = C - 1

    # Build even/odd adjacent index sets; pad odd to same static length
    i_even = jnp.arange(0, n_edges, 2, dtype=jnp.int32)
    i_odd = jnp.arange(1, n_edges, 2, dtype=jnp.int32)
    Ke = i_even.shape[0]
    Ko = i_odd.shape[0]
    pad = Ke - Ko
    i_odd_padded = jnp.concatenate([i_odd, -jnp.ones((pad,), dtype=jnp.int32)], axis=0)
    valid_even = jnp.ones((Ke,), dtype=bool)
    valid_odd = jnp.arange(Ke) < Ko

    # RNG: advance key and get subkeys
    key, k_par = random.split(key)
    key, k_u = random.split(key)
    parity = random.bernoulli(k_par)  # False->even, True->odd

    def pick_even():
        return i_even, valid_even

    def pick_odd():
        return i_odd_padded, valid_odd

    i_raw, valid = lax.cond(parity, pick_odd, pick_even)
    i = jnp.where(valid, i_raw, jnp.zeros_like(i_raw))
    j = i + 1

    delta = (beta[i] - beta[j]) * (log_probs[j] - log_probs[i])
    delta = jnp.where(valid, delta, -jnp.inf)
    ulog = jnp.log(random.uniform(k_u, shape=delta.shape))
    accept_sel = ulog < delta

    thetas_new = _apply_swaps_vec(thetas, i, j, accept_sel)
    logp_new = _apply_swaps_vec(log_probs, i, j, accept_sel)

    raster = jnp.zeros((n_edges,), dtype=bool).at[i].set(jnp.where(valid, accept_sel, False))

    if return_debug:
        dbg = {
            "delta": delta,
            "ulog": ulog,
            "pairs_i": i,
            "pairs_j": j,
            "accept_sel": accept_sel,
            "parity": parity,
        }
        return key, thetas_new, logp_new, raster, dbg
    return key, thetas_new, logp_new, raster

