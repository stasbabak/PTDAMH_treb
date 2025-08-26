# Archived `parallel_tempering_swap` implementations

The repository previously contained experimental versions of `parallel_tempering_swap`. They are preserved here for reference.

## Initial simple sweep

```python
@jax.jit
def parallel_tempering_swap(key, temperatures: jnp.ndarray,
                            thetas: jnp.ndarray, log_probs: jnp.ndarray):
    """Single adjacent-swap sweep (even-odd could be added if desired).

    Returns updated (thetas, log_probs), plus boolean decisions per edge.
    """
    C = temperatures.shape[0]
    betas = 1.0 / temperatures
    # Δβ * ΔlogL between neighbors (using current log_probs ~ log target)
    dlog = (betas[1:] - betas[:-1]) * (log_probs[1:] - log_probs[:-1])

    key, ukey = random.split(key)
    u = jnp.log(random.uniform(ukey, shape=(C - 1,)))
    do_swap = dlog >= u

    def swap_edge(i, carry):
        th, lp = carry
        def yes():
            th1 = th.at[i].set(th[i + 1])
            th1 = th1.at[i + 1].set(th[i])
            lp1 = lp.at[i].set(lp[i + 1])
            lp1 = lp1.at[i + 1].set(lp[i])
            return th1, lp1
        return lax.cond(do_swap[i], yes, lambda: (th, lp))

    thetas2, logp2 = lax.fori_loop(0, C - 1, swap_edge, (thetas, log_probs))
    return key, thetas2, logp2, do_swap
```

## Vectorized sweep with parity selection

```python
def _apply_swaps_vec(arr, i, j, accept):
    """
    arr: (C, ...) values to swap
    i,j: (K,) pair indices
    accept: (K,) booleans
    """
    ai, aj = arr[i], arr[j]
    acc = accept.astype(bool)
    if arr.ndim == 1:
        out = arr.at[i].set(jnp.where(acc, aj, ai))
        out = out.at[j].set(jnp.where(acc, ai, aj))
        return out
    else:
        # Expand mask to (K, 1, 1, ..., 1) to match ai/aj
        acc_b = acc.reshape(acc.shape + (1,)*(ai.ndim - 1))
        out = arr.at[i].set(jnp.where(acc_b, aj, ai))
        out = out.at[j].set(jnp.where(acc_b, ai, aj))
        return out

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
    i = i.astype(jnp.int32)
    j = j.astype(jnp.int32)

    # Correct MH exponent for swaps: Δ = (β_i - β_j) * (lp_j - lp_i)
    delta = (beta[i] - beta[j]) * (log_probs[j] - log_probs[i])     # (Ke,)
    delta = jnp.where(valid, delta, -jnp.inf)                        # mask padded
    ulog  = jnp.log(random.uniform(k_u, shape=delta.shape))
    accept_sel = ulog < delta
    accept_sel = accept_sel.astype(bool)                             # (Ke,)

    # Apply swaps simultaneously on the selected (non-overlapping) pairs
    thetas_new = _apply_swaps_vec(thetas,    i, j, accept_sel)
    logp_new   = _apply_swaps_vec(log_probs, i, j, accept_sel)

    # Raster of decisions over all edges (C-1,)
    raster = jnp.zeros((n_edges,), dtype=bool).at[i].set(jnp.where(valid, accept_sel, False))

    if return_debug:
        dbg = {"delta": delta, "ulog": ulog, "pairs_i": i, "pairs_j": j,
               "accept_sel": accept_sel, "parity": parity}
        return key, thetas_new, logp_new, raster, dbg
    return key, thetas_new, logp_new, raster
```
