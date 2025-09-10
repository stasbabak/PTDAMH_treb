# src/ptdamh/product_space_ctmc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.scipy.special import gammaln

Array = jnp.ndarray

from .epoch import (
    PTState,
    PTTrace,
    PTTrace2,
    EventLog,
    PROPOSAL_IDS,
    run_epoch,
    batched_log_prob,
    mh_accept_masked,
)

from .proposals import (
    redblue_mask,
    _propose_stretch_redblue,
    _propose_fullcov,
    _propose_eigenline,
    _propose_student_t,
    _propose_de_two_point,
)


# ---------- State ----------
@dataclass
class PSState:
    # Component parameters for ALL slots (active or inactive):
    phi: Array         # (C, W, Kmax, d)
    # Optional "rest" parameters (kept as-is by on/off):
    rest: Array | None # (C, W, Drest) or None
    # Mask m:
    m: Array           # (C, W, Kmax) bool
    # Cached tempered log target \tilde{\pi}_beta(m,phi)
    logpi: Array       # (C, W)

@dataclass
class BDEvent:
    dt: Array          # (C, W) holding time ~ Exp(Λ)
    kind: Array        # (C, W) int8: 0=on(birth), 1=off(death), 2=no-event
    index: Array       # (C, W) int32: which slot toggled (undefined if no-event)
    k_before: Array    # (C, W)
    k_after: Array     # (C, W)



@dataclass
class BDSingleEvent:
    dt: jnp.ndarray        # () scalar holding time
    kind: jnp.int8         # 0=birth, 1=death
    c: jnp.int32
    w: jnp.int32
    j: jnp.int32
    k_before: jnp.ndarray  # (C,W)
    k_after:  jnp.ndarray  # (C,W)


# ---------- Building blocks you plug in ----------
# These are UNTEMPERED logs. We temper the likelihood internally by beta[c].

# log p(phi) for an active slot (component prior)
LogPriorPhi = Callable[[Array], Array]           # (d,) -> ()
# log ψ(phi) for an inactive slot (pseudo-prior)
LogPseudoPhi = Callable[[Array], Array]          # (d,) -> ()
# log-likelihood given mask & all components (untampered)
# Must IGNORE inactive slots according to m
LogLikMasked = Callable[[Array, Array, Array | None], Array]   # (Kmax,d), (Kmax,), rest -> ()
# prior over k (number of actives), returns log p(k)
LogPk = Callable[[Array], Array]                 # ()int -> ()
# normalized birth density evaluator q_b at a given context mask
QbDensity = Callable[[Array, Array, Array, Array | None], Array]
#    phi_i (d,), m_ctx (Kmax,), phi_all (Kmax,d), rest -> density value in R+
"""
Input of QbDensity
- phi_i: the parameter vector for the active slot (d,)
- m_ctx: the context mask indicating active slots which we evaluate (Kmax,)
    in child evaluated m_ctx -> m^{(+i)} (bit i is toggled "on")
    in parent evaluated m_ctx -> m (current mask)
- phi_all: the parameter matrix for all slots (active and not) (Kmax,d)
- rest: optional additional parameters (Drest,)

Output of QbDensity
- density value of q_b(\phi_i | state)  
"""


# ---------- Helpers ----------
def _log_uniform_masks_given_k(Kmax: int, k: Array) -> Array:
    # log p(m | k) = -log binom(Kmax, k)
    k = jnp.asarray(k, dtype=jnp.int32)
    return - (gammaln(Kmax + 1.) - gammaln(k + 1.) - gammaln(Kmax - k + 1.))

def _log_symmetrization(k: Array) -> Array:
    # log (1/k!) = -log(k!)
    k = jnp.asarray(k, dtype=jnp.float32)
    return -gammaln(k + 1.0)

### or pre-compute and use the look-up table:
def make_logfactorial_table(Kmax: int, *, dtype=jnp.float32) -> jnp.ndarray:
    n = jnp.arange(Kmax + 1, dtype=dtype)
    return gammaln(n + 1.)  # shape (Kmax+1,)

def _log_uniform_masks_given_k_tab(k: jnp.ndarray, logfact: jnp.ndarray) -> jnp.ndarray:
    """
    Same as above, but uses a precomputed log-factorial table.
    """
    Kmax = int(logfact.shape[0] - 1)
    k = jnp.asarray(k, dtype=jnp.int32)
    invalid = (k < 0) | (k > Kmax)
    val = -(logfact[Kmax] - logfact[k] - logfact[Kmax - k])
    return jnp.where(invalid, -jnp.inf, val)

def _log_symmetrization_tab(k: jnp.ndarray, logfact: jnp.ndarray) -> jnp.ndarray:
    k = jnp.asarray(k, dtype=jnp.int32)
    return -logfact[k]  # because log(k!) = logfact[k]
###############################################################

def _count_k(m: Array) -> Array:
    return m.astype(jnp.int32).sum(axis=-1)   # sum over Kmax

def _logpi_extended_untempered(
    phi_kwd: Array,                 # (Kmax,d)
    m_k: Array,                     # (Kmax,) bool
    rest_d: Array | None,           # (Drest,) or None
    log_prior_phi: LogPriorPhi,
    log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked,
    log_p_k: LogPk,
    Kmax: int,
) -> Array:
    """Un-tempered log \tilde{\pi} (no beta yet)."""
    k = m_k.astype(jnp.int32).sum()
    # component terms
    def _one(i):
        logp = log_prior_phi(phi_kwd[i])
        logpsi = log_pseudo_phi(phi_kwd[i])
        return jnp.where(m_k[i], logp, logpsi)
    comp = jax.vmap(_one)(jnp.arange(Kmax)).sum()  # sum over priors and pseudo-priors

    ll = log_lik_masked(phi_kwd, m_k, rest_d)  # uses only active slots in log-likelihood

    return ( log_p_k(k)
           + _log_uniform_masks_given_k(Kmax, k)
           + _log_symmetrization(k)
           + comp + ll ) # propto log probability (see notes)

def _logpi_tempered(
    phi: Array, m: Array, rest: Array | None,
    betas: Array,                        # (C,)
    log_prior_phi: LogPriorPhi,
    log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked,
    log_p_k: LogPk,
    Kmax: int,
) -> Array:
    """Tempered: prior terms unscaled, likelihood scaled by beta[c]."""
    C, W, Kmax_local, d = phi.shape
    assert Kmax_local == Kmax
    beta_cw = betas[:, None]  # (C,1)
    # compute untempered logpi and untempered log-likelihood separately to scale
    def _per_chain(phi_cw, m_cw, rest_cw, beta):
        # split: compute untempered logpi and ll
        def ll_only(phi_kwd, m_k, rest_d):  # convenience
            return log_lik_masked(phi_kwd, m_k, rest_d)

        # untempered total
        logpi_un = _logpi_extended_untempered(phi_cw, m_cw, rest_cw,
                                              log_prior_phi, log_pseudo_phi,
                                              log_lik_masked=lambda a,b,c: 0.0,   # exclude LL
                                              log_p_k=log_p_k, Kmax=Kmax)
        # add tempered LL
        ll_un = ll_only(phi_cw, m_cw, rest_cw)
        return logpi_un + beta * ll_un
    f = jax.vmap(jax.vmap(_per_chain, in_axes=(0,0,0, None)), in_axes=(0,0,0,0))
    return f(phi, m, rest if rest is not None else jnp.zeros((C,W,0)), betas)
    # If rest=None we pass zeros; your log_lik_masked should ignore it.


# ---------- Rejection-free BD (on-off) step (child/parent evaluated) ----------
def onoff_ct_step(
    key: random.PRNGKey,
    state: PSState,
    *,
    betas: Array,                           # (C,)
    Kmax: int,
    qb_density: Callable[[Array, Array, Array, Array|None], Array],
    qb_eval_variant: Literal["child","parent"] = "child",
    log_prior_phi: Callable[[Array], Array],
    log_pseudo_phi: Callable[[Array], Array],
    log_lik_masked: Callable[[Array, Array, Array|None], Array],
    log_p_k: Callable[[Array], Array],
    sample_pseudo_phi: Callable[[random.PRNGKey], Array],     # sampling from pseudo-prior
) -> Tuple[PSState, BDEvent]:
    """
        One rejection-free BD step per (C,W) chain:

        Child-evaluated:
            λ_on_j = β q_b(φ_j | m^{(+j)}, ·), for m_j=0
            λ_off_i = β q_b(φ_i | m, ·) * exp( logπ(m^{(-i)}) - logπ(m) ), for m_i=1

        Parent-evaluated:
            λ_on_j = β q_b(φ_j | m, ·), for m_j=0
            λ_off_i = β q_b(φ_i | m^{(-i)}, ·) * exp( logπ(m^{(-i)}) - logπ(m) ), for m_i=1

        Detailed balance holds edge-wise, so the chosen event is always taken.
    """

    phi, m, rest, logpi_cur = state.phi, state.m, state.rest, state.logpi

    C, W, Kmax_local, d = phi.shape
    assert Kmax_local == Kmax
    beta_cw = betas[:, None]
    CW = C * W

    # -------- precompute logpi(m^{(-i)}) for all active i --------
    # build masks with slot i turned off
    arK = jnp.arange(Kmax)

    def turn_off_mask(m_cw, i):
        return m_cw.at[i].set(False)

    # compute off-state for each \phi_i (even if it is in off-state): vectorisation of a fixed size -- simplicity
    def logpi_off_all(phi_cw, m_cw, rest_cw, beta):
        # compute logpi for each i turned off
        def one_i(i):
            m_off = turn_off_mask(m_cw, i)
            return _logpi_tempered(phi_cw[None,None,:,:], m_off[None,None,:],
                                   rest_cw[None,None,...] if rest_cw is not None else None,
                                   jnp.array([beta])[None,...],
                                   log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax
                                   )[0,0]
        return jax.vmap(one_i)(arK)  # (Kmax,)
    logpi_off = jax.vmap(jax.vmap(logpi_off_all, in_axes=(0,0,0, None)),
                         in_axes=(0,0,0,0))(phi, m, rest if rest is not None else jnp.zeros((C,W,0)), betas)
    # shape (C,W,Kmax)

    # -------- hazards λ_on, λ_off --------
    def qb_at(mask_ctx, ph_i, ph_all, rest_cw):
        return qb_density(ph_i, mask_ctx, ph_all, rest_cw)

    # on-rates: vectorization over all j, q_b is computed even on "on" slots and then dismissed with jnp.where(~m_cw[j],...)
    def lam_on_one(c, w):
        m_cw = m[c,w]
        ph_cw = phi[c,w]
        rest_cw = None if rest is None else rest[c,w]
        def on_for_slot(j):
            if qb_eval_variant == "child":
                m_child = m_cw.at[j].set(True)
                ctx = m_child
            else:
                ctx = m_cw
            val = qb_at(ctx, ph_cw[j], ph_cw, rest_cw)
            return jnp.where(~m_cw[j], beta_cw[c,0] * val, 0.0)
        return jax.vmap(on_for_slot)(arK)
    lam_on = jax.vmap(jax.vmap(lam_on_one, in_axes=(None,0)), in_axes=(0,None))(jnp.arange(C), jnp.arange(W))
    # (C,W,Kmax)

    # off-rates: vectorization over all j, q_b is computed even on "on" slots and then dismissed with jnp.where(m_cw[j],...)
    def lam_off_one(c, w):
        m_cw = m[c,w]
        ph_cw = phi[c,w]
        rest_cw = None if rest is None else rest[c,w]
        def off_for_slot(i):
            if qb_eval_variant == "child":  # evaluate at parent m
                ctx = m_cw
            else:                           # parent-variant uses child-less ctx m^{(-i)}
                ctx = m_cw.at[i].set(False)
            val = qb_at(ctx, ph_cw[i], ph_cw, rest_cw)
            ratio = jnp.exp(logpi_off[c,w,i] - logpi_cur[c,w])
            return jnp.where(m_cw[i], beta_cw[c,0] * val * ratio, 0.0)
        return jax.vmap(off_for_slot)(arK)
    lam_off = jax.vmap(jax.vmap(lam_off_one, in_axes=(None,0)), in_axes=(0,None))(jnp.arange(C), jnp.arange(W))
    # (C,W,Kmax)

    # total intensity per chain
    lam_total = lam_on.sum(-1) + lam_off.sum(-1)     # (C,W)

    # holding time and event picking
    # sampling using CDF of exponential distribution : dt = -log(u)/Lambda : TODO check it
    key, k_hold, k_pick, k_resamp = random.split(key, 4)      # <<< CHANGED (extra key for refresh)
    eps = jnp.finfo(phi.dtype).tiny  # avoid log(0) / divide-by-zero
    u = random.uniform(k_hold, shape=(C, W), minval=eps, maxval=1.0)
    dt_raw = -jnp.log(u) / jnp.maximum(lam_total, eps)
    dt = jnp.where(lam_total > eps, dt_raw, jnp.inf)

    hazards = jnp.concatenate([lam_on, lam_off], axis=-1)  # (C,W,2Kmax) first half is on, second half is off
    # IMPORTANT: never choose zero-hazard events
    #sample this categorical efficiently with the Gumbel–Max trick
    logits = jnp.where(hazards > 0, jnp.log(hazards), -jnp.inf)   # <<< CHANGED
    g = random.gumbel(k_pick, logits.shape) # i.i.d. Gumbel-Max trick: check implementation
    #Gumbel: arg max_k(log w_k + G_k) \propto Categorical (w_k/\sum w_j), G_k is g above.
    idx_flat = jnp.argmax(logits + g, axis=-1)             # (C,W)
    chosen_is_on  = idx_flat < Kmax # is it in the first half?  -> "on" hasards
    idx_slot = jnp.where(chosen_is_on, idx_flat, idx_flat - Kmax).astype(jnp.int32) # (C, W) mapping back to Kmax slots

    no_event = lam_total <= eps
    kind = jnp.where(no_event, jnp.int8(2), jnp.where(chosen_is_on, jnp.int8(0), jnp.int8(1)))
    #•	2 = no event,
	#•	0 = birth (turn on),
	#•	1 = death (turn off).

    # toggle mask
    m_new = m # (C,W,Kmax)
    m_new = jnp.where(
        no_event[..., None], # (C,W,1)
        m_new,
        jnp.where(
            chosen_is_on[..., None], # (C,W,1)
            m_new.at[jnp.arange(C)[:, None], jnp.arange(W)[None, :], idx_slot].set(True),
            m_new.at[jnp.arange(C)[:, None], jnp.arange(W)[None, :], idx_slot].set(False),
        ),
    )

    # --------- REFRESH ON DEACTIVATION (ψ) ---------
    death_mask = (~chosen_is_on) & (~no_event)                                  # (C,W)   <<< ADDED
    # draw ψ-samples for every (C,W); they'll be used only where death_mask=True
    keys_rs = random.split(k_resamp, C * W).reshape(C, W)           # <<< ADDED
    psi_samples = jax.vmap(jax.vmap(sample_pseudo_phi))(keys_rs)                # (C,W,d) <<< ADDED

    # update phi at the death index with ψ-sample (vectorized over C,W)
    def _upd_phi(phi_cw: Array, idx: Array, do: Array, s: Array) -> Array:      # <<< ADDED
        # phi_cw: (Kmax,d), idx: (), do: bool, s: (d,)
        def setit(_):
            return phi_cw.at[idx].set(s)
        return jax.lax.cond(do, setit, lambda _: phi_cw, operand=None)

    phi_new = jax.vmap(jax.vmap(_upd_phi, in_axes=(0, 0, 0, 0)))(
        phi, idx_slot, death_mask, psi_samples
    )                                                                            # <<< ADDED

    # recompute tempered logpi for new (m_new, phi_new)
    logpi_new = _logpi_tempered(
        phi_new, m_new, rest, betas,
        log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax
    )

    k_before = m.sum(axis=-1).astype(jnp.int32)
    k_after  = m_new.sum(axis=-1).astype(jnp.int32)

    new_state = PSState(phi=phi_new, rest=rest, m=m_new, logpi=logpi_new)       # <<< CHANGED
    ev = BDEvent(dt=dt, kind=kind, index=idx_slot, k_before=k_before, k_after=k_after)
    return new_state, ev


def bd_hazards_only(
    state: PSState,
    *,
    betas: Array,                           # (C,)
    Kmax: int,
    qb_density: Callable[[Array, Array, Array, Array|None], Array],
    qb_eval_variant: Literal["child","parent"] = "child",
    log_prior_phi: Callable[[Array], Array],
    log_pseudo_phi: Callable[[Array], Array],
    log_lik_masked: Callable[[Array, Array, Array|None], Array],
    log_p_k: Callable[[Array], Array],
) -> Tuple[Array, Array, Array, Array]:
    """
    Returns (lam_on, lam_off, lam_total, logpi_off)
      lam_on:  (C,W,Kmax)
      lam_off: (C,W,Kmax)
      lam_total: (C,W)
      logpi_off: (C,W,Kmax)  (as in your step; can be reused by bd_fire_one)
    """

    phi, m, rest, logpi_cur = state.phi, state.m, state.rest, state.logpi
    C, W, Kmax_local, d = phi.shape
    assert Kmax_local == Kmax
    beta_cw = betas[:, None]
    CW = C * W

    # -------- precompute logpi(m^{(-i)}) for all active i --------
    # build masks with slot i turned off
    arK = jnp.arange(Kmax)

    def turn_off_mask(m_cw, i):
        return m_cw.at[i].set(False)

    # compute off-state for each \phi_i (even if it is in off-state): vectorisation of a fixed size -- simplicity
    def logpi_off_all(phi_cw, m_cw, rest_cw, beta):
        # compute logpi for each i turned off
        def one_i(i):
            m_off = turn_off_mask(m_cw, i)
            return _logpi_tempered(phi_cw[None,None,:,:], m_off[None,None,:],
                                   rest_cw[None,None,...] if rest_cw is not None else None,
                                   jnp.array([beta])[None,...],
                                   log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax
                                   )[0,0]
        return jax.vmap(one_i)(arK)  # (Kmax,)
    logpi_off = jax.vmap(jax.vmap(logpi_off_all, in_axes=(0,0,0, None)),
                         in_axes=(0,0,0,0))(phi, m, rest if rest is not None else jnp.zeros((C,W,0)), betas)
    # shape (C,W,Kmax)

    # -------- hazards λ_on, λ_off --------
    def qb_at(mask_ctx, ph_i, ph_all, rest_cw):
        return qb_density(ph_i, mask_ctx, ph_all, rest_cw)

    # on-rates: vectorization over all j, q_b is computed even on "on" slots and then dismissed with jnp.where(~m_cw[j],...)
    def lam_on_one(c, w):
        m_cw = m[c,w]
        ph_cw = phi[c,w]
        rest_cw = None if rest is None else rest[c,w]
        def on_for_slot(j):
            if qb_eval_variant == "child":
                m_child = m_cw.at[j].set(True)
                ctx = m_child
            else:
                ctx = m_cw
            val = qb_at(ctx, ph_cw[j], ph_cw, rest_cw)
            return jnp.where(~m_cw[j], beta_cw[c,0] * val, 0.0)
        return jax.vmap(on_for_slot)(arK)
    lam_on = jax.vmap(jax.vmap(lam_on_one, in_axes=(None,0)), in_axes=(0,None))(jnp.arange(C), jnp.arange(W))
    # (C,W,Kmax)

    # off-rates: vectorization over all j, q_b is computed even on "on" slots and then dismissed with jnp.where(m_cw[j],...)
    def lam_off_one(c, w):
        m_cw = m[c,w]
        ph_cw = phi[c,w]
        rest_cw = None if rest is None else rest[c,w]
        def off_for_slot(i):
            if qb_eval_variant == "child":  # evaluate at parent m
                ctx = m_cw
            else:                           # parent-variant uses child-less ctx m^{(-i)}
                ctx = m_cw.at[i].set(False)
            val = qb_at(ctx, ph_cw[i], ph_cw, rest_cw)
            ratio = jnp.exp(logpi_off[c,w,i] - logpi_cur[c,w])
            return jnp.where(m_cw[i], beta_cw[c,0] * val * ratio, 0.0)
        return jax.vmap(off_for_slot)(arK)
    lam_off = jax.vmap(jax.vmap(lam_off_one, in_axes=(None,0)), in_axes=(0,None))(jnp.arange(C), jnp.arange(W))
    # (C,W,Kmax)

    # total intensity per chain
    lam_total = lam_on.sum(-1) + lam_off.sum(-1)     # (C,W)

    return lam_on, lam_off, lam_total, logpi_off






def onoff_one_step(
    key: random.PRNGKey,
    state: PSState,
    lam_on: Array, lam_off: Array, logpi_off: Array,
    *,
    betas: Array, Kmax: int,
    qb_eval_variant: Literal["child","parent"] = "child",
    sample_pseudo_phi: Callable[[random.PRNGKey], Array],
    log_prior_phi: Callable[[Array], Array],
    log_pseudo_phi: Callable[[Array], Array],
    log_lik_masked: Callable[[Array, Array, Array|None], Array],
    log_p_k: Callable[[Array], Array],
) -> Tuple[PSState, BDEvent]:
    


    C, W, Kmax_local, d = state.phi.shape
    assert Kmax_local == Kmax

    key, k_hold, k_pick, k_resamp = random.split(key, 4)      # <<< CHANGED (extra key for refresh)

    hazards = jnp.concatenate([lam_on, lam_off], axis=-1)  # (C,W,2Kmax) first half is on, second half is off
    # IMPORTANT: never choose zero-hazard events
    #sample this categorical efficiently with the Gumbel–Max trick
    logits = jnp.where(hazards > 0, jnp.log(hazards), -jnp.inf)   # <<< CHANGED
    g = random.gumbel(k_pick, logits.shape) # i.i.d. Gumbel-Max trick: check implementation
    #Gumbel: arg max_k(log w_k + G_k) \propto Categorical (w_k/\sum w_j), G_k is g above.
    idx_flat = jnp.argmax(logits + g, axis=-1)             # (C,W)
    chosen_is_on  = idx_flat < Kmax # is it in the first half?  -> "on" hasards
    idx_slot = jnp.where(chosen_is_on, idx_flat, idx_flat - Kmax).astype(jnp.int32) # (C, W) mapping back to Kmax slots

    lam_total = hazards.sum(-1)
    eps = jnp.finfo(phi.dtype).tiny  # avoid log(0) / divide-by-zero
    no_event = lam_total <= eps

    ### Compute dt of BD process
    u = random.uniform(k_hold, shape=(C, W), minval=eps, maxval=1.0)
    dt_raw = -jnp.log(u) / jnp.maximum(lam_total, eps)
    dt = jnp.where(lam_total > eps, dt_raw, jnp.inf)

    # toggle mask safely with lax.cond
    def _toggle_mask(m_cw, idx, do, turn_on):
        def _apply(_):
            return jax.lax.select(turn_on, m_cw.at[idx].set(True), m_cw.at[idx].set(False))
        return jax.lax.cond(do, _apply, lambda _: m_cw, operand=None)

    m_new = jax.vmap(jax.vmap(_toggle_mask))(state.m, idx_slot, ~no_event, chosen_is_on)

    # refresh ψ on death only
    death_mask = (~chosen_is_on) & (~no_event)                                  # (C,W)   <<< ADDED
    # draw ψ-samples for every (C,W); they'll be used only where death_mask=True
    keys_rs = random.split(k_resamp, C * W).reshape(C, W)           # <<< ADDED
    psi_samples = jax.vmap(jax.vmap(sample_pseudo_phi))(keys_rs)                # (C,W,d) <<< ADDED

    # update phi at the death index with ψ-sample (vectorized over C,W)
    def _upd_phi(phi_cw: Array, idx: Array, do: Array, s: Array) -> Array:      # <<< ADDED
        # phi_cw: (Kmax,d), idx: (), do: bool, s: (d,)
        def setit(_):
            return phi_cw.at[idx].set(s)
        return jax.lax.cond(do, setit, lambda _: phi_cw, operand=None)

    phi_new = jax.vmap(jax.vmap(_upd_phi, in_axes=(0, 0, 0, 0)))(
        phi, idx_slot, death_mask, psi_samples
    )

    # recompute tempered logpi
    logpi_new = _logpi_tempered(
        phi_new, m_new, state.rest, betas,
        log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax
    )

    kind = jnp.where(no_event, jnp.int8(2), jnp.where(chosen_is_on, jnp.int8(0), jnp.int8(1)))
    #•	2 = no event,
	#•	0 = birth (turn on),
	#•	1 = death (turn off).
    k_before = m.sum(axis=-1).astype(jnp.int32)
    k_after  = m_new.sum(axis=-1).astype(jnp.int32)

    ev = BDEvent(
        dt = dt,
        kind = kind,
        index = idx_slot,
        k_before = k_before,
        k_after  = k_after,
    )
    new_state = PSState(phi=phi_new, rest=state.rest, m=m_new, logpi=logpi_new)
    return new_state, ev



### CHANGED: add functional trace append helpers
def _trace_append(trace, state):
    idx = trace.current_index
    thetas = trace.thetas.at[idx].set(state.thetas)
    logps  = trace.log_probs.at[idx].set(state.log_probs)
    return PTTrace(thetas, logps, trace.max_length, idx + 1)

def _trace_maybe_append(trace, state, accepted_mask):
    # append only if anyone accepted in this substep
    return jax.lax.cond(
        accepted_mask.any(),
        lambda tr: _trace_append(tr, state),
        lambda tr: tr,
        trace
    )



# A small utility to "restrict" proposals to coordinates of slot j
def _restrict_slot(prop_all, cur_all, slot_slice_or_mask):
    if isinstance(slot_slice_or_mask, slice):
        sl = slot_slice_or_mask
        return cur_all.at[..., sl].set(prop_all[..., sl])
    else:
        maskD = slot_slice_or_mask.astype(jnp.bool_)[None, None, :]
        return jnp.where(maskD, prop_all, cur_all)

def gibbs_mh_single_sweep_over_active(
    key,
    pt_state: PTState,                    # thetas: (C,W,D), log_probs: (C,W)
    m_mask_cwk: jnp.ndarray,              # (C,W,Kmax) active slots from PSState
    slot_slices: Tuple,                   # length Kmax; each is slice or bool mask over D selecting φ_j
    betas: jnp.ndarray,                   # (C,)
    *,
    # your three proposals (unchanged APIs)
    do_stretch: bool,
    do_rw: bool,
    do_de: bool,
    # proposal params you already use
    stretch_keys,                         # tuple of PRNG keys needed by stretch
    stretch_a: float,
    Ls: jnp.ndarray, evecs: jnp.ndarray, evals_pd: jnp.ndarray,    # for RW kinds
    scales: dict,                         # {"small","line","big"}
    de_keys,                              # (key_partner, key_gamma)
    cross_rate: float, 
    gamma_de: float,
    # likelihood
    log_prob_fn_single: Callable[[jnp.ndarray], jnp.ndarray], ### shouldn't it be log_lik_masked?
    lik_chunk: int,
):
    C, W, D = pt_state.thetas.shape
    betas = betas  # (C,)

    def mh_apply(rng_key, cur_state: PTState, prop, log_qcorr, move_mask):
        C, W, D = cur_state.thetas.shape
        flat = prop.reshape((C*W, D))
        prop_lp = batched_log_prob(log_prob_fn_single, flat, chunk=lik_chunk).reshape((C, W))
        acc, _ = mh_accept_masked(rng_key, cur_state.log_probs, prop_lp, betas, log_qcorr, move_mask)
        th_new = jnp.where(acc[:, :, None], prop, cur_state.thetas)
        lp_new = jnp.where(acc,           prop_lp, cur_state.log_probs)
        return PTState(th_new, lp_new), acc


    # iterate over all slots j (Gibbs–MH style)
    def do_one_slot(carry, j):
        key_s, st, trace = carry
        move_mask = m_mask_cwk[:, :, j]           # only walkers with slot j active
        sl = slot_slices[j] 

        # Sequential: STRETCH -> RW -> DE, but only if flags are True
        th_base = st.thetas

        # 1) Stretch (restricted to active walkers)
        if do_stretch:
            k_mask, k_p1, k_z1, k_p2, k_z2 = stretch_keys
            red, blue = redblue_mask(k_mask, C, W)
            red = red & move_mask
            blue = blue & move_mask
            prop1, logJ1, _, has1, _ = _propose_stretch_redblue(k_p1, k_z1, th_base, subset_mask=red, a=stretch_a, z=None)
            prop1 = _restrict_slot(prop1, th_base, sl)
            key_s, k_acc1 = random.split(key_s)
            st, acc1 = mh_apply(k_acc1, st, prop1, logJ1, red & has1)
            trace = _trace_maybe_append(trace, st, acc1)

            prop2, logJ2, _, has2, _ = _propose_stretch_redblue(k_p2, k_z2, st.thetas, subset_mask=blue, a=stretch_a, z=None)
            prop2 = _restrict_slot(prop2, st.thetas, sl)
            key_s, k_acc2 = random.split(key_s)
            st, acc2 = mh_apply(k_acc2, st, prop2, logJ2, blue & has2)
            trace = _trace_maybe_append(trace, st, acc2)

        # 2) RW (fullcov/eigenline/student_t), but keep only slot coords
        if do_rw:
            # fullcov as an example (you can switch like before)
            prop_rw = _propose_fullcov(random.split(key_s,2)[0], st.thetas, Ls, scale=scales["small"])
            prop_rw = _restrict_slot(prop_rw, st.thetas, sl)
            key_s, k_acc3 = random.split(key_s)
            st, acc3 = mh_apply(k_acc3, st, prop_rw, jnp.zeros_like(st.log_probs), move_mask)
            trace = _trace_maybe_append(trace, st, acc3)

        # 3) DE (partners only among move_mask implicitly—proposal gives full ensemble; we mask on apply)
        if do_de:
            kp, kg = de_keys
            prop_de, idx_y, idx_z, has_pair, xmask = _propose_de_two_point(kp, kg, st.thetas,
                                gamma=None, gamma_scale=gamma_de, crossover_rate=cross_rate, jitter_scale=1e-6,
                                eligible_mask=move_mask)
            prop_de = _restrict_slot(prop_de, st.thetas, sl)
            key_s, k_acc4 = random.split(key_s)
            st, acc4 = mh_apply(k_acc4, st, prop_de, jnp.zeros_like(st.log_probs), move_mask & has_pair)
            trace = _trace_maybe_append(trace, st, acc4)

        return (key_s, trace, st)

    (key_out, trace_out, pt_trace_out) = jax.lax.scan(do_one_slot, (key, pt_state, pt_trace), jnp.arange(len(slot_slices)))
    return key_out, trace_out, pt_trace_out



### I want each chain to run its own process (different rates of BD & MH) but aI want pseudo-synch at the PT swap




# ### process runner (integrator)
# def run_epoch_ct(
#     key: jax.random.PRNGKey,
#     pt_init: PTState,                 # (C,W,D), (C,W)
#     ps_init: PSState,                 # phi:(C,W,Kmax,d), m:(C,W,Kmax), logpi:(C,W)
#     *,
#     T_end: float,
#     rho_mh: float,                    # Poisson rate ρ for in-model MH
#     MH_BATCH: int = 5,
#     # product-space / likelihood bits
#     betas: jnp.ndarray,               # (C,)
#     Kmax: int,
#     qb_density,
#     qb_eval_variant: Literal["child","parent"] = "child",
#     log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k,
#     sample_pseudo_phi,
#     # mapping φ_j -> coordinates in θ (length Kmax)
#     slot_slices: Tuple,               # each is slice or bool mask over D
#     # proposals setup (unchanged)
#     stretch_a: float, scales: dict, evecs, evals_pd, Ls,
#     cross_rate: float, gamma_de: float,
#     log_prob_fn_single: Callable[[jnp.ndarray], jnp.ndarray],
#     lik_chunk: int = 64,
#     max_steps: int = 10_000,          # safety cap
#     ### CHANGED: PTTrace buffer capacity
#     pttrace_capacity: int = 1_000_000,
# ):
#     """
#     Returns:
#       pt_state_final, ps_state_final,
#       trace_thetas: (S+1, C, W, D)
#       trace_masks:  (S+1, C, W, Kmax)
#       trace_dt:     (S,)   holding time per global step (same Δt applied to all chains)
#       trace_kind:   (S,)   0=BD(on), 1=BD(off), 2=no-BD (MH tick)
#     """
#     key, k_bd, k_mh = random.split(key, 3)
#     C, W, D = pt_init.thetas.shape

#     # allocate logs
#     # thetas_log = jnp.zeros((max_steps+1, C, W, D), dtype=pt_init.thetas.dtype).at[0].set(pt_init.thetas)
#     masks_log  = jnp.zeros((max_steps+1, C, W, Kmax), dtype=ps_init.m.dtype).at[0].set(ps_init.m)
#     dt_bd_log  = jnp.zeros((max_steps,), dtype=jnp.float32)
#     kind_log   = jnp.full((max_steps,), 2, dtype=jnp.int8)  # default "MH tick"
#     t = jnp.array(0.0, dtype=jnp.float32)
#     step = 0

#     pt_state = pt_init
#     ps_state = ps_init

#     def cond_fun(carry):
#         step, t, *_ = carry
#         return jnp.logical_and(step < max_steps, t < T_end)

#     def body_fun(carry):
#         step, t, key, pt_state, ps_state, thetas_log, masks_log, dt_log, kind_log = carry
#         key, k1, k2, k3, k4 = random.split(key, 5)

#         # 1) BD hazards (no dt)
#         lam_on, lam_off, lam_total, logpi_off = bd_hazards_only(
#             ps_state, betas=betas, Kmax=Kmax, qb_density=qb_density,
#             qb_eval_variant=qb_eval_variant,
#             log_prior_phi=log_prior_phi, log_pseudo_phi=log_pseudo_phi,
#             log_lik_masked=log_lik_masked, log_p_k=log_p_k,
#         )
#         eps = jnp.finfo(pt_state.thetas.dtype).tiny
#         Lam = lam_total.sum()  # global competition clock (simple choice)
#         # 2) Two exponential clocks
#         u1 = random.uniform(k1, (), minval=eps, maxval=1.0)
#         u2 = random.uniform(k2, (), minval=eps, maxval=1.0)
#         tau_bd = jnp.where(Lam > eps, -jnp.log(u1) / Lam, jnp.inf)
#         tau_mh = -jnp.log(u2) / jnp.maximum(rho_mh, eps)


#         do_mh = tau_mh < tau_bd
#         dt    = jnp.minimum(tau_mh, tau_bd)
#         t_new = t + dt

#         def do_mh_tick(args):
#             key_i, pt_i, ps_i = args
#             # Do MH_BATCH Gibbs sweeps over active slots
#             def one_batch(carry_b, _):
#                 def one_batch(carry_b, _):
#                 key_b, pt_b, trace_b = carry_b
#                 # flags: enable all three in sequence, like your run_epoch
#                 k_mask, k_p1, k_z1, k_p2, k_z2 = random.split(key_b, 5)
#                 kp, kg = random.split(k_z2, 2)
#                 key_out, pt_next, trace_next = gibbs_mh_single_sweep_over_active(
#                     key_b, pt_b, ps_i.m, slot_slices, betas,
#                     do_stretch=True, do_rw=True, do_de=True,
#                     stretch_keys=(k_mask, k_p1, k_z1, k_p2, k_z2),
#                     stretch_a=stretch_a,
#                     Ls=Ls, evecs=evecs, evals_pd=evals_pd, scales=scales,
#                     de_keys=(kp, kg),
#                     cross_rate=cross_rate, gamma_de=gamma_de,
#                     log_prob_fn_single=log_prob_fn_single, lik_chunk=lik_chunk,
#                     pt_trace=trace_b,                                    # ### CHANGED
#                 )
#                 return (key_out, pt_next, trace_next)

#             (key_out, pt_out, pt_trace_out) = jax.lax.scan(one_batch, (key_i, pt_i), jnp.arange(MH_BATCH))
#             return key_out, pt_out, ps_i, pt_trace_out, jnp.int8(2)  # kind=2 (MH tick)

#         def do_bd_tick(args):
#             key_i, pt_i, ps_i = args
#             # Pick and fire one BD event using hazards (no dt redraw)
#             key_f, key_rs = random.split(key_i)
#             ps_next, fired = onoff_one_step(
#                 key_f, ps_i, lam_on, lam_off, logpi_off,
#                 betas=betas, Kmax=Kmax, qb_eval_variant=qb_eval_variant,
#                 sample_pseudo_phi=sample_pseudo_phi,
#                 log_prior_phi=log_prior_phi, log_pseudo_phi=log_pseudo_phi,
#                 log_lik_masked=log_lik_masked, log_p_k=log_p_k,
#             )
#             return key_rs, pt_i, ps_next, fired.kind.max()  # 0/1 across (C,W)

#         key_new, pt_new, ps_new, kind_evt = jax.lax.cond(
#             do_mh, do_mh_tick, do_bd_tick, (k3, pt_state, ps_state)
#         )

#         # log
#         thetas_log = thetas_log.at[step+1].set(pt_new.thetas)
#         masks_log  = masks_log.at[step+1].set(ps_new.m)
#         dt_log     = dt_log.at[step].set(dt)
#         kind_log   = kind_log.at[step].set(kind_evt)

#         return (step+1, t_new, key_new, pt_new, ps_new, thetas_log, masks_log, dt_log, kind_log)

#     (step_f, t_f, key_f, pt_f, ps_f, thetas_log, masks_log, dt_log, kind_log) = jax.lax.while_loop(
#         cond_fun, body_fun,
#         (step, t, key, pt_state, ps_state, thetas_log, masks_log, dt_log, kind_log)
#     )

#     # trim logs to actual length
#     thetas_log = thetas_log[:step_f+1]
#     masks_log  = masks_log[:step_f+1]
#     dt_log     = dt_log[:step_f]
#     kind_log   = kind_log[:step_f]

#     return pt_f, ps_f, thetas_log, masks_log, dt_log, kind_log



# # Shapes
# C, W, Kmax, d = 8, 16, 4, Npar_src
# betas = 1.0 / temperatures  # (C,)

# # Initial state: start with some mask m0, all phi ~ ψ, cache logpi
# phi0  = jax.random.uniform(random.PRNGKey(0), (C, W, Kmax, d))
# m0    = jnp.zeros((C, W, Kmax), dtype=bool)  # start empty, say
# rest0 = None

# logpi0 = _logpi_tempered(phi0, m0, rest0, betas,
#                          log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax)

# ps = PSState(phi=phi0, rest=rest0, m=m0, logpi=logpi0)

### possible proposals:# Uniform on [0,1]^d (ψ) used as q_b
# def qb_density_uniform(phi_i, m_ctx, phi_all, rest):
#     inside = jnp.logical_and(jnp.all(phi_i >= 0), jnp.all(phi_i <= 1))
#     return jnp.where(inside, 1.0, 0.0)

# # Residual-guided mixture: q_b ∝ (1-α) ψ + α * exp(Δℓ(φ_i; context)); normalized
# def qb_density_residual(phi_i, m_ctx, phi_all, rest):
#     log_psi = 0.0  # if ψ is uniform on unit box
#     log_bump = residual_gain_loglik(phi_i, m_ctx, phi_all, rest)  # your Δℓ
#     alpha = 0.3
#     # return a *normalized* density value; here we assume bump is already normalized
#     # If not, include its normalizing constant.
#     return (1 - alpha) * jnp.exp(log_psi) + alpha * jnp.exp(log_bump)


# # One BD step:
# key = random.PRNGKey(123)
# ps, ev = bd_onoff_ct_step(
#     key, ps,
#     betas=betas, Kmax=Kmax,
#     qb_density=qb_density,                 # normalized
#     qb_eval_variant="child",               # or "parent"
#     log_prior_phi=log_prior_phi,
#     log_pseudo_phi=log_pseudo_phi,
#     log_lik_masked=log_lik_masked,
#     log_p_k=log_p_k,
# )