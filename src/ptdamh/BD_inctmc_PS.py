# src/ptdamh/product_space_ctmc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.scipy.special import gammaln

Array = jnp.ndarray

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
    comp = jax.vmap(_one)(jnp.arange(Kmax)).sum()

    ll = log_lik_masked(phi_kwd, m_k, rest_d)  # uses only active slots

    return ( log_p_k(k)
           + _log_uniform_masks_given_k(Kmax, k)
           + _log_symmetrization(k)
           + comp + ll )

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
    sample_pseudo_phi: Callable[[random.PRNGKey], Array],     # <<< ADDED
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

    # on-rates
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

    # off-rates
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
    # sampling using CDF of exponential distribution --> check if correct.
    key, k_hold, k_pick, k_resamp = random.split(key, 4)      # <<< CHANGED (extra key for refresh)
    eps = jnp.finfo(phi.dtype).tiny  # avoid log(0) / divide-by-zero
    u = random.uniform(k_hold, shape=(C, W), minval=eps, maxval=1.0)
    dt = -jnp.log(u) / jnp.maximum(lam_total, eps)

    hazards = jnp.concatenate([lam_on, lam_off], axis=-1)  # (C,W,2Kmax)
    # IMPORTANT: never choose zero-hazard events
    #sample this categorical efficiently with the Gumbel–Max trick
    logits = jnp.where(hazards > 0, jnp.log(hazards), -jnp.inf)   # <<< CHANGED
    g = random.gumbel(k_pick, logits.shape) # i.i.d. Gumbel-Max trick: check implementation
    idx_flat = jnp.argmax(logits + g, axis=-1)             # (C,W)
    chosen_is_on  = idx_flat < Kmax
    idx_slot = jnp.where(chosen_is_on, idx_flat, idx_flat - Kmax).astype(jnp.int32)

    no_event = lam_total <= eps
    kind = jnp.where(no_event, jnp.int8(2), jnp.where(chosen_is_on, jnp.int8(0), jnp.int8(1)))

    # toggle mask
    m_new = m
    m_new = jnp.where(
        no_event[..., None],
        m_new,
        jnp.where(
            chosen_is_on[..., None],
            m_new.at[jnp.arange(C)[:, None], jnp.arange(W)[None, :], idx_slot].set(True),
            m_new.at[jnp.arange(C)[:, None], jnp.arange(W)[None, :], idx_slot].set(False),
        ),
    )

    # --------- REFRESH ON DEACTIVATION (ψ) ---------
    death_mask = (~chosen_is_on) & (~no_event)                                  # (C,W)   <<< ADDED
    # draw ψ-samples for every (C,W); they'll be used only where death_mask=True
    keys_rs = random.split(k_resamp, C * W).reshape(C, W, 2)[:, :, 0]           # <<< ADDED
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