# src/ptdamh/BD_MH_step.py
### Need debugging!!!!

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Dict, List

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.scipy.special import gammaln

Array = jnp.ndarray

# ---- External proposals you already have ----
from .proposals import (
    redblue_mask,
    _propose_stretch_redblue,
    _propose_fullcov,
    _propose_eigenline,       # available if you want to wire later
    _propose_student_t,       # available if you want to wire later
    _propose_de_two_point,
)

# ==============================
#         Core States
# ==============================

@dataclass
class PTState:
    thetas: Array     # (C, W, D)
    log_probs: Array  # (C, W)

@dataclass
class PSState:
    phi: Array        # (C, W, Kmax, d)
    rest: Array | None
    m: Array          # (C, W, Kmax) bool
    logpi: Array      # (C, W)


# ==============================
#            Traces
# ==============================

@dataclass
class PTTrace:
    """
    Record the ensemble state AFTER EVERY MH SUBMOVE (attempt).
    - thetas/log_probs: snapshot after the submove (accepted or rejected)
    - accepted: (C,W) mask for that submove
    - prop_id: int8 code for proposal kind (see PROPOSAL_IDS)
    - slot_j : which Gibbs slot was updated (-1 if N/A)
    """
    thetas: Array        # (N, C, W, D)
    log_probs: Array     # (N, C, W)
    accepted: Array      # (N, C, W) bool
    prop_id: Array       # (N,) int8
    slot_j: Array        # (N,) int32
    max_length: int
    current_index: int

    @staticmethod
    def init(max_length: int, C: int, W: int, D: int, dtype=jnp.float32) -> "PTTrace":
        return PTTrace(
            thetas=jnp.zeros((max_length, C, W, D), dtype=dtype),
            log_probs=jnp.zeros((max_length, C, W), dtype=dtype),
            accepted=jnp.zeros((max_length, C, W), dtype=jnp.bool_),
            prop_id=jnp.zeros((max_length,), dtype=jnp.int8),
            slot_j=jnp.full((max_length,), -1, dtype=jnp.int32),
            max_length=max_length,
            current_index=0,
        )

    def append_always(self, st: PTState, accepted_mask: Array, prop_id: int, slot_j: int) -> "PTTrace":
        i = jnp.minimum(self.current_index, self.max_length - 1)
        th  = self.thetas.at[i].set(st.thetas)
        lp  = self.log_probs.at[i].set(st.log_probs)
        acc = self.accepted.at[i].set(accepted_mask)
        pid = self.prop_id.at[i].set(jnp.int8(prop_id))
        sj  = self.slot_j.at[i].set(jnp.int32(slot_j))
        return PTTrace(th, lp, acc, pid, sj, self.max_length, jnp.minimum(self.current_index + 1, self.max_length))

    def truncate(self) -> "PTTrace":
        i = self.current_index
        return PTTrace(self.thetas[:i], self.log_probs[:i], self.accepted[:i], self.prop_id[:i], self.slot_j[:i], int(i), int(i))


@dataclass
class PSTrace:
    """
    Record PS snapshots AFTER EVERY BD EVENT (always accepted).
    """
    phi: Array       # (N, C, W, Kmax, d)
    m: Array         # (N, C, W, Kmax)
    logpi: Array     # (N, C, W)
    max_length: int
    current_index: int

    @staticmethod
    def init(max_length: int, C: int, W: int, Kmax: int, d: int, dtype=jnp.float32, mdtype=jnp.bool_) -> "PSTrace":
        return PSTrace(
            phi=jnp.zeros((max_length, C, W, Kmax, d), dtype=dtype),
            m=jnp.zeros((max_length, C, W, Kmax), dtype=mdtype),
            logpi=jnp.zeros((max_length, C, W), dtype=dtype),
            max_length=max_length,
            current_index=0,
        )

    def append(self, st: PSState) -> "PSTrace":
        i = jnp.minimum(self.current_index, self.max_length - 1)
        phi = self.phi.at[i].set(st.phi)
        m = self.m.at[i].set(st.m)
        lp = self.logpi.at[i].set(st.logpi)
        return PSTrace(phi, m, lp, self.max_length, jnp.minimum(self.current_index + 1, self.max_length))

    def truncate(self) -> "PSTrace":
        i = self.current_index
        return PSTrace(self.phi[:i], self.m[:i], self.logpi[:i], int(i), int(i))


@dataclass
class EventLog:
    """
    Per-event record (BD + MH ticks)
      kind: 0=BD_birth, 1=BD_death, 2=MH_tick
      dt:   holding time for this event (τ_bd for BD, τ_mh for MH)
      c,w,j: which chain/slot (-1,-1,-1 for MH ticks)
    """
    kind: Array       # (E,) int8
    dt: Array         # (E,) float32
    c: Array          # (E,) int32
    w: Array          # (E,) int32
    j: Array          # (E,) int32
    max_length: int
    current_index: int

    @staticmethod
    def init(max_length: int) -> "EventLog":
        return EventLog(
            kind=jnp.zeros((max_length,), dtype=jnp.int8),
            dt=jnp.zeros((max_length,), dtype=jnp.float32),
            c=jnp.full((max_length,), -1, dtype=jnp.int32),
            w=jnp.full((max_length,), -1, dtype=jnp.int32),
            j=jnp.full((max_length,), -1, dtype=jnp.int32),
            max_length=max_length,
            current_index=0,
        )

    def append(self, kind: int, dt: float, c: int, w: int, j: int) -> "EventLog":
        i = jnp.minimum(self.current_index, self.max_length - 1)
        k = self.kind.at[i].set(jnp.int8(kind))
        d = self.dt.at[i].set(jnp.float32(dt))
        cc = self.c.at[i].set(jnp.int32(c))
        ww = self.w.at[i].set(jnp.int32(w))
        jj = self.j.at[i].set(jnp.int32(j))
        return EventLog(k, d, cc, ww, jj, self.max_length, jnp.minimum(self.current_index + 1, self.max_length))

    def truncate(self) -> "EventLog":
        i = self.current_index
        return EventLog(self.kind[:i], self.dt[:i], self.c[:i], self.w[:i], self.j[:i], int(i), int(i))


### Defines which temperatures I want to trace (save) and how often
@dataclass
class TraceConfig:
    # temperatures to record; if None -> default to argmax(beta) (cold chain)
    chain_inds: Optional[List[int]] = None
    # record after every N MH sweeps (NOT submoves) — you already log submoves in PTTrace
    record_every: int = 1
    # cap sizes for ring buffers
    max_pt_events: int = 10000
    max_ps_events: int = 10000
    max_events: int   = 10000

def _resolve_chain_inds(betas: np.ndarray, cfg: TraceConfig) -> List[int]:
    if cfg.chain_inds is not None and len(cfg.chain_inds) > 0:
        return list(cfg.chain_inds)
    return [int(np.argmax(betas))]


## bundle all tracers together
@dataclass
class TraceConfig:
    # temperatures to record; if None -> default to argmax(beta) (cold chain)
    chain_inds: Optional[List[int]] = None
    # record after every N MH sweeps (NOT submoves) — you already log submoves in PTTrace
    record_every: int = 1
    # cap sizes for ring buffers
    max_pt_events: int = 10000
    max_ps_events: int = 10000
    max_events: int   = 10000

def _resolve_chain_inds(betas: np.ndarray, cfg: TraceConfig) -> List[int]:
    if cfg.chain_inds is not None and len(cfg.chain_inds) > 0:
        return list(cfg.chain_inds)
    return [int(np.argmax(betas))]


class TraceManager:
    def __init__(self, *, C:int, W:int, D:int, Kmax:int, d:int,
                 betas: np.ndarray, cfg: TraceConfig,
                 dtype=jnp.float32, mdtype=jnp.bool_):
        self.cfg = cfg
        self.chain_inds = _resolve_chain_inds(betas, cfg)

        # full-state buffers (per-event)
        self.pt = PTTrace.init(cfg.max_pt_events, C, W, D, dtype=dtype)
        self.ps = PSTrace.init(cfg.max_ps_events, C, W, Kmax, d, dtype=dtype, mdtype=mdtype)
        self.ev = EventLog.init(cfg.max_events)

        # optional per-sweep snapshots of selected chains
        self._snap_theta = []
        self._snap_phi   = []
        self._snap_m     = []
        self._snap_ll    = []
        self._snap_t     = []

        self._mh_tick = 0

    # You already log PT submoves via PTTrace.append_always(...) where you make the move.
    def record_pt_submove(self, st: PTState, accepted_mask, prop_id: int, slot_j: int):
        self.pt = self.pt.append_always(st, accepted_mask, prop_id, slot_j)

    def record_bd_event(self, ps_state: PSState, dt: float, c:int, w:int, j:int, kind:int):
        # PSTrace: snapshot after every BD (always accepted)
        self.ps = self.ps.append(ps_state)
        # EventLog: kind 0/1 for birth/death (you already do this)
        self.ev = self.ev.append(kind=kind, dt=dt, c=c, w=w, j=j)

    def record_mh_tick(self, dt: float):
        # EventLog: kind 2 for MH tick (no c,w,j)
        self.ev = self.ev.append(kind=2, dt=dt, c=-1, w=-1, j=-1)
        self._mh_tick += 1

    # Optional: store compact snapshots of selected temperature chains after a sweep
    def snapshot_selected_chains(self, pt_state: PTState, ps_state: PSState, t_abs: float):
        if (self._mh_tick % self.cfg.record_every) != 0:
            return
        ci = self.chain_inds
        self._snap_theta.append(pt_state.thetas[ci].copy())  # (Nc,W,D)
        self._snap_phi.append(  ps_state.phi[ci].copy())     # (Nc,W,Kmax,d)
        self._snap_m.append(    ps_state.m[ci].copy())       # (Nc,W,Kmax)
        self._snap_ll.append(   pt_state.log_probs[ci].copy())# (Nc,W)
        self._snap_t.append(float(t_abs))

    def export(self) -> Dict[str, np.ndarray]:
        # stack optional snapshots; keep None if empty
        def _stack(xs): return None if len(xs)==0 else np.stack(xs, axis=0)
        return {
            "snapshot_theta": _stack(self._snap_theta),   # (Trec,Nc,W,D)
            "snapshot_phi":   _stack(self._snap_phi),     # (Trec,Nc,W,Kmax,d)
            "snapshot_m":     _stack(self._snap_m),       # (Trec,Nc,W,Kmax)
            "snapshot_ll":    _stack(self._snap_ll),      # (Trec,Nc,W)
            "snapshot_t":     (None if len(self._snap_t)==0 else np.asarray(self._snap_t)),
            "chain_inds":     np.asarray(self.chain_inds, dtype=np.int32),
        }

    def bundle(self) -> Traces:
        return Traces(self.pt, self.ps, self.ev, self.chain_inds)

    def finalize(self) -> Dict[str, np.ndarray]:
        tb = self.bundle().finalize()
        out = {
            "pt_thetas":     np.array(tb.pt.thetas),
            "pt_log_probs":  np.array(tb.pt.log_probs),
            "pt_accepted":   np.array(tb.pt.accepted),
            "pt_prop_id":    np.array(tb.pt.prop_id),
            "pt_slot_j":     np.array(tb.pt.slot_j),

            "ps_phi":        np.array(tb.ps.phi),
            "ps_m":          np.array(tb.ps.m),
            "ps_logpi":      np.array(tb.ps.logpi),

            "ev_kind":       np.array(tb.ev.kind),
            "ev_dt":         np.array(tb.ev.dt),
            "ev_c":          np.array(tb.ev.c),
            "ev_w":          np.array(tb.ev.w),
            "ev_j":          np.array(tb.ev.j),
            "chain_inds":    np.asarray(self.chain_inds, dtype=np.int32),
        }
        out.update(self.export())
        return out

# ==============================
#     Target & Likelihood
# ==============================

LogPriorPhi  = Callable[[Array], Array]           # (d,) -> ()
LogPseudoPhi = Callable[[Array], Array]           # (d,) -> ()
LogLikMasked = Callable[[Array, Array, Array | None], Array]  # (Kmax,d),(Kmax,),rest -> ()
LogPk        = Callable[[Array], Array]           # int -> ()
QbDensity    = Callable[[Array, Array, Array, Array | None], Array]

def _count_k(m: Array) -> Array:
    return m.astype(jnp.int32).sum(axis=-1)  # (C,W)

def _log_uniform_masks_given_k(Kmax: int, k: Array) -> Array:
    k = jnp.asarray(k, dtype=jnp.int32)
    return - (gammaln(Kmax + 1.) - gammaln(k + 1.) - gammaln(Kmax - k + 1.))

def _log_symmetrization(k: Array) -> Array:
    k = jnp.asarray(k, dtype=jnp.float32)
    return -gammaln(k + 1.0)

def _logpi_extended_untempered(
    phi_kwd: Array, m_k: Array, rest_d: Array | None,
    log_prior_phi: LogPriorPhi, log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked, log_p_k: LogPk, Kmax: int,
) -> Array:
    k = m_k.astype(jnp.int32).sum()
    def _one(i):
        logp = log_prior_phi(phi_kwd[i])
        logpsi = log_pseudo_phi(phi_kwd[i])
        return jnp.where(m_k[i], logp, logpsi)
    comp = jax.vmap(_one)(jnp.arange(Kmax)).sum()
    ll = log_lik_masked(phi_kwd, m_k, rest_d)
    return log_p_k(k) + _log_uniform_masks_given_k(Kmax, k) + _log_symmetrization(k) + comp + ll

def _logpi_tempered(
    phi: Array, m: Array, rest: Array | None, betas: Array,
    log_prior_phi: LogPriorPhi, log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked, log_p_k: LogPk, Kmax: int,
) -> Array:
    C, W, Kmax_local, d = phi.shape
    assert Kmax_local == Kmax
    def _per_chain(phi_cw, m_cw, rest_cw, beta):
        logpi_un = _logpi_extended_untempered(
            phi_cw, m_cw, rest_cw, log_prior_phi, log_pseudo_phi,
            log_lik_masked=lambda a,b,c: 0.0, log_p_k=log_p_k, Kmax=Kmax
        )
        ll_un = log_lik_masked(phi_cw, m_cw, rest_cw)
        return logpi_un + beta * ll_un
    f = jax.vmap(jax.vmap(_per_chain, in_axes=(0,0,0,None)), in_axes=(0,0,0,0))
    return f(phi, m, rest if rest is not None else jnp.zeros((C,W,0)), betas)


# ==============================
#         BD Hazards
# ==============================

def bd_hazards_only(
    state: PSState,
    *,
    betas: Array, Kmax: int,
    qb_density: QbDensity, qb_eval_variant: Literal["child","parent"],
    log_prior_phi: LogPriorPhi, log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked, log_p_k: LogPk,
) -> Tuple[Array, Array, Array]:
    """Return lam_on, lam_off, lam_total."""
    phi, m, rest, logpi_cur = state.phi, state.m, state.rest, state.logpi
    C, W, Kmax_local, d = phi.shape
    assert Kmax_local == Kmax
    beta_cw = betas[:, None]
    arK = jnp.arange(Kmax)

    def turn_off_mask(m_cw, i):
        return m_cw.at[i].set(False)

    # logπ(m^{(-i)}) for current state
    def logpi_off_all(phi_cw, m_cw, rest_cw, beta):
        def one_i(i):
            m_off = turn_off_mask(m_cw, i)
            return _logpi_tempered(
                phi_cw[None,None,:,:], m_off[None,None,:],
                rest_cw[None,None,...] if rest_cw is not None else None,
                jnp.array([beta])[None,...],
                log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax
            )[0,0]
        return jax.vmap(one_i)(arK)
    logpi_off = jax.vmap(jax.vmap(logpi_off_all, in_axes=(0,0,0,None)),
                         in_axes=(0,0,0,0))(phi, m, rest if rest is not None else jnp.zeros((C,W,0)), betas)

    def qb_at(mask_ctx, ph_i, ph_all, rest_cw):
        return qb_density(ph_i, mask_ctx, ph_all, rest_cw)

    def lam_on_one(c, w):
        m_cw = m[c, w]; ph_cw = phi[c, w]; rest_cw = None if rest is None else rest[c, w]
        def on_for_slot(j):
            ctx = m_cw if qb_eval_variant == "parent" else m_cw.at[j].set(True)
            val = qb_at(ctx, ph_cw[j], ph_cw, rest_cw)
            return jnp.where(~m_cw[j], beta_cw[c,0] * val, 0.0)
        return jax.vmap(on_for_slot)(arK)
    lam_on = jax.vmap(jax.vmap(lam_on_one, in_axes=(None,0)), in_axes=(0,None))(jnp.arange(C), jnp.arange(W))

    def lam_off_one(c, w):
        m_cw = m[c, w]; ph_cw = phi[c, w]; rest_cw = None if rest is None else rest[c, w]
        def off_for_slot(i):
            ctx = m_cw if qb_eval_variant == "child" else m_cw.at[i].set(False)
            val = qb_at(ctx, ph_cw[i], ph_cw, rest_cw)
            ratio = jnp.exp(logpi_off[c,w,i] - logpi_cur[c,w])
            return jnp.where(m_cw[i], betas[c] * val * ratio, 0.0)
        return jax.vmap(off_for_slot)(arK)
    lam_off = jax.vmap(jax.vmap(lam_off_one, in_axes=(None,0)), in_axes=(0,None))(jnp.arange(C), jnp.arange(W))

    lam_total = lam_on.sum(-1) + lam_off.sum(-1)
    return lam_on, lam_off, lam_total


def bd_fire_on_chain_once(
    key: random.PRNGKey,
    state: PSState,
    c: int, w: int,
    lam_on_cw: Array, lam_off_cw: Array, lam_total_cw: Array,
    *,
    betas: Array, Kmax: int,
    sample_pseudo_phi: Callable[[random.PRNGKey], Array],
    log_prior_phi: LogPriorPhi, log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked, log_p_k: LogPk,
) -> Tuple[PSState, int, int, float]:
    """
    Fire exactly one BD event on (c,w):
      - choose birth/death slot proportional to hazards
      - toggle; refresh φ_j on death
      - recompute tempered logπ
    Returns: (new_state, kind, j, dt_chain) with kind 0=birth, 1=death
    """
    eps = jnp.finfo(state.phi.dtype).tiny
    key, k_dt, k_e, k_rs = random.split(key, 4)

    # dt for THIS chain
    u = random.uniform(k_dt, (), minval=eps, maxval=1.0)
    dt = -jnp.log(u) / jnp.maximum(lam_total_cw, eps)

    hazards = jnp.concatenate([lam_on_cw, lam_off_cw], axis=0)  # (2Kmax,)
    logits  = jnp.where(hazards > 0, jnp.log(hazards), -jnp.inf)
    g = random.gumbel(k_e, logits.shape)
    idx = jnp.argmax(logits + g)
    chosen_is_on = idx < lam_on_cw.shape[0]
    j = jnp.where(chosen_is_on, idx, idx - lam_on_cw.shape[0]).astype(jnp.int32)

    # toggle only (c,w)
    m = state.m
    m_cw = m[c, w]
    m_cw_new = jax.lax.select(chosen_is_on, m_cw.at[j].set(True), m_cw.at[j].set(False))
    m_new = m.at[c, w].set(m_cw_new)

    # refresh φ_j on death only
    phi = state.phi
    phi_cw = phi[c, w]
    def _set_sample(_):
        return phi_cw.at[j].set(sample_pseudo_phi(k_rs))
    phi_cw_new = jax.lax.cond(~chosen_is_on, _set_sample, lambda _: phi_cw, operand=None)
    phi_new = phi.at[c, w].set(phi_cw_new)

    # recompute tempered logπ
    logpi_new = _logpi_tempered(
        phi_new, m_new, state.rest, betas,
        log_prior_phi, log_pseudo_phi, log_lik_masked, log_p_k, Kmax
    )
    ps_new = PSState(phi=phi_new, rest=state.rest, m=m_new, logpi=logpi_new)
    kind = int(0) if bool(chosen_is_on) else int(1)
    return ps_new, kind, int(j), float(dt)


# ==============================
#       Likelihood batching
# ==============================

def batched_log_prob(
    log_prob_fn_single: Callable[[Array], Array],
    xs_flat: Array,
    chunk: int = 8192
) -> Array:
    B, D = xs_flat.shape
    f = jax.vmap(log_prob_fn_single)
    if (chunk is None) or (B <= chunk):
        return f(xs_flat)
    n_chunks = (B + chunk - 1) // chunk
    pad = n_chunks * chunk - B
    xs_pad = jnp.pad(xs_flat, ((0, pad), (0, 0)), mode="edge")
    xs_blocks = xs_pad.reshape((n_chunks, chunk, D))
    ys_blocks = jax.vmap(f)(xs_blocks)
    return ys_blocks.reshape((n_chunks * chunk,))[:B]

def mh_accept_masked(
    key: random.PRNGKey,
    current_lp: Array, proposed_lp: Array,
    betas: Array, log_qcorr: Array, move_mask: Array,
) -> Tuple[Array, Array]:
    delta = (proposed_lp - current_lp) * betas[:, None] + log_qcorr
    log_u = jnp.log(random.uniform(key, shape=current_lp.shape))
    accept = (log_u < jnp.minimum(0.0, delta)) & move_mask
    return accept, delta


# ==============================
#        MH Proposals
# ==============================

PROPOSAL_IDS: Dict[str, int] = {
    "stretch_A": 0,     # red half
    "stretch_B": 1,     # blue half
    "rw_fullcov": 2,
    "de_two_point": 3,
}

def _restrict_slot(prop_all: Array, cur_all: Array, slot_slice_or_mask) -> Array:
    if isinstance(slot_slice_or_mask, slice):
        sl = slot_slice_or_mask
        return cur_all.at[..., sl].set(prop_all[..., sl])
    else:
        maskD = slot_slice_or_mask.astype(jnp.bool_)[None, None, :]
        return jnp.where(maskD, prop_all, cur_all)


def gibbs_mh_single_sweep_over_active(
    key: random.PRNGKey,
    pt_state: PTState,                    # (C,W,D) + (C,W)
    m_mask_cwk: Array,                    # (C,W,Kmax)
    slot_slices: Tuple,                   # len Kmax; slice/bool mask over D for φ_j
    betas: Array,                         # (C,)
    *,
    do_stretch: bool,
    do_rw: bool,
    do_de: bool,
    stretch_keys, stretch_a: float,
    Ls: Array, evecs: Array, evals_pd: Array,
    scales: Dict[str, Array],
    de_keys, cross_rate: float, gamma_de: float,
    log_prob_fn_single: Callable[[Array], Array],
    lik_chunk: int,
    pt_trace: PTTrace,
) -> Tuple[random.PRNGKey, PTState, PTTrace]:
    C, W, D = pt_state.thetas.shape

    def mh_apply(rng_key, cur_state: PTState, prop, log_qcorr, move_mask):
        flat = prop.reshape((C * W, D))
        prop_lp = batched_log_prob(log_prob_fn_single, flat, chunk=lik_chunk).reshape((C, W))
        acc, _ = mh_accept_masked(rng_key, cur_state.log_probs, prop_lp, betas, log_qcorr, move_mask)
        th_new = jnp.where(acc[:, :, None], prop, cur_state.thetas)
        lp_new = jnp.where(acc,           prop_lp, cur_state.log_probs)
        return PTState(th_new, lp_new), acc

    def do_one_slot(carry, j):
        key_s, st, trace = carry
        move_mask = m_mask_cwk[:, :, j]
        sl = slot_slices[j]
        th_base = st.thetas

        # 1) Stretch red/blue restricted to active walkers
        if do_stretch:
            k_mask, k_p1, k_z1, k_p2, k_z2 = stretch_keys
            red, blue = redblue_mask(k_mask, C, W)
            red &= move_mask; blue &= move_mask

            # A) red half
            prop1, logJ1, _, has1, _ = _propose_stretch_redblue(k_p1, k_z1, th_base, subset_mask=red, a=stretch_a, z=None)
            prop1 = _restrict_slot(prop1, th_base, sl)
            key_s, k_acc1 = random.split(key_s)
            st, acc1 = mh_apply(k_acc1, st, prop1, logJ1, red & has1)
            trace = trace.append_always(st, acc1, PROPOSAL_IDS["stretch_A"], slot_j=int(j))

            # B) blue half
            prop2, logJ2, _, has2, _ = _propose_stretch_redblue(k_p2, k_z2, st.thetas, subset_mask=blue, a=stretch_a, z=None)
            prop2 = _restrict_slot(prop2, st.thetas, sl)
            key_s, k_acc2 = random.split(key_s)
            st, acc2 = mh_apply(k_acc2, st, prop2, logJ2, blue & has2)
            trace = trace.append_always(st, acc2, PROPOSAL_IDS["stretch_B"], slot_j=int(j))

        # 2) RW (example: fullcov)
        if do_rw:
            key_s, k_rw, k_acc3 = random.split(key_s, 3)
            prop_rw = _propose_fullcov(k_rw, st.thetas, Ls, scale=scales["small"])
            prop_rw = _restrict_slot(prop_rw, st.thetas, sl)
            st, acc3 = mh_apply(k_acc3, st, prop_rw, jnp.zeros_like(st.log_probs), move_mask)
            trace = trace.append_always(st, acc3, PROPOSAL_IDS["rw_fullcov"], slot_j=int(j))

        # 3) DE (partners restricted on apply via move_mask; your impl should accept eligible_mask)
        if do_de:
            key_s, kp, kg, k_acc4 = random.split(key_s, 4)
            prop_de, idx_y, idx_z, has_pair, xmask = _propose_de_two_point(
                kp, kg, st.thetas,
                gamma=None, gamma_scale=gamma_de, crossover_rate=cross_rate, jitter_scale=1e-6,
                eligible_mask=move_mask,   # your modified signature
            )
            prop_de = _restrict_slot(prop_de, st.thetas, sl)
            st, acc4 = mh_apply(k_acc4, st, prop_de, jnp.zeros_like(st.log_probs), move_mask & has_pair)
            trace = trace.append_always(st, acc4, PROPOSAL_IDS["de_two_point"], slot_j=int(j))

        return (key_s, st, trace), ()

    (key_out, st_out, trace_out), _ = jax.lax.scan(
        do_one_slot, (key, pt_state, pt_trace), jnp.arange(len(slot_slices))
    )
    return key_out, st_out, trace_out


# ==============================
#         PT swap pass
# ==============================

def pt_swap_pass(
    key: random.PRNGKey,
    state: PTState,
    betas: Array,
    even_pass: bool,
) -> Tuple[PTState, Array, Array]:
    thetas, lps = state.thetas, state.log_probs
    C, W, D = thetas.shape
    start = 0 if even_pass else 1
    idx_low  = jnp.arange(start, C - 1, 2)
    idx_high = idx_low + 1

    bi, bj = betas[idx_low], betas[idx_high]
    li = lps[idx_low, :]
    lj = lps[idx_high, :]
    delta = (bi - bj)[:, None] * (lj - li)
    log_u = jnp.log(random.uniform(key, shape=delta.shape))
    accept_pairs = log_u < jnp.minimum(0.0, delta)

    def swap_pair(carry, p):
        th, lp = carry
        i = idx_low[p]; j = idx_high[p]
        mask = accept_pairs[p]
        mask_wd = mask[:, None]
        thi, thj = th[i], th[j]
        lpi, lpj = lp[i], lp[j]
        new_i = jnp.where(mask_wd, thj, thi)
        new_j = jnp.where(mask_wd, thi, thj)
        new_lpi = jnp.where(mask, lpj, lpi)
        new_lpj = jnp.where(mask, lpi, lpj)
        th = th.at[i].set(new_i); th = th.at[j].set(new_j)
        lp = lp.at[i].set(new_lpi); lp = lp.at[j].set(new_lpj)
        return (th, lp), ()

    (thetas_new, lps_new), _ = lax.scan(swap_pair, (thetas, lps), jnp.arange(idx_low.shape[0]))

    # optional accept/attempt masks
    acc_mask = jnp.zeros((C, W), dtype=jnp.bool_)
    def mark_pair(acc, p):
        i = idx_low[p]; j = idx_high[p]
        m = accept_pairs[p]
        acc = acc.at[i].set(m | acc[i])
        acc = acc.at[j].set(m | acc[j])
        return acc, ()
    acc_mask, _ = lax.scan(mark_pair, acc_mask, jnp.arange(idx_low.shape[0]))

    att_mask = jnp.zeros((C, W), dtype=jnp.bool_)
    def mark_attempt(acc, p):
        i = idx_low[p]; j = idx_high[p]
        m = jnp.ones((W,), dtype=jnp.bool_)
        acc = acc.at[i].set(m | acc[i])
        acc = acc.at[j].set(m | acc[j])
        return acc, ()
    att_mask, _ = lax.scan(mark_attempt, att_mask, jnp.arange(idx_low.shape[0]))

    return PTState(thetas=thetas_new, log_probs=lps_new), acc_mask, att_mask


# ==============================
#         Main Runner
# ==============================

def run_epoch_ct(
    key: random.PRNGKey,
    pt_init: PTState,
    ps_init: PSState,
    *,
    T_end: float,
    rho_mh: float,
    MH_BATCH: int,
    betas: Array, Kmax: int,
    qb_density: QbDensity,
    qb_eval_variant: Literal["child","parent"],
    log_prior_phi: LogPriorPhi, log_pseudo_phi: LogPseudoPhi,
    log_lik_masked: LogLikMasked, log_p_k: LogPk,
    sample_pseudo_phi: Callable[[random.PRNGKey], Array],
    slot_slices: Tuple,  # mapping φ_j -> coordinates in θ
    stretch_a: float, scales: Dict[str, Array], evecs: Array, evals_pd: Array, Ls: Array,
    cross_rate: float, gamma_de: float,
    log_prob_fn_single: Callable[[Array], Array],
    lik_chunk: int = 64,
    max_steps: int = 10_000,
    max_ps_events: int = 1_000_000,
    max_pt_moves: int = 1_000_000,
    max_bd_per_window: int = 1024,
):
    """
    Hybrid scheduler:
      - Draw τ_mh ~ Exp(ρ); within [t, t+τ_mh) simulate BD independently per walker (could be many).
      - After BD loop, do MH (Gibbs over active φ_j): SM → RW → DE; then PT swaps.
      - Restart BD clocks after each MH tick (memoryless).
    Returns:
      final PTState, final PSState, PTTrace (MH submoves), PSTrace (BD snapshots), EventLog
    """
    key = random.fold_in(key, 0)
    C, W, D = pt_init.thetas.shape
    d = ps_init.phi.shape[-1]

    # traces
    pt_trace = PTTrace.init(max_pt_moves, C, W, D, dtype=pt_init.thetas.dtype)
    # optionally record the initial state as a move with prop_id=-1:
    zero_acc = jnp.zeros((C, W), dtype=jnp.bool_)
    pt_trace = pt_trace.append_always(pt_init, zero_acc, prop_id=-1, slot_j=-1)

    ps_trace = PSTrace.init(max_ps_events, C, W, Kmax, d, dtype=ps_init.phi.dtype, mdtype=ps_init.m.dtype).append(ps_init)
    ev_log   = EventLog.init(max_ps_events)

    t = jnp.array(0.0, dtype=jnp.float32)
    step = 0

    pt_state = pt_init
    ps_state = ps_init

    def cond_fun(carry):
        step, t, *_ = carry
        return jnp.logical_and(step < max_steps, t < T_end)

    def body_fun(carry):
        step, t, key, pt_state, ps_state, pt_trace, ps_trace, ev_log = carry
        key, k_mh, k_loop = random.split(key, 3)
        eps = jnp.finfo(ps_state.phi.dtype).tiny

        # Hazards at start of window
        lam_on, lam_off, lam_total = bd_hazards_only(
            ps_state, betas=betas, Kmax=Kmax,
            qb_density=qb_density, qb_eval_variant=qb_eval_variant,
            log_prior_phi=log_prior_phi, log_pseudo_phi=log_pseudo_phi,
            log_lik_masked=log_lik_masked, log_p_k=log_p_k,
        )

        # Draw MH window length
        u_mh = random.uniform(k_mh, (), minval=eps, maxval=1.0)
        tau_mh = -jnp.log(u_mh) / jnp.maximum(rho_mh, eps)
        rem = tau_mh

        # BD loop inside the window
        key_loop_local = k_loop
        bd_count = 0

        def bd_cond(state_tuple):
            rem_, bd_cnt, *_ = state_tuple
            return jnp.logical_and(rem_ > 0.0, bd_cnt < max_bd_per_window)

        def bd_body(state_tuple):
            rem_, bd_cnt, key_l, ps_cur, lam_on_cur, lam_off_cur, lam_tot_cur, pt_tr, ps_tr, ev = state_tuple
            C_, W_ = lam_tot_cur.shape
            key_l, k_dt, k_fire = random.split(key_l, 3)

            # candidate dt per chain
            eps_loc = jnp.finfo(ps_cur.phi.dtype).tiny
            u_all = random.uniform(k_dt, (C_, W_), minval=eps_loc, maxval=1.0)
            dt_all = -jnp.log(u_all) / jnp.maximum(lam_tot_cur, eps_loc)  # inf where lam=0

            # earliest chain
            dt_min_flat = dt_all.reshape((-1,))
            idx_flat = jnp.argmin(dt_min_flat)
            dt_min = dt_min_flat[idx_flat]
            c_min = (idx_flat // W_).astype(jnp.int32)
            w_min = (idx_flat %  W_).astype(jnp.int32)

            def stop_case(_):
                return (0.0, bd_cnt, key_l, ps_cur, lam_on_cur, lam_off_cur, lam_tot_cur, pt_tr, ps_tr, ev), False

            def fire_case(_):
                ps_next, kind, j_slot, dt_chain = bd_fire_on_chain_once(
                    k_fire, ps_cur, int(c_min), int(w_min),
                    lam_on_cur[c_min, w_min], lam_off_cur[c_min, w_min], lam_tot_cur[c_min, w_min],
                    betas=betas, Kmax=Kmax,
                    sample_pseudo_phi=sample_pseudo_phi,
                    log_prior_phi=log_prior_phi, log_pseudo_phi=log_pseudo_phi,
                    log_lik_masked=log_lik_masked, log_p_k=log_p_k,
                )
                # record PS snapshot & BD event
                ps_tr2 = ps_tr.append(ps_next)
                ev2 = ev.append(kind=kind, dt=dt_chain, c=int(c_min), w=int(w_min), j=int(j_slot))

                # recompute hazards (simple path)
                lam_on_new, lam_off_new, lam_tot_new = bd_hazards_only(
                    ps_next, betas=betas, Kmax=Kmax,
                    qb_density=qb_density, qb_eval_variant=qb_eval_variant,
                    log_prior_phi=log_prior_phi, log_pseudo_phi=log_pseudo_phi,
                    log_lik_masked=log_lik_masked, log_p_k=log_p_k,
                )
                return (rem_ - dt_min, bd_cnt + 1, key_l, ps_next, lam_on_new, lam_off_new, lam_tot_new, pt_tr, ps_tr2, ev2), True

            (rem_new, bd_cnt_new, key_new, ps_new, lam_on_new, lam_off_new, lam_tot_new, pt_tr_new, ps_tr_new, ev_new), fired = \
                jax.lax.cond(dt_min < rem_, fire_case, stop_case, operand=None)

            # Continue with updated state if fired; else keep as is
            return (jnp.where(fired, rem_new, rem_),
                    jnp.where(fired, bd_cnt_new, bd_cnt),
                    key_new, ps_new, lam_on_new, lam_off_new, lam_tot_new, pt_tr_new, ps_tr_new, ev_new)

        (rem_out, bd_count_out, key_after_bd, ps_after_bd, lam_on_after, lam_off_after, lam_tot_after, pt_trace_mid, ps_trace_mid, ev_log_mid) = \
            jax.lax.while_loop(
                bd_cond, bd_body,
                (rem, bd_count, key_loop_local, ps_state, lam_on, lam_off, lam_total, pt_trace, ps_trace, ev_log)
            )

        # Advance time to MH tick
        t_new = t + tau_mh

        # MH batch (Gibbs over active), then PT swaps
        def one_batch(carry_b, _):
            key_b, pt_b, trace_b = carry_b
            k_mask, k_p1, k_z1, k_p2, k_z2 = random.split(key_b, 5)
            kp, kg = random.split(k_z2, 2)
            key_out, pt_next, trace_next = gibbs_mh_single_sweep_over_active(
                key_b, pt_b, ps_after_bd.m, slot_slices, betas,
                do_stretch=True, do_rw=True, do_de=True,
                stretch_keys=(k_mask, k_p1, k_z1, k_p2, k_z2),
                stretch_a=stretch_a,
                Ls=Ls, evecs=evecs, evals_pd=evals_pd, scales=scales,
                de_keys=(kp, kg),
                cross_rate=cross_rate, gamma_de=gamma_de,
                log_prob_fn_single=log_prob_fn_single, lik_chunk=lik_chunk,
                pt_trace=trace_b,
            )
            return (key_out, pt_next, trace_next), ()

        (key_sw, pt_after_mh, pt_trace_after), _ = jax.lax.scan(
            one_batch, (key_after_bd, pt_state, pt_trace_mid), jnp.arange(MH_BATCH)
        )
        key_sw, k_e, k_o = random.split(key_sw, 3)
        st_sw, _, _ = pt_swap_pass(k_e, pt_after_mh, betas, even_pass=True)
        st_sw, _, _ = pt_swap_pass(k_o, st_sw,        betas, even_pass=False)

        # Record one MH event with dt = tau_mh (c=w=j=-1)
        ev_log2 = ev_log_mid.append(kind=2, dt=float(tau_mh), c=-1, w=-1, j=-1)

        return (step + 1, t_new, key_sw, st_sw, ps_after_bd, pt_trace_after, ps_trace_mid, ev_log2)

    (step_f, t_f, key_f, pt_f, ps_f, pt_trace_f, ps_trace_f, ev_log_f) = jax.lax.while_loop(
        cond_fun, body_fun,
        (step, t, key, pt_state, ps_state, pt_trace, ps_trace, ev_log)
    )

    return pt_f, ps_f, pt_trace_f.truncate(), ps_trace_f.truncate(), ev_log_f.truncate()