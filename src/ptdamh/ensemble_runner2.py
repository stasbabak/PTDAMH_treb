# src/ptdamh/epoch_runner2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple, Sequence
from typing import List, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax, random, vmap
import numpy as np
from jax.scipy.linalg import solve_triangular
from tqdm import trange, tqdm

from tqdm.auto import tqdm
from tqdm.auto import trange

from .utilities import (
    temperature_ladder,
    _empirical_cov,
    _shrink_spd,
    _circular_mean,
    _wrapped_diff,
    _empirical_cov_wrapped,
)

from .epoch import (
    PTState,
    PTTrace,
    EventLog,
    PROPOSAL_IDS,
    run_epoch,
)

# --------- containers for results ---------


@dataclass
class EpochRecord:
    trace: PTTrace
    log: EventLog

@dataclass
class MultiEpochResult:
    final: PTState
    covs: jnp.ndarray                   # (C, D, D) last updated covariances
    records: List[EpochRecord]          # traces & logs for all epochs
    accept_summaries: List[Dict[str, float]]
    covs_history: Optional[List[jnp.ndarray]] = None

# --------- small helpers ---------

def _summarize_acceptances(log: EventLog) -> Dict[str, float]:
    """Overall acceptance / swap rates per epoch, including per-RW-kind."""
    out: Dict[str, float] = {}

    # slot indices: 0=stretch, 1=RW, 2=DE, 3=PT
    # Stretch
    att = log.attempted[:, 0]  # (T,C,W)
    acc = log.accepted[:, 0]
    out["stretch_rate"] = float(acc.sum() / jnp.maximum(att.sum(), 1))

    # RW by kind
    rw_ids = log.ids[:, 1]  # (T,)
    for name in ("rw_fullcov", "rw_eigenline", "rw_student_t"):
        mask_t = (rw_ids == PROPOSAL_IDS[name])      # (T,)
        if mask_t.any():
            att_k = log.attempted[:, 1][mask_t]     # (Tk,C,W)
            acc_k = log.accepted[:, 1][mask_t]      # (Tk,C,W)
            out[f"{name}_rate"] = float(acc_k.sum() / jnp.maximum(att_k.sum(), 1))
        else:
            out[f"{name}_rate"] = float("nan")

    # DE
    att = log.attempted[:, 2]; acc = log.accepted[:, 2]
    out["de_rate"] = float(acc.sum() / jnp.maximum(att.sum(), 1))

    # PT swap (combined even/odd)
    att = log.attempted[:, 3]; acc = log.accepted[:, 3]
    out["pt_swap_rate"] = float(acc.sum() / jnp.maximum(att.sum(), 1))

    return out

def recompute_covs_from_records_window(
    covs_prev: jnp.ndarray,                 # (C, D, D) fallback where no data
    records: List[EpochRecord],             # each: .trace (PTTrace), .log (EventLog)
    *,
    fold_mask: Optional[jnp.ndarray] = None,# (D,) or None
    period: float = 1.0,
    include_pt_in_cov: bool = False,        # usually False (PT swaps ≠ local geometry)
    ridge: float = 1e-6,
    window_min_epochs: int = 20,
    window_frac: float = 0.40,
    min_samples: int = 2,                   # per temperature
) -> jnp.ndarray:

    """
    Build per-temperature covariances using accepted samples across a rolling window of epochs.
    Window rule:
      - if len(records) <= window_min_epochs: use all epochs
      - else: use last ceil(window_frac * len(records)) epochs
    """
    if len(records) == 0:
        return covs_prev

    # Determine window of epochs
    E = len(records)
    if E <= window_min_epochs:
        start = 0
    else:
        k = max(1, int(jnp.ceil(window_frac * E)))
        start = E - k
    sel = records[start:]  # selected epochs

    # Grab shapes from the latest epoch
    T, C, W, D = sel[-1].trace.thetas[1:].shape

    covs_new = []
    for c in range(C):
        # Collect accepted samples for temperature c across the window
        chunks = []
        for rec in sel:
            X_tcwd = rec.trace.thetas[1:, c, :, :]          # (T, W, D) post-iteration states
            mask_local = (rec.log.accepted[:, 0]            # stretch
                          | rec.log.accepted[:, 1]          # RW
                          | rec.log.accepted[:, 2])         # DE
            mask_cw = mask_local[:, c, :]                   # (T, W)
            if include_pt_in_cov:
                mask_cw = mask_cw | rec.log.accepted[:, 3][:, c, :]  # add PT if desired

            X_flat = X_tcwd.reshape(T * W, D)               # (T*W, D)
            m_flat = mask_cw.reshape(T * W)                 # (T*W,)
            # Keep only accepted samples
            # (Boolean indexing here is fine; this path isn't jitted.)
            X_sel = X_flat[m_flat]
            if X_sel.shape[0] > 0:
                chunks.append(X_sel)

        if len(chunks) == 0 or sum(x.shape[0] for x in chunks) < min_samples:
            # not enough data: keep previous cov
            covs_new.append(covs_prev[c])
            continue

        Xc = jnp.concatenate(chunks, axis=0)               # (Nc, D), Nc = accepted across window

        # Your wrapped empirical covariance (handles periodic dims)
        out = _empirical_cov_wrapped(Xc, fold_idx=fold_mask, period=period)
        Cc = out[0] if (isinstance(out, tuple) and len(out) > 0) else out
        Cc = Cc + ridge * jnp.eye(D, dtype=Cc.dtype)        # numeric stability
        covs_new.append(Cc)

    return jnp.stack(covs_new, axis=0)  # (C, D, D)
# --------- main multi-epoch driver ---------

def run_adaptive_epochs(
    key: random.PRNGKey,
    init_state: PTState,
    *,
    log_prob_fn_single: Callable[[jnp.ndarray], jnp.ndarray],
    n_epochs: int,
    steps_per_epoch: int,
    temperatures: jnp.ndarray,          # (C,)
    covs_init: jnp.ndarray,             # (C,D,D)
    # proposal controls
    W_sm: float,
    rw_weights: Tuple[float, float, float],
    W_de: float,
    # geometry
    fold_mask: Optional[jnp.ndarray] = None,
    period: float = 1.0,
    # infra
    lik_chunk: int = 96,
    ridge: float = 1e-6,
    adapt_cov: bool = True,                    ### if to adapt covariance between epochs or fix it
    scale_small: Optional[jnp.ndarray] = None,           # (C,)
    scale_line: Optional[jnp.ndarray] = None,            # (C,)
    scale_big: Optional[jnp.ndarray] = None,             # (C,)
    include_pt_in_cov: bool = False,
) -> MultiEpochResult:
    """
    Run `n_epochs`. After each epoch, recompute per-temp covariance using
    `_empirical_cov_wrapped` on *accepted* samples from that epoch.
    Collect (trace, log) for every epoch for posterior & diagnostics.
    """
    assert log_prob_fn_single is not None, "Provide your (boxed) log-prob function."
    
    tqdm_desc: str = "Epochs"
    tqdm_leave: bool = True

    state = init_state
    covs  = covs_init
    records: List[EpochRecord] = []
    accept_summaries: List[Dict[str, float]] = []
    covs_history: List[jnp.ndarray] = []

    use_tqdm = True
    pbar = tqdm(range(n_epochs), desc=tqdm_desc, leave=tqdm_leave, disable=not use_tqdm)



    k = key
    for ep in pbar:
        k, k_run = random.split(k)
        covs_history.append(covs)

        # ---- run one epoch ----
        state, trace, log = run_epoch(
            key=k_run,
            init_state=state,
            log_prob_fn_single=log_prob_fn_single,
            temperatures=temperatures,
            covs=covs,
            scale_small=scale_small,
            scale_line=scale_line,
            scale_big=scale_big,
            n_steps=steps_per_epoch,
            W_sm=W_sm,
            rw_weights=rw_weights,
            W_de=W_de,
            fold_mask=fold_mask,
            period=period,
            lik_chunk=lik_chunk,
        )

        # ---- collect artifacts ----
        records.append(EpochRecord(trace=trace, log=log))
        accept_summaries.append(_summarize_acceptances(log))

        # ---- covariance refresh (per temperature) using your utilities ----
        # Use the state at the end of each iteration (trace.thetas[1:])
        # do_adapt = adapt_cov and (adapt_until_epoch is None or (ep + 1) <= adapt_until_epoch) 
        if adapt_cov:
            covs = recompute_covs_from_records_window(
                covs_prev=covs,
                records=records,
                fold_mask=fold_mask,
                period=period,
                include_pt_in_cov=False,    # or True, if you want PT to contribute
                ridge=1e-6,
                window_min_epochs=20,
                window_frac=0.40,
                min_samples=2,
            )

        # carry final state to next epoch (already folded inside run_epoch)
        state = state

        # Keep it readable; round and skip NaNs elegantly
        if use_tqdm and (((ep + 1) % 10 == 0) or ((ep + 1) == n_epochs)):
            summ = accept_summaries[-1]
            def fmt(x): 
                try:    return f"{float(x):.2f}" if x == x else "nan"
                except: return str(x)
            pbar.set_description(f"{tqdm_desc} {ep+1}/{n_epochs}")
            pbar.set_postfix({
                "stretch": fmt(summ["stretch_rate"]),
                "rw_full": fmt(summ["rw_fullcov_rate"]),
                "rw_eig":  fmt(summ["rw_eigenline_rate"]),
                "rw_t":    fmt(summ["rw_student_t_rate"]),
                "DE":      fmt(summ["de_rate"]),
                "PT":      fmt(summ["pt_swap_rate"]),
            }, refresh=True)


    return MultiEpochResult(
        final=state,
        covs=covs,
        records=records,
        accept_summaries=accept_summaries,
        covs_history=covs_history,
    )


def resume_from_state_with_runner(
    key: random.PRNGKey,
    *,
    final_state: PTState,          # from previous run's end
    covs_last: jnp.ndarray,        # (C,D,D) from previous run's end
    log_prob_fn_single,            # Callable[[jnp.ndarray], jnp.ndarray]
    n_epochs: int,
    steps_per_epoch: int,
    temperatures: jnp.ndarray,     # must match previous ladder length/order
    # proposal controls
    W_sm: float,
    rw_weights: Tuple[float, float, float],
    W_de: float,
    # geometry
    fold_mask: Optional[jnp.ndarray] = None,
    period: float = 1.0,
    # infra
    lik_chunk: int = 96,
    ridge: float = 1e-6,
    scale_small: Optional[jnp.ndarray] = None,
    scale_line: Optional[jnp.ndarray] = None,
    scale_big: Optional[jnp.ndarray] = None,
    include_pt_in_cov: bool = False,
) -> MultiEpochResult:
    """
    Memory-friendly resume: just call run_adaptive_epochs starting from
    the provided final_state and covs_last. Fresh records are produced.
    """
    C_state = final_state.thetas.shape[0]
    C_covs  = covs_last.shape[0]
    assert temperatures.shape[0] == C_state == C_covs, \
        "Mismatch in number of temperatures between state/covs/temperatures."

    return run_adaptive_epochs(
        key=key,
        init_state=final_state,
        log_prob_fn_single=log_prob_fn_single,
        n_epochs=int(n_epochs),
        steps_per_epoch=int(steps_per_epoch),
        temperatures=temperatures,
        covs_init=covs_last,
        W_sm=W_sm,
        rw_weights=rw_weights,
        W_de=W_de,
        fold_mask=fold_mask,
        period=period,
        lik_chunk=lik_chunk,
        ridge=ridge,
        scale_small=scale_small,
        scale_line=scale_line,
        scale_big=scale_big,
        include_pt_in_cov=include_pt_in_cov,
    )

