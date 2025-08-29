"""Top-level package for PTDAMH utilities.

This module re-exports commonly used proposal helper functions so that they
are available directly under :mod:`ptdamh`.
"""

from .proposals import (
    _build_epoch_components,
    _propose_fullcov,
    _propose_eigenline,
    _propose_student_t,
    _propose_pcn,
    _pcn_logq_delta,
    build_general_mixture_components_per_chain,
)

from .ensemble_runner import (
    run_epoch_device_fast_ensemble,
    run_epoch_device_fast_M23_ensemble,
    run_adaptive_pt_device_fast,
)

__all__ = [
    "_build_epoch_components",
    "_propose_fullcov",
    "_propose_eigenline",
    "_propose_student_t",
    "_propose_pcn",
    "_pcn_logq_delta",
    "build_general_mixture_components_per_chain",
    "run_epoch_device_fast_ensemble",
    "run_epoch_device_fast_M23_ensemble",
    "run_adaptive_pt_device_fast",
]

