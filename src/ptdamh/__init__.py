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

__all__ = [
    "_build_epoch_components",
    "_propose_fullcov",
    "_propose_eigenline",
    "_propose_student_t",
    "_propose_pcn",
    "_pcn_logq_delta",
    "build_general_mixture_components_per_chain",
]

