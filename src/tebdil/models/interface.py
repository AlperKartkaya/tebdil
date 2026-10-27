"""RecurrentAdapter protocol and model-agnostic interface definitions."""

from __future__ import annotations

from tebdil.core.state import (
    BranchState,
    ExcursionConfig,
    RecurrentAdapter,
    TransitionContext,
    validate_state_compatibility,
)

__all__ = [
    "BranchState",
    "ExcursionConfig",
    "RecurrentAdapter",
    "TransitionContext",
    "validate_state_compatibility",
]
