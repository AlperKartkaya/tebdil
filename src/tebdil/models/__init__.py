"""Model adapters and protocol definitions for TEBDIL."""

from tebdil.models.interface import (
    BranchState,
    ExcursionConfig,
    RecurrentAdapter,
    TransitionContext,
    validate_state_compatibility,
)
from tebdil.models.ouro_adapter import (
    PINNED_OURO_BASE_REV,
    PINNED_OURO_THINKING_REV,
    OuroRecurrentAdapter,
)
from tebdil.models.toy_adapter import ToyLoopedModel, ToyRecurrentAdapter

__all__ = [
    "BranchState",
    "ExcursionConfig",
    "OuroRecurrentAdapter",
    "PINNED_OURO_BASE_REV",
    "PINNED_OURO_THINKING_REV",
    "RecurrentAdapter",
    "ToyLoopedModel",
    "ToyRecurrentAdapter",
    "TransitionContext",
    "validate_state_compatibility",
]
