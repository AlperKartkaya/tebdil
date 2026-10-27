"""Core latent excursion, branching, perturbation, control, and policy operators."""

from tebdil.core.branching import fork_branches
from tebdil.core.controls import (
    execute_direct_state_mixing_step,
    execute_norm_matched_random_perturbation_step,
    execute_random_direction_excursion_step,
    execute_update_mixing_step,
    execute_visit_without_return_step,
)
from tebdil.core.excursions import (
    ExcursionTelemetry,
    build_visit_state,
    compute_branch_similarity,
    execute_excursion_step,
)
from tebdil.core.perturbations import (
    PerturbationReport,
    apply_perturbation,
    build_scope_mask,
)
from tebdil.core.policies import (
    AdaptiveExcursionPolicy,
    AdaptivePolicyConfig,
    PolicyDecision,
    compute_last_token_uncertainty,
    compute_stagnation_ratio,
)
from tebdil.core.scheduler import (
    StepSnapshot,
    TrajectoryScheduleResult,
    run_scheduled_trajectory,
)
from tebdil.core.state import (
    BranchState,
    ExcursionConfig,
    TransitionContext,
    check_tensor_finite,
    compute_rms,
    frobenius_norm,
    hash_tensor,
)

__all__ = [
    "AdaptiveExcursionPolicy",
    "AdaptivePolicyConfig",
    "BranchState",
    "ExcursionConfig",
    "ExcursionTelemetry",
    "PerturbationReport",
    "PolicyDecision",
    "StepSnapshot",
    "TrajectoryScheduleResult",
    "TransitionContext",
    "apply_perturbation",
    "build_scope_mask",
    "build_visit_state",
    "check_tensor_finite",
    "compute_branch_similarity",
    "compute_last_token_uncertainty",
    "compute_rms",
    "compute_stagnation_ratio",
    "execute_direct_state_mixing_step",
    "execute_excursion_step",
    "execute_norm_matched_random_perturbation_step",
    "execute_random_direction_excursion_step",
    "execute_update_mixing_step",
    "execute_visit_without_return_step",
    "fork_branches",
    "frobenius_norm",
    "hash_tensor",
    "run_scheduled_trajectory",
]
