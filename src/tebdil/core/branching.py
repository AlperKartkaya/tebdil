"""Branch creation and isolated multi-trajectory initialization."""

from __future__ import annotations

from typing import List, Tuple
from tebdil.core.perturbations import PerturbationReport, apply_perturbation
from tebdil.core.state import BranchState


def fork_branches(
    base_state: BranchState,
    count: int = 2,
    perturbation_method: str = "normalized_gaussian",
    relative_norm: float = 0.01,
    seed: int = 42,
    token_scope: str = "full",
    selected_positions: Tuple[int, ...] = (),
) -> tuple[List[BranchState], List[PerturbationReport]]:
    """Fork `base_state` into `count` isolated branches.

    Branch 0 ('A') is always an exact unperturbed clone of `base_state`.
    Branches 1..count-1 ('B', 'C', ...) receive seeded perturbations with distinct seeds.
    """
    if count < 1:
        raise ValueError(f"Branch count must be >= 1, got {count}")

    branches: List[BranchState] = []
    reports: List[PerturbationReport] = []

    branch_names = [chr(ord("A") + i) if i < 26 else f"B{i}" for i in range(count)]

    # Branch A: unperturbed reference trajectory
    branch_a = base_state.clone(new_branch_id=branch_names[0])
    _, rep_a = apply_perturbation(
        branch_a.hidden,
        method="none",
        relative_norm=0.0,
        seed=seed,
        token_scope=token_scope,
        selected_positions=selected_positions,
    )
    branches.append(branch_a)
    reports.append(rep_a)

    for idx in range(1, count):
        branch_seed = seed + idx * 1009
        cloned = base_state.clone(new_branch_id=branch_names[idx])
        perturbed_hidden, rep = apply_perturbation(
            cloned.hidden,
            method=perturbation_method,
            relative_norm=relative_norm,
            seed=branch_seed,
            token_scope=token_scope,
            selected_positions=selected_positions,
        )
        cloned.hidden = perturbed_hidden
        branches.append(cloned)
        reports.append(rep)

    return branches, reports
