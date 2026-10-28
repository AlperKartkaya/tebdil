"""Discrete and parameterized action space for the self-directed TEBDIL controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


VALID_ACTION_TYPES = ("CONTINUE", "EXCURSION", "EXIT")


@dataclass(frozen=True)
class ControllerAction:
    action_id: str
    action_type: str  # CONTINUE | EXCURSION | EXIT
    donor_index: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    token_scope: str = "full"
    max_relative_norm: float = 0.35

    def __post_init__(self) -> None:
        if self.action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action_type: {self.action_type}")


def make_continue_action() -> ControllerAction:
    return ControllerAction(
        action_id="CONTINUE",
        action_type="CONTINUE",
        donor_index=0,
        alpha=0.0,
        beta=0.0,
    )


def make_exit_action() -> ControllerAction:
    return ControllerAction(
        action_id="EXIT",
        action_type="EXIT",
        donor_index=0,
        alpha=0.0,
        beta=0.0,
    )


def make_excursion_action(
    donor_index: int = 0,
    alpha: float = 0.65,
    beta: float = 0.65,
    token_scope: str = "full",
    max_relative_norm: float = 0.35,
) -> ControllerAction:
    aid = f"EXCURSION_d{donor_index}_a{alpha:.2f}_b{beta:.2f}_{token_scope}"
    return ControllerAction(
        action_id=aid,
        action_type="EXCURSION",
        donor_index=int(donor_index),
        alpha=float(alpha),
        beta=float(beta),
        token_scope=str(token_scope),
        max_relative_norm=float(max_relative_norm),
    )


def build_discrete_action_space(
    num_donors: int = 1,
    alpha_beta_grid: Sequence[Tuple[float, float]] = (
        (0.35, 0.50),
        (0.65, 0.65),
        (0.85, 0.85),
    ),
    token_scopes: Sequence[str] = ("full",),
    max_relative_norm: float = 0.35,
    allow_exit: bool = True,
) -> List[ControllerAction]:
    """Construct the discrete candidate action set A = {CONTINUE, EXIT, EXCURSION(...)}."""
    actions: List[ControllerAction] = [make_continue_action()]
    if allow_exit:
        actions.append(make_exit_action())
    for d_idx in range(max(0, num_donors)):
        for alpha, beta in alpha_beta_grid:
            for scope in token_scopes:
                actions.append(
                    make_excursion_action(
                        donor_index=d_idx,
                        alpha=alpha,
                        beta=beta,
                        token_scope=scope,
                        max_relative_norm=max_relative_norm,
                    )
                )
    return actions
