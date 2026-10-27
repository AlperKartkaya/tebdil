"""Label-free adaptive excursion trigger and donor selection policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor

from tebdil.core.excursions import compute_branch_similarity
from tebdil.core.state import BranchState, RecurrentAdapter, check_tensor_finite, compute_rms


@dataclass(frozen=True)
class AdaptivePolicyConfig:
    min_relative_divergence: float = 0.002
    max_relative_divergence: float = 0.50
    min_entropy_trigger: float = 0.25
    max_margin_trigger: float = 3.5
    stagnation_ratio_threshold: float = 0.85
    donor_selection: str = "confidence_gain"  # fixed | random | diversity | mid_diverse | confidence_gain
    extra_visit_cost: int = 1
    rms_Instability_limit: float = 40.0


@dataclass(frozen=True)
class PolicyDecision:
    action: str  # "visit" | "no_visit"
    donor_index: int
    reason: str
    entropy_a: float
    margin_a: float
    donor_entropy: float
    donor_margin: float
    relative_divergence: float
    stagnation_ratio: float
    adaptive_alpha: float
    adaptive_beta: float


def compute_last_token_uncertainty(logits: Tensor) -> tuple[float, float]:
    """Compute (shannon_entropy_nats, top1_top2_margin) on the last token's logits."""
    last_logits = logits[:, -1, :].detach().float()
    probs = torch.softmax(last_logits, dim=-1)
    log_probs = torch.log_softmax(last_logits, dim=-1)
    entropy = float(-(probs * log_probs).sum(dim=-1).mean().item())
    if last_logits.shape[-1] >= 2:
        top2 = torch.topk(last_logits, k=2, dim=-1).values
        margin = float((top2[:, 0] - top2[:, 1]).mean().item())
    else:
        margin = 0.0
    return entropy, margin


def compute_stagnation_ratio(state: BranchState) -> float:
    """Compute relative step update norm ||H_t - H_{t-1}||_F / (||H_t||_F + eps)."""
    if len(state.hidden_history) < 2:
        return 0.5
    h_curr = state.hidden_history[-1].detach().float()
    h_prev = state.hidden_history[-2].detach().float()
    delta = float(torch.linalg.vector_norm(h_curr - h_prev).item())
    base = float(torch.linalg.vector_norm(h_curr).item())
    return delta / (base + 1e-12)


class AdaptiveExcursionPolicy:
    """Causal, label-free policy deciding WHEN, WHOM, and HOW MUCH to visit.

    Strictly prohibited from accessing ground-truth labels or validators.
    """

    def __init__(self, config: AdaptivePolicyConfig = AdaptivePolicyConfig()):
        self.config = config

    def decide(
        self,
        adapter: RecurrentAdapter,
        state_a: BranchState,
        donors: Sequence[BranchState],
        budget_remaining: int,
        base_alpha: float = 0.5,
        base_beta: float = 0.5,
        seed: int = 42,
    ) -> PolicyDecision:
        if not donors:
            return PolicyDecision(
                action="no_visit",
                donor_index=0,
                reason="no_donors",
                entropy_a=0.0,
                margin_a=0.0,
                donor_entropy=0.0,
                donor_margin=0.0,
                relative_divergence=0.0,
                stagnation_ratio=0.0,
                adaptive_alpha=base_alpha,
                adaptive_beta=base_beta,
            )

        if budget_remaining < self.config.extra_visit_cost:
            return PolicyDecision(
                action="no_visit",
                donor_index=0,
                reason="insufficient_budget",
                entropy_a=0.0,
                margin_a=0.0,
                donor_entropy=0.0,
                donor_margin=0.0,
                relative_divergence=0.0,
                stagnation_ratio=0.0,
                adaptive_alpha=base_alpha,
                adaptive_beta=base_beta,
            )

        if not check_tensor_finite(state_a.hidden) or compute_rms(state_a.hidden) > self.config.rms_Instability_limit:
            return PolicyDecision(
                action="no_visit",
                donor_index=0,
                reason="branch_a_unstable",
                entropy_a=0.0,
                margin_a=0.0,
                donor_entropy=0.0,
                donor_margin=0.0,
                relative_divergence=0.0,
                stagnation_ratio=0.0,
                adaptive_alpha=base_alpha,
                adaptive_beta=base_beta,
            )

        logits_a = adapter.readout_logits(state_a)
        entropy_a, margin_a = compute_last_token_uncertainty(logits_a)
        stag_a = compute_stagnation_ratio(state_a)

        # Evaluate candidate donors using only label-free signals
        valid_candidates = []
        for idx, d_state in enumerate(donors):
            if not check_tensor_finite(d_state.hidden):
                continue
            if compute_rms(d_state.hidden) > self.config.rms_Instability_limit:
                continue
            _, rel_div, _ = compute_branch_similarity(state_a.hidden, d_state.hidden)
            if rel_div < self.config.min_relative_divergence or rel_div > self.config.max_relative_divergence:
                continue
            d_logits = adapter.readout_logits(d_state)
            d_ent, d_margin = compute_last_token_uncertainty(d_logits)
            valid_candidates.append((idx, rel_div, d_ent, d_margin))

        if not valid_candidates:
            _, rel_div0, _ = compute_branch_similarity(state_a.hidden, donors[0].hidden)
            return PolicyDecision(
                action="no_visit",
                donor_index=0,
                reason="divergence_out_of_bounds_or_unstable",
                entropy_a=entropy_a,
                margin_a=margin_a,
                donor_entropy=0.0,
                donor_margin=0.0,
                relative_divergence=rel_div0,
                stagnation_ratio=stag_a,
                adaptive_alpha=base_alpha,
                adaptive_beta=base_beta,
            )

        # Select donor according to strategy
        mode = self.config.donor_selection
        if mode == "fixed":
            chosen = valid_candidates[0]
        elif mode == "random":
            gen = torch.Generator(device="cpu").manual_seed(int(seed + state_a.recurrent_step))
            pick = int(torch.randint(0, len(valid_candidates), (1,), generator=gen).item())
            chosen = valid_candidates[pick]
        elif mode == "diversity":
            chosen = max(valid_candidates, key=lambda x: x[1])
        elif mode == "mid_diverse":
            target_div = 0.5 * (self.config.min_relative_divergence + self.config.max_relative_divergence)
            chosen = min(valid_candidates, key=lambda x: abs(x[1] - target_div))
        else:  # confidence_gain
            chosen = max(valid_candidates, key=lambda x: (x[3] - margin_a) - 0.5 * (x[2] - entropy_a))

        chosen_idx, rel_div, d_ent, d_margin = chosen

        # Trigger condition: A exhibits high uncertainty / low margin OR donor has stronger margin
        uncertain_a = (entropy_a >= self.config.min_entropy_trigger) or (margin_a <= self.config.max_margin_trigger)
        donor_more_confident = (d_margin > margin_a + 0.05) or (d_ent < entropy_a - 0.02)
        stagnation_risk = stag_a <= self.config.stagnation_ratio_threshold

        if uncertain_a and (donor_more_confident or stagnation_risk):
            # Modulate beta by confidence differential so uncertain A relies moderately on promising donor
            conf_gain = max(0.0, min(1.0, (d_margin - margin_a) / (abs(margin_a) + 1.0)))
            adj_beta = float(max(0.2, min(0.85, base_beta * (0.8 + 0.4 * conf_gain))))
            return PolicyDecision(
                action="visit",
                donor_index=chosen_idx,
                reason="triggered_uncertainty_and_donor_signal",
                entropy_a=entropy_a,
                margin_a=margin_a,
                donor_entropy=d_ent,
                donor_margin=d_margin,
                relative_divergence=rel_div,
                stagnation_ratio=stag_a,
                adaptive_alpha=base_alpha,
                adaptive_beta=adj_beta,
            )

        return PolicyDecision(
            action="no_visit",
            donor_index=chosen_idx,
            reason="branch_a_confident_continue_normal",
            entropy_a=entropy_a,
            margin_a=margin_a,
            donor_entropy=d_ent,
            donor_margin=d_margin,
            relative_divergence=rel_div,
            stagnation_ratio=stag_a,
            adaptive_alpha=base_alpha,
            adaptive_beta=base_beta,
        )
