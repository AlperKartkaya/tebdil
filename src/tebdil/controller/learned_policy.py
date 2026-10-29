"""Self-directed learned controller and baseline policies for TEBDIL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from tebdil.controller.action_space import (
    ControllerAction,
    build_discrete_action_space,
    make_continue_action,
    make_excursion_action,
    make_exit_action,
)
from tebdil.controller.features import (
    FEATURE_SCHEMA,
    StateFeatures,
    extract_state_features,
    featurize_state_action,
)
from tebdil.core.state import BranchState, RecurrentAdapter, check_tensor_finite, compute_rms


@dataclass(frozen=True)
class ControllerDecision:
    action_type: str  # CONTINUE | EXCURSION | EXIT
    action_id: str
    action: str  # visit | no_visit | exit (backward compatible with PolicyDecision)
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
    token_scope: str = "full"
    predicted_advantage: float = 0.0
    action_scores: Dict[str, float] = field(default_factory=dict)
    controller_overhead_ms: float = 0.0
    state_features: Optional[Dict[str, Any]] = None
    chosen_depth: int = 2
    selected_donor: str = "none"
    effective_alpha: float = 0.0
    effective_beta: float = 0.0
    rejected_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    use_heuristic_gates: bool = True


@dataclass
class LearnedControllerCheckpoint:
    controller_id: str
    model_type: str  # "ridge" | "mlp"
    feature_schema: List[str]
    feature_mean: List[float]
    feature_std: List[float]
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    mlp_w1: List[List[float]] = field(default_factory=list)
    mlp_b1: List[float] = field(default_factory=list)
    mlp_w2: List[float] = field(default_factory=list)
    mlp_b2: float = 0.0
    alpha_beta_grid: List[List[float]] = field(
        default_factory=lambda: [[0.65, 0.65], [0.35, 0.50], [0.85, 0.85]]
    )
    token_scopes: List[str] = field(default_factory=lambda: ["full"])
    allow_exit: bool = True
    use_heuristic_gates: bool = True
    utility_version: str = "v2_shaped"
    excursion_min_advantage: float = 0.0
    exit_min_advantage: float = 0.0
    max_excursions_per_trajectory: int = 1
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LearnedControllerCheckpoint":
        data = json.loads(path.read_text(encoding="utf-8"))
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def predict_raw_score(self, feat_vec: Sequence[float]) -> float:
        x = np.asarray(feat_vec, dtype=np.float64)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        std = np.asarray(self.feature_std, dtype=np.float64)
        # Keep index 0..3 (bias + action indicator one-hots) unscaled if std==1 and mean==0
        x_norm = (x - mean) / np.where(std > 1e-8, std, 1.0)
        if self.model_type == "mlp" and self.mlp_w1:
            w1 = np.asarray(self.mlp_w1, dtype=np.float64)  # [D, H]
            b1 = np.asarray(self.mlp_b1, dtype=np.float64)  # [H]
            w2 = np.asarray(self.mlp_w2, dtype=np.float64)  # [H]
            h = np.tanh(x_norm @ w1 + b1)
            return float(np.dot(h, w2) + self.mlp_b2)
        w = np.asarray(self.weights, dtype=np.float64)
        return float(np.dot(x_norm, w) + self.bias)


class SelfDirectedController:
    """Inference-time controller over A = {CONTINUE, EXCURSION(d, alpha, beta), EXIT}.

    Strictly operates on causal inference-time observations z_t.
    """

    def __init__(
        self,
        mode: str = "learned_ridge",
        checkpoint: Optional[LearnedControllerCheckpoint] = None,
        checkpoint_path: Optional[Path] = None,
        min_relative_divergence: float = 0.002,
        max_relative_divergence: float = 0.85,
        rms_instability_limit: float = 40.0,
        e2_max_margin_trigger: float = 0.70,
        e2_min_entropy_trigger: float = 1.20,
        budget_exit_min_margin: float = 1.15,
        budget_exit_max_entropy: float = 0.65,
        max_excursions: int = 1,
        allow_exit: bool = True,
        use_heuristic_gates: Optional[bool] = None,
    ):
        self.mode = mode
        if checkpoint is None and checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = LearnedControllerCheckpoint.load(checkpoint_path)
        self.checkpoint = checkpoint
        self.min_relative_divergence = min_relative_divergence
        self.max_relative_divergence = max_relative_divergence
        self.rms_instability_limit = rms_instability_limit
        self.e2_max_margin_trigger = e2_max_margin_trigger
        self.e2_min_entropy_trigger = e2_min_entropy_trigger
        self.budget_exit_min_margin = budget_exit_min_margin
        self.budget_exit_max_entropy = budget_exit_max_entropy
        self.max_excursions = max_excursions
        self.allow_exit = allow_exit
        if use_heuristic_gates is None:
            self.use_heuristic_gates = (
                bool(checkpoint.use_heuristic_gates) if checkpoint is not None else True
            )
        else:
            self.use_heuristic_gates = bool(use_heuristic_gates)
        self._controller_calls = 0

    @property
    def controller_calls(self) -> int:
        return self._controller_calls

    def decide(
        self,
        adapter: RecurrentAdapter,
        state_a: BranchState,
        donors: Sequence[BranchState],
        total_steps: int,
        budget_remaining: int,
        base_alpha: float = 0.65,
        base_beta: float = 0.65,
        base_scope: str = "full",
        max_relative_norm: float = 0.35,
        prev_excursions: int = 0,
        prev_excursion_margin_delta: float = 0.0,
        prev_top1_prob_a: Optional[float] = None,
        prev_margin_a: Optional[float] = None,
        seed: int = 42,
    ) -> ControllerDecision:
        import time

        t0 = time.perf_counter()
        self._controller_calls += 1

        if not check_tensor_finite(state_a.hidden) or compute_rms(state_a.hidden) > self.rms_instability_limit:
            dt = (time.perf_counter() - t0) * 1000.0
            return ControllerDecision(
                action_type="CONTINUE",
                action_id="CONTINUE",
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
                token_scope=base_scope,
                controller_overhead_ms=round(dt, 4),
                chosen_depth=int(state_a.recurrent_step),
                selected_donor="none",
                effective_alpha=0.0,
                effective_beta=0.0,
                use_heuristic_gates=self.use_heuristic_gates,
            )

        sf: StateFeatures = extract_state_features(
            adapter=adapter,
            state_a=state_a,
            donors=donors,
            total_steps=total_steps,
            budget_remaining=budget_remaining,
            prev_excursions=prev_excursions,
            prev_excursion_margin_delta=prev_excursion_margin_delta,
            prev_top1_prob_a=prev_top1_prob_a,
            prev_margin_a=prev_margin_a,
        )

        valid_donor_indices = [
            d.donor_index
            for d in sf.donors
            if (
                check_tensor_finite(donors[d.donor_index].hidden)
                and compute_rms(donors[d.donor_index].hidden) <= self.rms_instability_limit
                and self.min_relative_divergence <= d.rel_distance <= self.max_relative_divergence
            )
        ]

        default_d = sf.donors[0] if sf.donors else None
        d0_ent = default_d.donor_entropy if default_d else 0.0
        d0_mar = default_d.donor_margin if default_d else 0.0
        d0_div = default_d.rel_distance if default_d else 0.0

        def _make_decision(
            act: ControllerAction,
            reason: str,
            adv: float = 0.0,
            scores: Optional[Dict[str, float]] = None,
            rejected: Optional[Dict[str, Dict[str, Any]]] = None,
        ) -> ControllerDecision:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            d_idx = act.donor_index if (0 <= act.donor_index < len(sf.donors)) else 0
            d_obj = sf.donors[d_idx] if sf.donors else None
            legacy_act = (
                "visit"
                if act.action_type == "EXCURSION"
                else ("exit" if act.action_type == "EXIT" else "no_visit")
            )
            if act.action_type == "EXCURSION":
                sel_donor = (
                    donors[d_idx].branch_id
                    if (0 <= d_idx < len(donors))
                    else f"D{d_idx}"
                )
                eff_a = float(act.alpha)
                eff_b = float(act.beta)
            else:
                sel_donor = "none"
                eff_a = 0.0
                eff_b = 0.0
            return ControllerDecision(
                action_type=act.action_type,
                action_id=act.action_id,
                action=legacy_act,
                donor_index=d_idx,
                reason=reason,
                entropy_a=sf.entropy_a,
                margin_a=sf.margin_a,
                donor_entropy=d_obj.donor_entropy if d_obj else d0_ent,
                donor_margin=d_obj.donor_margin if d_obj else d0_mar,
                relative_divergence=d_obj.rel_distance if d_obj else d0_div,
                stagnation_ratio=sf.stagnation_ratio_a,
                adaptive_alpha=act.alpha if act.action_type == "EXCURSION" else base_alpha,
                adaptive_beta=act.beta if act.action_type == "EXCURSION" else base_beta,
                token_scope=act.token_scope if act.action_type == "EXCURSION" else base_scope,
                predicted_advantage=round(adv, 6),
                action_scores={k: round(v, 5) for k, v in (scores or {}).items()},
                controller_overhead_ms=round(dt_ms, 4),
                state_features=sf.to_dict(),
                chosen_depth=int(state_a.recurrent_step),
                selected_donor=sel_donor,
                effective_alpha=round(eff_a, 4),
                effective_beta=round(eff_b, 4),
                rejected_actions=rejected or {},
                use_heuristic_gates=self.use_heuristic_gates,
            )

        # 1. Simple deterministic / stochastic baseline modes
        if self.mode == "always_continue":
            return _make_decision(make_continue_action(), "baseline_always_continue")

        if self.mode == "fixed_exit":
            return _make_decision(make_exit_action(), "baseline_fixed_exit")

        if self.mode == "always_excursion":
            if valid_donor_indices and prev_excursions < self.max_excursions and budget_remaining >= 1:
                act = make_excursion_action(
                    donor_index=valid_donor_indices[0],
                    alpha=base_alpha,
                    beta=base_beta,
                    token_scope=base_scope,
                    max_relative_norm=max_relative_norm,
                )
                return _make_decision(act, "baseline_always_excursion")
            return _make_decision(make_continue_action(), "no_valid_donor_or_budget")

        if self.mode == "random_controller":
            cands = [make_continue_action()]
            if self.allow_exit:
                cands.append(make_exit_action())
            if valid_donor_indices and prev_excursions < self.max_excursions and budget_remaining >= 1:
                for d_i in valid_donor_indices:
                    cands.append(
                        make_excursion_action(
                            donor_index=d_i,
                            alpha=base_alpha,
                            beta=base_beta,
                            token_scope=base_scope,
                            max_relative_norm=max_relative_norm,
                        )
                    )
            gen = torch.Generator(device="cpu").manual_seed(int(seed + state_a.recurrent_step * 97))
            pick = int(torch.randint(0, len(cands), (1,), generator=gen).item())
            chosen = cands[pick]
            rej = {
                c.action_id: {"score": None, "status": "not_sampled_uniform"}
                for c in cands
                if c.action_id != chosen.action_id
            }
            return _make_decision(chosen, "baseline_random_choice", rejected=rej)

        # 2. Gate-only ablation baseline (P1.1): applies exact E3 heuristic gates without learning
        if self.mode == "gate_only":
            if self.allow_exit and sf.margin_a >= 0.72:
                return _make_decision(
                    make_exit_action(),
                    "gate_only_margin_exit",
                    adv=sf.margin_a - 0.72,
                    rejected={"CONTINUE": {"score": 0.0, "status": "lower_priority_than_exit_gate"}},
                )
            uncertain_a = (sf.margin_a <= self.e2_max_margin_trigger) and (
                sf.entropy_a >= self.e2_min_entropy_trigger
            )
            if uncertain_a and valid_donor_indices and prev_excursions < self.max_excursions and budget_remaining >= 1:
                act = make_excursion_action(
                    donor_index=valid_donor_indices[0],
                    alpha=base_alpha,
                    beta=base_beta,
                    token_scope=base_scope,
                    max_relative_norm=max_relative_norm,
                )
                return _make_decision(
                    act,
                    "gate_only_uncertainty_excursion",
                    adv=self.e2_max_margin_trigger - sf.margin_a,
                    rejected={
                        "CONTINUE": {"score": 0.0, "status": "uncertainty_gate_triggered"},
                        "EXIT": {"score": None, "status": "gated_out_margin_below_0.72"},
                    },
                )
            return _make_decision(
                make_continue_action(),
                "gate_only_continue",
                rejected={
                    "EXIT": {"score": None, "status": "gated_out_margin_below_0.72"},
                    "EXCURSION": {"score": None, "status": "gated_out_not_uncertain_or_no_budget"},
                },
            )

        # 3. Heuristic baselines: E2 and Compute-Budgeted Heuristic (H_budget)
        if self.mode in ("e2_heuristic", "budgeted_heuristic"):
            if (
                self.mode == "budgeted_heuristic"
                and self.allow_exit
                and sf.margin_a >= self.budget_exit_min_margin
                and sf.entropy_a <= self.budget_exit_max_entropy
            ):
                return _make_decision(
                    make_exit_action(),
                    "budgeted_heuristic_confident_early_exit",
                    adv=sf.margin_a - self.budget_exit_min_margin,
                )

            uncertain_a = (sf.margin_a <= self.e2_max_margin_trigger) and (
                sf.entropy_a >= self.e2_min_entropy_trigger
            )
            if uncertain_a and valid_donor_indices and prev_excursions < self.max_excursions and budget_remaining >= 1:
                best_d = max(
                    valid_donor_indices,
                    key=lambda idx: sf.donors[idx].margin_gain + 0.5 * sf.donors[idx].entropy_reduction,
                )
                if self.mode == "budgeted_heuristic" and sf.donors[best_d].margin_gain < -0.25:
                    return _make_decision(
                        make_continue_action(),
                        "budgeted_heuristic_skip_weak_donor",
                    )
                act = make_excursion_action(
                    donor_index=best_d,
                    alpha=base_alpha,
                    beta=base_beta,
                    token_scope=base_scope,
                    max_relative_norm=max_relative_norm,
                )
                return _make_decision(act, f"{self.mode}_triggered_uncertainty")
            return _make_decision(make_continue_action(), f"{self.mode}_continue_normal")

        # 4. Phase 5 Donor-Selection Policies (random, nearest, farthest, confidence)
        if self.mode in ("donor_random", "donor_nearest", "donor_farthest", "donor_confidence"):
            if not valid_donor_indices or prev_excursions >= self.max_excursions or budget_remaining < 1:
                return _make_decision(make_continue_action(), "no_valid_donors")
            # Trigger only when uncertain (matching E2 gating) so donor selection is isolated cleanly
            uncertain_a = (sf.margin_a <= self.e2_max_margin_trigger) and (
                sf.entropy_a >= self.e2_min_entropy_trigger
            )
            if not uncertain_a:
                return _make_decision(make_continue_action(), f"{self.mode}_confident_no_visit")

            if self.mode == "donor_random":
                gen = torch.Generator(device="cpu").manual_seed(int(seed + state_a.recurrent_step * 131))
                pick = int(torch.randint(0, len(valid_donor_indices), (1,), generator=gen).item())
                chosen_d = valid_donor_indices[pick]
            elif self.mode == "donor_nearest":
                chosen_d = min(valid_donor_indices, key=lambda idx: sf.donors[idx].rel_distance)
            elif self.mode == "donor_farthest":
                chosen_d = max(valid_donor_indices, key=lambda idx: sf.donors[idx].rel_distance)
            else:  # donor_confidence
                chosen_d = max(valid_donor_indices, key=lambda idx: sf.donors[idx].donor_margin)

            act = make_excursion_action(
                donor_index=chosen_d,
                alpha=base_alpha,
                beta=base_beta,
                token_scope=base_scope,
                max_relative_norm=max_relative_norm,
            )
            return _make_decision(act, f"selected_{self.mode}_d{chosen_d}")

        # 5. Learned Self-Directed Controller (learned_ridge, learned_mlp, donor_learned)
        if self.checkpoint is None:
            raise ValueError(f"SelfDirectedController in mode '{self.mode}' requires a loaded checkpoint.")

        cont_action = make_continue_action()
        cont_vec = featurize_state_action(sf, cont_action)
        base_q = self.checkpoint.predict_raw_score(cont_vec)

        if self.mode == "donor_learned":
            if not valid_donor_indices or prev_excursions >= self.max_excursions or budget_remaining < 1:
                return _make_decision(cont_action, "no_valid_donors")
            uncertain_a = (sf.margin_a <= self.e2_max_margin_trigger) and (
                sf.entropy_a >= self.e2_min_entropy_trigger
            )
            if not uncertain_a:
                return _make_decision(cont_action, "donor_learned_confident_no_visit")
            d_scores: Dict[str, float] = {}
            best_d_act: Optional[ControllerAction] = None
            best_d_score = -1e9
            for d_idx in valid_donor_indices:
                cand_d = make_excursion_action(
                    donor_index=d_idx,
                    alpha=base_alpha,
                    beta=base_beta,
                    token_scope=base_scope,
                    max_relative_norm=max_relative_norm,
                )
                s_val = self.checkpoint.predict_raw_score(featurize_state_action(sf, cand_d)) - base_q
                d_scores[cand_d.action_id] = s_val
                if s_val > best_d_score:
                    best_d_score = s_val
                    best_d_act = cand_d
            assert best_d_act is not None
            rej_d = {
                aid: {"score": round(sc, 5), "status": "suboptimal_predicted_advantage"}
                for aid, sc in d_scores.items()
                if aid != best_d_act.action_id
            }
            return _make_decision(
                best_d_act,
                reason=f"selected_donor_learned_d{best_d_act.donor_index}",
                adv=best_d_score,
                scores=d_scores,
                rejected=rej_d,
            )

        ab_grid = [tuple(pair) for pair in self.checkpoint.alpha_beta_grid]
        scopes = list(self.checkpoint.token_scopes) if self.checkpoint.token_scopes else [base_scope]
        can_exit_base = bool(self.allow_exit and self.checkpoint.allow_exit)
        if self.use_heuristic_gates:
            allow_exit_flag = can_exit_base and (sf.margin_a >= 0.72)
            allow_exc_flag = (sf.margin_a <= self.e2_max_margin_trigger) and (
                sf.entropy_a >= self.e2_min_entropy_trigger
            )
        else:
            allow_exit_flag = can_exit_base
            allow_exc_flag = True

        candidate_actions = build_discrete_action_space(
            num_donors=len(donors),
            alpha_beta_grid=ab_grid,
            token_scopes=scopes,
            max_relative_norm=max_relative_norm,
            allow_exit=can_exit_base,
        )

        scores: Dict[str, float] = {"CONTINUE": 0.0}
        rejected_actions: Dict[str, Dict[str, Any]] = {}
        best_action: ControllerAction = cont_action
        best_adv: float = 0.0

        max_exc = min(self.max_excursions, self.checkpoint.max_excursions_per_trajectory)

        for cand in candidate_actions:
            if cand.action_type == "CONTINUE":
                continue
            if cand.action_type == "EXIT":
                if not allow_exit_flag:
                    rejected_actions[cand.action_id] = {
                        "score": None,
                        "status": "gated_out_margin_below_0.72" if can_exit_base else "exit_disabled",
                    }
                    continue
                vec = featurize_state_action(sf, cand)
                raw_adv = self.checkpoint.predict_raw_score(vec) - base_q
                adj_adv = raw_adv - self.checkpoint.exit_min_advantage
                scores[cand.action_id] = adj_adv
                if adj_adv > best_adv:
                    best_adv = adj_adv
                    best_action = cand
            elif cand.action_type == "EXCURSION":
                if cand.donor_index not in valid_donor_indices:
                    rejected_actions[cand.action_id] = {
                        "score": None,
                        "status": "invalid_or_divergent_donor",
                    }
                    continue
                if prev_excursions >= max_exc or budget_remaining < 1:
                    rejected_actions[cand.action_id] = {
                        "score": None,
                        "status": "excursion_budget_exhausted",
                    }
                    continue
                if not allow_exc_flag:
                    rejected_actions[cand.action_id] = {
                        "score": None,
                        "status": "gated_out_not_uncertain",
                    }
                    continue
                vec = featurize_state_action(sf, cand)
                raw_adv = self.checkpoint.predict_raw_score(vec) - base_q
                adj_adv = raw_adv - self.checkpoint.excursion_min_advantage
                scores[cand.action_id] = adj_adv
                if adj_adv > best_adv:
                    best_adv = adj_adv
                    best_action = cand

        for aid, sc in scores.items():
            if aid != best_action.action_id:
                rejected_actions[aid] = {
                    "score": round(sc, 5),
                    "status": "suboptimal_predicted_advantage",
                }

        gate_tag = "gated" if self.use_heuristic_gates else "ungated"
        reason = f"learned_{self.checkpoint.model_type}_{gate_tag}_selected_{best_action.action_id}"
        return _make_decision(
            best_action,
            reason=reason,
            adv=best_adv,
            scores=scores,
            rejected=rejected_actions,
        )
