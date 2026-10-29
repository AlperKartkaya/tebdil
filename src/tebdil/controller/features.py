"""Causal, label-free inference-time state feature extraction for TEBDIL controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor

from tebdil.controller.action_space import ControllerAction
from tebdil.core.excursions import compute_branch_similarity
from tebdil.core.policies import compute_last_token_uncertainty, compute_stagnation_ratio
from tebdil.core.state import BranchState, RecurrentAdapter, compute_rms


def _last_token_jsd(logits_p: Tensor, logits_q: Tensor) -> float:
    p = torch.softmax(logits_p[:, -1, :].detach().float(), dim=-1).clamp_min(1e-12)
    q = torch.softmax(logits_q[:, -1, :].detach().float(), dim=-1).clamp_min(1e-12)
    m = 0.5 * (p + q)
    kl_pm = float((p * (p.log() - m.log())).sum(dim=-1).mean().item())
    kl_qm = float((q * (q.log() - m.log())).sum(dim=-1).mean().item())
    return max(0.0, 0.5 * (kl_pm + kl_qm))


FEATURE_SCHEMA: Tuple[str, ...] = (
    "bias",
    "is_continue",
    "is_exit",
    "is_excursion",
    "norm_step",
    "norm_budget",
    "entropy_a",
    "margin_a",
    "top1_prob_a",
    "delta_top1_prob_a",
    "stagnation_ratio_a",
    "update_cosine_a",
    "native_gate_prob_a",
    "exit_x_margin_a",
    "exit_x_entropy_a",
    "exit_x_gate_a",
    "exit_x_top1_prob_a",
    "exc_x_entropy_a",
    "exc_x_low_margin_a",
    "exc_x_donor_margin_gain",
    "exc_x_donor_entropy_red",
    "exc_x_donor_disagreement",
    "exc_x_donor_jsd",
    "exc_x_donor_rel_dist",
    "exc_x_alpha",
    "exc_x_beta",
    "exc_x_eff_step_size",
    "exc_x_is_last_token_scope",
    "exc_x_prev_excursions",
)


@dataclass(frozen=True)
class DonorFeatures:
    donor_index: int
    branch_id: str
    frob_distance: float
    rel_distance: float
    cosine_sim: float
    donor_entropy: float
    donor_margin: float
    donor_top1_prob: float
    donor_top1_token: int
    margin_gain: float
    entropy_reduction: float
    top1_disagreement: float
    logit_jsd: float


@dataclass(frozen=True)
class StateFeatures:
    step: int
    total_steps: int
    normalized_step: float
    budget_remaining: int
    normalized_budget: float
    entropy_a: float
    margin_a: float
    top1_prob_a: float
    top1_token_a: int
    delta_top1_prob_a: float
    delta_margin_a: float
    stagnation_ratio_a: float
    update_cosine_a: float
    rms_a: float
    native_gate_prob_a: float
    prev_excursions: int
    prev_excursion_margin_delta: float
    donors: Tuple[DonorFeatures, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_top1_prob_and_token(logits: Tensor) -> Tuple[float, int]:
    last_logits = logits[:, -1, :].detach().float()
    probs = torch.softmax(last_logits, dim=-1)
    top_val, top_idx = probs.max(dim=-1)
    return float(top_val.mean().item()), int(top_idx[0].item())


def _compute_recent_update_cosine(state: BranchState) -> float:
    """Cosine similarity between V_{t-1} = H_t - H_{t-1} and V_{t-2} = H_{t-1} - H_{t-2}."""
    if len(state.hidden_history) < 2:
        return 1.0
    h_curr = state.hidden_history[-1].detach().float().reshape(-1)
    h_prev = state.hidden_history[-2].detach().float().reshape(-1)
    v_last = h_curr - h_prev
    if len(state.hidden_history) >= 3:
        h_prev2 = state.hidden_history[-3].detach().float().reshape(-1)
        v_prev = h_prev - h_prev2
    else:
        v_prev = h_prev
    n1 = float(torch.linalg.vector_norm(v_last).item())
    n2 = float(torch.linalg.vector_norm(v_prev).item())
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    return float(torch.dot(v_last, v_prev).item() / (n1 * n2))


def _compute_native_gate_prob(state: BranchState) -> float:
    if not state.gate_history:
        return 0.5
    last_gate = state.gate_history[-1][:, -1, :].detach().float()
    return float(torch.sigmoid(last_gate).mean().item())


def extract_state_features(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    donors: Sequence[BranchState],
    total_steps: int = 4,
    budget_remaining: int = 4,
    prev_excursions: int = 0,
    prev_excursion_margin_delta: float = 0.0,
    prev_top1_prob_a: Optional[float] = None,
    prev_margin_a: Optional[float] = None,
) -> StateFeatures:
    """Extract strictly causal, label-free inference-time features z_t."""
    logits_a = adapter.readout_logits(state_a)
    entropy_a, margin_a = compute_last_token_uncertainty(logits_a)
    top1_prob_a, top1_tok_a = _compute_top1_prob_and_token(logits_a)
    stag_a = compute_stagnation_ratio(state_a)
    upd_cos_a = _compute_recent_update_cosine(state_a)
    rms_a = compute_rms(state_a.hidden)
    gate_prob_a = _compute_native_gate_prob(state_a)

    delta_prob = (top1_prob_a - prev_top1_prob_a) if prev_top1_prob_a is not None else 0.0
    delta_mar = (margin_a - prev_margin_a) if prev_margin_a is not None else 0.0

    donor_feats: List[DonorFeatures] = []
    for idx, d_state in enumerate(donors):
        frob_d, rel_d, cos_d = compute_branch_similarity(state_a.hidden, d_state.hidden)
        logits_d = adapter.readout_logits(d_state)
        ent_d, mar_d = compute_last_token_uncertainty(logits_d)
        prob_d, tok_d = _compute_top1_prob_and_token(logits_d)
        jsd_d = _last_token_jsd(logits_a, logits_d)
        donor_feats.append(
            DonorFeatures(
                donor_index=idx,
                branch_id=d_state.branch_id,
                frob_distance=round(frob_d, 6),
                rel_distance=round(rel_d, 6),
                cosine_sim=round(cos_d, 6),
                donor_entropy=round(ent_d, 6),
                donor_margin=round(mar_d, 6),
                donor_top1_prob=round(prob_d, 6),
                donor_top1_token=tok_d,
                margin_gain=round(mar_d - margin_a, 6),
                entropy_reduction=round(entropy_a - ent_d, 6),
                top1_disagreement=1.0 if tok_d != top1_tok_a else 0.0,
                logit_jsd=round(jsd_d, 6),
            )
        )

    denom_steps = max(1, total_steps)
    denom_budget = max(1, total_steps * max(1, len(donors) + 1))
    return StateFeatures(
        step=state_a.recurrent_step,
        total_steps=total_steps,
        normalized_step=round(state_a.recurrent_step / float(denom_steps), 6),
        budget_remaining=budget_remaining,
        normalized_budget=round(budget_remaining / float(denom_budget), 6),
        entropy_a=round(entropy_a, 6),
        margin_a=round(margin_a, 6),
        top1_prob_a=round(top1_prob_a, 6),
        top1_token_a=top1_tok_a,
        delta_top1_prob_a=round(delta_prob, 6),
        delta_margin_a=round(delta_mar, 6),
        stagnation_ratio_a=round(stag_a, 6),
        update_cosine_a=round(upd_cos_a, 6),
        rms_a=round(rms_a, 6),
        native_gate_prob_a=round(gate_prob_a, 6),
        prev_excursions=int(prev_excursions),
        prev_excursion_margin_delta=round(prev_excursion_margin_delta, 6),
        donors=tuple(donor_feats),
    )


def featurize_state_action(
    state_features: StateFeatures | Dict[str, Any],
    action: ControllerAction | Dict[str, Any],
) -> List[float]:
    """Map (StateFeatures, ControllerAction) into a fixed-length numerical vector matching FEATURE_SCHEMA."""
    if isinstance(state_features, dict):
        sf = state_features
        donors = sf.get("donors", [])
    else:
        sf = state_features.to_dict()
        donors = sf.get("donors", [])

    if isinstance(action, dict):
        act_type = str(action.get("action_type", "CONTINUE"))
        d_idx = int(action.get("donor_index", 0))
        alpha = float(action.get("alpha", 0.0))
        beta = float(action.get("beta", 0.0))
        scope = str(action.get("token_scope", "full"))
    else:
        act_type = action.action_type
        d_idx = action.donor_index
        alpha = action.alpha
        beta = action.beta
        scope = action.token_scope

    is_cont = 1.0 if act_type == "CONTINUE" else 0.0
    is_exit = 1.0 if act_type == "EXIT" else 0.0
    is_exc = 1.0 if act_type == "EXCURSION" else 0.0

    d_feat = donors[d_idx] if (0 <= d_idx < len(donors)) else (donors[0] if donors else {})
    mar_gain = float(d_feat.get("margin_gain", 0.0))
    ent_red = float(d_feat.get("entropy_reduction", 0.0))
    disagree = float(d_feat.get("top1_disagreement", 0.0))
    jsd = float(d_feat.get("logit_jsd", 0.0))
    rel_dist = float(d_feat.get("rel_distance", 0.0))

    ent_a = float(sf.get("entropy_a", 0.0))
    mar_a = float(sf.get("margin_a", 0.0))
    prob_a = float(sf.get("top1_prob_a", 0.0))
    d_prob_a = float(sf.get("delta_top1_prob_a", 0.0))
    stag_a = float(sf.get("stagnation_ratio_a", 0.0))
    ucos_a = float(sf.get("update_cosine_a", 0.0))
    gate_a = float(sf.get("native_gate_prob_a", 0.5))
    n_step = float(sf.get("normalized_step", 0.5))
    n_budg = float(sf.get("normalized_budget", 0.5))
    prev_exc = float(sf.get("prev_excursions", 0))

    is_last_tok = 1.0 if scope == "last_token" else 0.0

    vec = [
        0.0,  # reserved zero baseline
        0.0,  # is_continue baseline anchored at 0
        is_exit,
        is_exc,
        is_exit * n_step,
        is_exit * n_budg,
        is_exit * stag_a,
        is_exit * ucos_a,
        is_exit * d_prob_a,
        is_exc * n_step,
        is_exc * n_budg,
        is_exc * stag_a,
        is_exc * ucos_a,
        is_exit * mar_a,
        is_exit * ent_a,
        is_exit * gate_a,
        is_exit * prob_a,
        is_exc * ent_a,
        is_exc * max(0.0, 1.0 - mar_a),
        is_exc * mar_gain,
        is_exc * ent_red,
        is_exc * disagree,
        is_exc * jsd,
        is_exc * rel_dist,
        is_exc * alpha,
        is_exc * beta,
        is_exc * max(0.0, 1.0 - 1.5 * abs(alpha - 0.65) - 1.5 * abs(beta - 0.65)),
        is_exc * is_last_tok,
        is_exc * prev_exc,
    ]
    return vec
