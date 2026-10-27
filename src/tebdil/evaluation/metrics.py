"""Evaluation metrics: logit divergences, rescue/degradation rates, and compute normalized scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class LogitComparisonMetrics:
    max_abs_diff: float
    mean_abs_diff: float
    relative_diff: float
    kl_p_q: float
    js_divergence: float
    top1_match: bool
    topk_overlap: float


def compare_logits(
    logits_p: Tensor,
    logits_q: Tensor,
    top_k: int = 5,
) -> LogitComparisonMetrics:
    """Compute numerical and distributional distance between two logit tensors on the last token."""
    p_last = logits_p[:, -1, :].detach().float()
    q_last = logits_q[:, -1, :].detach().float()

    diff = (p_last - q_last).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    norm_p = float(torch.linalg.vector_norm(p_last).item())
    norm_diff = float(torch.linalg.vector_norm(p_last - q_last).item())
    rel_diff = norm_diff / (norm_p + 1e-12)

    prob_p = torch.softmax(p_last, dim=-1).clamp_min(1e-12)
    prob_q = torch.softmax(q_last, dim=-1).clamp_min(1e-12)
    prob_m = 0.5 * (prob_p + prob_q)

    kl_pq = float((prob_p * (prob_p.log() - prob_q.log())).sum(dim=-1).mean().item())
    kl_pm = float((prob_p * (prob_p.log() - prob_m.log())).sum(dim=-1).mean().item())
    kl_qm = float((prob_q * (prob_q.log() - prob_m.log())).sum(dim=-1).mean().item())
    jsd = max(0.0, 0.5 * (kl_pm + kl_qm))

    top1_p = int(p_last.argmax(dim=-1).item())
    top1_q = int(q_last.argmax(dim=-1).item())

    k = min(top_k, p_last.shape[-1])
    tk_p = set(torch.topk(p_last, k=k, dim=-1).indices[0].tolist())
    tk_q = set(torch.topk(q_last, k=k, dim=-1).indices[0].tolist())
    overlap = len(tk_p.intersection(tk_q)) / float(k)

    return LogitComparisonMetrics(
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        relative_diff=rel_diff,
        kl_p_q=kl_pq,
        js_divergence=jsd,
        top1_match=(top1_p == top1_q),
        topk_overlap=overlap,
    )


def compute_token_logprob(logits: Tensor, target_token_id: int) -> float:
    """Return log-probability of `target_token_id` at the final sequence position."""
    last_logits = logits[:, -1, :].detach().float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    idx = max(0, min(int(target_token_id), log_probs.shape[-1] - 1))
    return float(log_probs[0, idx].item())


@dataclass(frozen=True)
class MethodPairedSummary:
    method: str
    num_instances: int
    accuracy: float
    baseline_accuracy: float
    paired_accuracy_diff: float
    ci_95_low: float
    ci_95_high: float
    mcnemar_p_value: float
    rescue_rate: float
    rescued_count: int
    base_wrong_count: int
    degradation_rate: float
    degraded_count: int
    base_correct_count: int
    oracle_rescue_rate: float
    oracle_rescued_count: int
    oracle_eligible_count: int
    mean_gold_logprob: float
    mean_delta_gold_logprob: float
    mean_jsd_vs_baseline: float
    mean_curvature_frob: float
    mean_relative_curvature: float
    mean_recurrent_calls: float
    mean_estimated_flops: float
    mean_wall_time_ms: float
    instability_count: int


def summarize_paired_records(
    method: str,
    records: Sequence[Dict[str, Any]],
    baseline_records: Optional[Sequence[Dict[str, Any]]] = None,
    donor_records: Optional[Sequence[Dict[str, Any]]] = None,
    bootstrap_fn: Optional[Any] = None,
    mcnemar_fn: Optional[Any] = None,
) -> MethodPairedSummary:
    """Compute paired scientific metrics for `method` relative to `baseline_records` (S0)."""
    n = len(records)
    if n == 0:
        return MethodPairedSummary(
            method=method,
            num_instances=0,
            accuracy=0.0,
            baseline_accuracy=0.0,
            paired_accuracy_diff=0.0,
            ci_95_low=0.0,
            ci_95_high=0.0,
            mcnemar_p_value=1.0,
            rescue_rate=0.0,
            rescued_count=0,
            base_wrong_count=0,
            degradation_rate=0.0,
            degraded_count=0,
            base_correct_count=0,
            oracle_rescue_rate=0.0,
            oracle_rescued_count=0,
            oracle_eligible_count=0,
            mean_gold_logprob=0.0,
            mean_delta_gold_logprob=0.0,
            mean_jsd_vs_baseline=0.0,
            mean_curvature_frob=0.0,
            mean_relative_curvature=0.0,
            mean_recurrent_calls=0.0,
            mean_estimated_flops=0.0,
            mean_wall_time_ms=0.0,
            instability_count=0,
        )

    base_map: Dict[str, Dict[str, Any]] = {}
    if baseline_records is not None:
        for r in baseline_records:
            key = f"{r['problem_id']}:{r['seed']}"
            base_map[key] = r

    donor_map: Dict[str, bool] = {}
    if donor_records is not None:
        for r in donor_records:
            key = f"{r['problem_id']}:{r['seed']}"
            donor_map[key] = bool(r.get("donor_b_correct", r.get("is_correct", False)))

    m_corr: List[int] = []
    b_corr: List[int] = []
    rescued = 0
    base_wrong = 0
    degraded = 0
    base_correct = 0
    oracle_rescued = 0
    oracle_eligible = 0

    gold_lps: List[float] = []
    delta_lps: List[float] = []
    jsds: List[float] = []
    curvs: List[float] = []
    rel_curvs: List[float] = []
    calls_list: List[float] = []
    flops_list: List[float] = []
    wall_list: List[float] = []
    instabilities = 0

    for rec in records:
        key = f"{rec['problem_id']}:{rec['seed']}"
        base_rec = base_map.get(key, rec)
        is_c = int(bool(rec.get("is_correct", False)))
        is_b = int(bool(base_rec.get("base_a_correct", base_rec.get("is_correct", False))))
        is_donor_c = bool(
            rec.get(
                "donor_b_correct",
                donor_map.get(key, False),
            )
        )

        m_corr.append(is_c)
        b_corr.append(is_b)

        if is_b == 0:
            base_wrong += 1
            if is_c == 1:
                rescued += 1
            if is_donor_c:
                oracle_eligible += 1
                if is_c == 1:
                    oracle_rescued += 1
        else:
            base_correct += 1
            if is_c == 0:
                degraded += 1

        lp = float(rec.get("gold_logprob", 0.0))
        base_lp = float(base_rec.get("gold_logprob", lp))
        gold_lps.append(lp)
        delta_lps.append(lp - base_lp)
        jsds.append(float(rec.get("jsd_vs_s0", 0.0)))
        curvs.append(float(rec.get("curvature_frob", 0.0)))
        rel_curvs.append(float(rec.get("curvature_relative", 0.0)))
        calls_list.append(float(rec.get("recurrent_block_calls", 0)))
        flops_list.append(float(rec.get("estimated_flops", 0.0)))
        wall_list.append(float(rec.get("wall_time_ms", 0.0)))
        if not bool(rec.get("post_finite", True)) or bool(rec.get("skipped_unstable", False)):
            instabilities += 1

    acc = float(np.mean(m_corr))
    base_acc = float(np.mean(b_corr))
    diff = acc - base_acc

    if bootstrap_fn is not None:
        _, ci_low, ci_high = bootstrap_fn(b_corr, m_corr)
    else:
        ci_low, ci_high = diff, diff

    if mcnemar_fn is not None:
        p_val = mcnemar_fn(b_corr, m_corr)
    else:
        p_val = 1.0

    return MethodPairedSummary(
        method=method,
        num_instances=n,
        accuracy=acc,
        baseline_accuracy=base_acc,
        paired_accuracy_diff=diff,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        mcnemar_p_value=p_val,
        rescue_rate=(rescued / base_wrong) if base_wrong > 0 else 0.0,
        rescued_count=rescued,
        base_wrong_count=base_wrong,
        degradation_rate=(degraded / base_correct) if base_correct > 0 else 0.0,
        degraded_count=degraded,
        base_correct_count=base_correct,
        oracle_rescue_rate=(oracle_rescued / oracle_eligible) if oracle_eligible > 0 else 0.0,
        oracle_rescued_count=oracle_rescued,
        oracle_eligible_count=oracle_eligible,
        mean_gold_logprob=float(np.mean(gold_lps)),
        mean_delta_gold_logprob=float(np.mean(delta_lps)),
        mean_jsd_vs_baseline=float(np.mean(jsds)),
        mean_curvature_frob=float(np.mean(curvs)),
        mean_relative_curvature=float(np.mean(rel_curvs)),
        mean_recurrent_calls=float(np.mean(calls_list)),
        mean_estimated_flops=float(np.mean(flops_list)),
        mean_wall_time_ms=float(np.mean(wall_list)),
        instability_count=instabilities,
    )
