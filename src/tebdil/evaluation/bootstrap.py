"""Paired bootstrap confidence intervals, cluster-aware bootstrap, and exact McNemar tests."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np
from scipy.stats import binomtest


def paired_bootstrap_ci(
    baseline_correct: Sequence[int],
    method_correct: Sequence[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired accuracy difference (method - baseline) and instance-level bootstrap CI."""
    b = np.asarray(baseline_correct, dtype=np.float64)
    m = np.asarray(method_correct, dtype=np.float64)
    if len(b) != len(m):
        raise ValueError("Paired arrays must have identical length.")
    if len(b) == 0:
        return 0.0, 0.0, 0.0

    diffs = m - b
    mean_diff = float(np.mean(diffs))
    if np.all(diffs == diffs[0]):
        return mean_diff, mean_diff, mean_diff

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_diff, low, high


def cluster_paired_bootstrap_ci(
    cluster_ids: Sequence[str],
    baseline_correct: Sequence[int],
    method_correct: Sequence[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired accuracy difference and cluster-aware bootstrap CI resampling by problem_id."""
    if len(cluster_ids) != len(baseline_correct) or len(cluster_ids) != len(method_correct):
        raise ValueError("cluster_ids, baseline_correct, and method_correct must have identical length.")
    if len(cluster_ids) == 0:
        return 0.0, 0.0, 0.0

    cluster_diffs: Dict[str, List[float]] = {}
    for cid, b_val, m_val in zip(cluster_ids, baseline_correct, method_correct):
        cluster_diffs.setdefault(str(cid), []).append(float(m_val) - float(b_val))

    unique_cids = sorted(cluster_diffs.keys())
    c_means = np.asarray([np.mean(cluster_diffs[cid]) for cid in unique_cids], dtype=np.float64)
    mean_diff = float(np.mean(c_means))
    if len(c_means) <= 1 or np.all(c_means == c_means[0]):
        return mean_diff, mean_diff, mean_diff

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c_means), size=(n_resamples, len(c_means)))
    boot_means = c_means[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_diff, low, high


def mcnemar_exact_test(
    baseline_correct: Sequence[int],
    method_correct: Sequence[int],
) -> float:
    """Two-sided exact binomial McNemar test on discordant pairs."""
    b = np.asarray(baseline_correct, dtype=np.int64)
    m = np.asarray(method_correct, dtype=np.int64)
    if len(b) != len(m) or len(b) == 0:
        return 1.0
    rescued = int(np.sum((b == 0) & (m == 1)))
    degraded = int(np.sum((b == 1) & (m == 0)))
    n_discordant = rescued + degraded
    if n_discordant == 0:
        return 1.0
    res = binomtest(rescued, n=n_discordant, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def mcnemar_cluster_exact_test(
    cluster_ids: Sequence[str],
    baseline_correct: Sequence[int],
    method_correct: Sequence[int],
) -> float:
    """Cluster-level exact binomial sign test on per-problem mean accuracy differences."""
    if len(cluster_ids) == 0:
        return 1.0
    cluster_diffs: Dict[str, List[float]] = {}
    for cid, b_val, m_val in zip(cluster_ids, baseline_correct, method_correct):
        cluster_diffs.setdefault(str(cid), []).append(float(m_val) - float(b_val))
    pos = sum(1 for vals in cluster_diffs.values() if np.mean(vals) > 1e-9)
    neg = sum(1 for vals in cluster_diffs.values() if np.mean(vals) < -1e-9)
    n_disc = pos + neg
    if n_disc == 0:
        return 1.0
    return float(binomtest(pos, n=n_disc, p=0.5, alternative="two-sided").pvalue)


def discordant_power_analysis(num_unique_problems: int = 18, num_seeds: int = 3) -> Dict[str, Any]:
    """Summarize statistical resolution and minimum discordant pairs needed for exact p < 0.05."""
    min_zero_loss_rescues = 6  # 2 * (0.5 ** 6) = 0.03125 < 0.05
    table: List[Dict[str, Any]] = []
    for n_resc in range(1, min(11, num_unique_problems + 1)):
        p_zero_deg = float(binomtest(n_resc, n=n_resc, p=0.5, alternative="two-sided").pvalue)
        p_one_deg = float(binomtest(n_resc, n=n_resc + 1, p=0.5, alternative="two-sided").pvalue)
        table.append(
            {
                "rescued_clusters": n_resc,
                "accuracy_gain_pct_if_0_degraded": round(100.0 * n_resc / max(1, num_unique_problems), 2),
                "exact_p_if_0_degraded": round(p_zero_deg, 5),
                "exact_p_if_1_degraded": round(p_one_deg, 5),
            }
        )
    return {
        "num_unique_problems": int(num_unique_problems),
        "num_seeds": int(num_seeds),
        "num_seeded_evaluations": int(num_unique_problems * num_seeds),
        "min_discordant_clusters_for_p_lt_0_05": min_zero_loss_rescues,
        "min_accuracy_gain_pct_for_p_lt_0_05": round(
            100.0 * min_zero_loss_rescues / max(1, num_unique_problems), 2
        ),
        "discordant_resolution_table": table,
    }
