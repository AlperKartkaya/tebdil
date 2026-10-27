"""Paired bootstrap confidence intervals and exact McNemar hypothesis tests."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.stats import binomtest


def paired_bootstrap_ci(
    baseline_correct: Sequence[int],
    method_correct: Sequence[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired accuracy difference (method - baseline) and bootstrap CI."""
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
