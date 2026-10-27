"""Nonlinear curvature C_t quantification and causal ablation analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np
from scipy.stats import spearmanr


def analyze_curvature(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the nonlinear excursion curvature component C_t across interventions."""
    excursion_recs = [
        r for r in records if float(r.get("curvature_frob", 0.0)) > 0.0 or str(r.get("method", "")).startswith("E")
    ]
    if not excursion_recs:
        return {
            "num_excursion_records": 0,
            "mean_curvature_frob": 0.0,
            "std_curvature_frob": 0.0,
            "mean_relative_curvature": 0.0,
            "spearman_curvature_vs_delta_logprob": 0.0,
            "spearman_p_value": 1.0,
            "rescued_mean_curvature": 0.0,
            "non_rescued_mean_curvature": 0.0,
            "causal_ablation_comparison": {},
        }

    curv_norms = np.array([float(r.get("curvature_frob", 0.0)) for r in excursion_recs], dtype=np.float64)
    rel_curvs = np.array([float(r.get("curvature_relative", 0.0)) for r in excursion_recs], dtype=np.float64)
    delta_lps = np.array([float(r.get("delta_gold_logprob", 0.0)) for r in excursion_recs], dtype=np.float64)

    if len(curv_norms) >= 3 and np.std(curv_norms) > 1e-12 and np.std(delta_lps) > 1e-12:
        rho, pval = spearmanr(curv_norms, delta_lps)
        rho_val = float(rho) if np.isfinite(rho) else 0.0
        p_val = float(pval) if np.isfinite(pval) else 1.0
    else:
        rho_val, p_val = 0.0, 1.0

    rescued_curvs = [
        float(r.get("curvature_frob", 0.0))
        for r in excursion_recs
        if not bool(r.get("base_a_correct", False)) and bool(r.get("is_correct", False))
    ]
    unrescued_curvs = [
        float(r.get("curvature_frob", 0.0))
        for r in excursion_recs
        if not bool(r.get("base_a_correct", False)) and not bool(r.get("is_correct", False))
    ]

    # Compare E1 (full curvature), M1/E1_no_curv (zero curvature), E1_rand_curv (random direction curvature)
    by_m: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_m.setdefault(str(r.get("method", "")), []).append(r)

    causal_comp: Dict[str, Dict[str, float]] = {}
    for m_key in ("E1", "M1", "E1_no_curv", "E1_rand_curv", "E0", "E2"):
        if m_key in by_m:
            m_list = by_m[m_key]
            acc = float(np.mean([1.0 if r.get("is_correct") else 0.0 for r in m_list]))
            dlp = float(np.mean([float(r.get("delta_gold_logprob", 0.0)) for r in m_list]))
            c_norm = float(np.mean([float(r.get("curvature_frob", 0.0)) for r in m_list]))
            causal_comp[m_key] = {
                "accuracy": round(acc, 4),
                "mean_delta_gold_logprob": round(dlp, 5),
                "mean_curvature_frob": round(c_norm, 5),
            }

    return {
        "num_excursion_records": len(excursion_recs),
        "mean_curvature_frob": round(float(np.mean(curv_norms)), 6),
        "std_curvature_frob": round(float(np.std(curv_norms)), 6),
        "mean_relative_curvature": round(float(np.mean(rel_curvs)), 6),
        "spearman_curvature_vs_delta_logprob": round(rho_val, 5),
        "spearman_p_value": round(p_val, 5),
        "rescued_mean_curvature": round(float(np.mean(rescued_curvs)), 6) if rescued_curvs else 0.0,
        "non_rescued_mean_curvature": round(float(np.mean(unrescued_curvs)), 6) if unrescued_curvs else 0.0,
        "causal_ablation_comparison": causal_comp,
    }
