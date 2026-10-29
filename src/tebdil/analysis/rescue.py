"""Paired rescue, degradation, and statistical comparison analysis across methods."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from tebdil.evaluation.bootstrap import (
    discordant_power_analysis,
    mcnemar_exact_test,
    paired_bootstrap_ci,
)
from tebdil.evaluation.metrics import MethodPairedSummary, summarize_paired_records


def compute_rescue_matrix(
    records: Sequence[Dict[str, Any]],
    baseline_method: str = "S0",
) -> Dict[str, Any]:
    """Group records by method and compute paired rescue/degradation summaries against baseline."""
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        m = str(r.get("method", "unknown"))
        by_method.setdefault(m, []).append(r)

    base_recs = by_method.get(baseline_method)
    if base_recs is None and by_method:
        first_m = next(iter(by_method.keys()))
        base_recs = by_method[first_m]

    e1_recs = by_method.get("E1")

    summaries: Dict[str, Dict[str, Any]] = {}
    for method, m_recs in by_method.items():
        s: MethodPairedSummary = summarize_paired_records(
            method=method,
            records=m_recs,
            baseline_records=base_recs,
            e1_records=e1_recs,
            bootstrap_fn=paired_bootstrap_ci,
            mcnemar_fn=mcnemar_exact_test,
        )
        summaries[method] = asdict(s)

    # Also compute per-family breakdown
    families = sorted({str(r.get("family", "unknown")) for r in records})
    by_family: Dict[str, Dict[str, Any]] = {}
    if len(families) > 1:
        for fam in families:
            fam_recs = [r for r in records if str(r.get("family")) == fam]
            fam_by_m: Dict[str, List[Dict[str, Any]]] = {}
            for r in fam_recs:
                fam_by_m.setdefault(str(r.get("method")), []).append(r)
            fam_base = fam_by_m.get(baseline_method)
            fam_e1 = fam_by_m.get("E1")
            by_family[fam] = {}
            for method, m_recs in fam_by_m.items():
                s = summarize_paired_records(
                    method=method,
                    records=m_recs,
                    baseline_records=fam_base,
                    e1_records=fam_e1,
                    bootstrap_fn=paired_bootstrap_ci,
                    mcnemar_fn=mcnemar_exact_test,
                )
                by_family[fam][method] = asdict(s)

    # Compute per-seed breakdown
    seeds = sorted({int(r.get("seed", 42)) for r in records})
    unique_pids = sorted({str(r.get("problem_id", "unknown")) for r in records})
    per_seed: Dict[str, Dict[str, Any]] = {}
    for sd in seeds:
        sd_key = str(sd)
        per_seed[sd_key] = {}
        for method, m_recs in by_method.items():
            sd_recs = [r for r in m_recs if int(r.get("seed", 42)) == sd]
            if sd_recs:
                acc_sd = sum(1 for r in sd_recs if bool(r.get("is_correct", False))) / float(len(sd_recs))
                calls_sd = sum(float(r.get("recurrent_block_calls", 0)) for r in sd_recs) / float(len(sd_recs))
                per_seed[sd_key][method] = {
                    "num_problems": len(sd_recs),
                    "accuracy": round(acc_sd, 4),
                    "mean_recurrent_calls": round(calls_sd, 2),
                }

    num_seeded = len(base_recs) if base_recs is not None else 0
    return {
        "baseline_method": baseline_method,
        "num_unique_problems": len(unique_pids),
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_seeded_evaluations": num_seeded,
        "power_analysis": discordant_power_analysis(
            num_unique_problems=len(unique_pids),
            num_seeds=max(1, len(seeds)),
        ),
        "methods": summaries,
        "by_family": by_family,
        "per_seed": per_seed,
    }
