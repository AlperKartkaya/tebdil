"""Paired rescue, degradation, and statistical comparison analysis across methods."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from tebdil.evaluation.bootstrap import mcnemar_exact_test, paired_bootstrap_ci
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

    summaries: Dict[str, Dict[str, Any]] = {}
    for method, m_recs in by_method.items():
        s: MethodPairedSummary = summarize_paired_records(
            method=method,
            records=m_recs,
            baseline_records=base_recs,
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
            by_family[fam] = {}
            for method, m_recs in fam_by_m.items():
                s = summarize_paired_records(
                    method=method,
                    records=m_recs,
                    baseline_records=fam_base,
                    bootstrap_fn=paired_bootstrap_ci,
                    mcnemar_fn=mcnemar_exact_test,
                )
                by_family[fam][method] = asdict(s)

    return {
        "baseline_method": baseline_method,
        "methods": summaries,
        "by_family": by_family,
    }
