"""Stepwise accuracy and logit confidence progression across recurrent depth."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np


def analyze_step_progression(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-step RMS, Frobenius norm, entropy, margin, and branch divergence."""
    step_buckets: Dict[int, Dict[str, List[float]]] = {}
    for rec in records:
        for snap in rec.get("step_snapshots", []):
            step = int(snap["step"])
            bucket = step_buckets.setdefault(
                step,
                {
                    "rms_A": [],
                    "frob_A": [],
                    "entropy_A": [],
                    "margin_A": [],
                    "ab_distance_frob": [],
                    "ab_relative_distance": [],
                    "ab_cosine_similarity": [],
                },
            )
            rms_map = snap.get("branch_rms", {})
            frob_map = snap.get("branch_frob", {})
            ent_map = snap.get("branch_entropy", {})
            mar_map = snap.get("branch_margin", {})
            if "A" in rms_map:
                bucket["rms_A"].append(float(rms_map["A"]))
            if "A" in frob_map:
                bucket["frob_A"].append(float(frob_map["A"]))
            if "A" in ent_map:
                bucket["entropy_A"].append(float(ent_map["A"]))
            if "A" in mar_map:
                bucket["margin_A"].append(float(mar_map["A"]))
            bucket["ab_distance_frob"].append(float(snap.get("ab_distance_frob", 0.0)))
            bucket["ab_relative_distance"].append(float(snap.get("ab_relative_distance", 0.0)))
            bucket["ab_cosine_similarity"].append(float(snap.get("ab_cosine_similarity", 1.0)))

    summary: Dict[int, Dict[str, float]] = {}
    for step in sorted(step_buckets.keys()):
        b = step_buckets[step]
        summary[step] = {
            k: round(float(np.mean(vals)), 6) if vals else 0.0
            for k, vals in b.items()
        }
    return {"steps": summary, "num_records": len(records)}
