"""Compute-normalized accuracy and Pareto frontier calculation."""

from __future__ import annotations

from typing import Any, Dict, List


def compute_pareto_frontier(method_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Identify Pareto-efficient non-oracle methods trading off recurrent calls vs accuracy."""
    points: List[Dict[str, Any]] = []
    for method, stats in method_summaries.items():
        is_oracle = method in ("P1", "E_oracle")
        calls = float(stats.get("mean_recurrent_calls", 0.0))
        acc = float(stats.get("accuracy", 0.0))
        net_diff = float(stats.get("paired_accuracy_diff", 0.0))
        flops = float(stats.get("mean_estimated_flops", 0.0))
        wall = float(stats.get("mean_wall_time_ms", 0.0))
        points.append(
            {
                "method": method,
                "is_oracle": is_oracle,
                "mean_recurrent_calls": round(calls, 3),
                "mean_estimated_flops": flops,
                "mean_wall_time_ms": round(wall, 2),
                "accuracy": round(acc, 4),
                "paired_accuracy_diff": round(net_diff, 4),
                "gain_per_extra_call": round(net_diff / max(1.0, calls - 4.0), 5)
                if calls > 4.0
                else 0.0,
            }
        )

    non_oracle = [p for p in points if not p["is_oracle"]]
    pareto_optimal: List[str] = []
    for p in non_oracle:
        dominated = False
        for q in non_oracle:
            if q["method"] == p["method"]:
                continue
            if (
                q["mean_recurrent_calls"] <= p["mean_recurrent_calls"]
                and q["accuracy"] >= p["accuracy"]
                and (
                    q["mean_recurrent_calls"] < p["mean_recurrent_calls"]
                    or q["accuracy"] > p["accuracy"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(p["method"])

    points.sort(key=lambda x: (x["mean_recurrent_calls"], -x["accuracy"]))
    return {
        "points": points,
        "pareto_optimal_non_oracle": pareto_optimal,
    }
