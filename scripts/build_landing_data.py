"""Compile real TEBDIL experimental artifacts into site/static/tebdil_data.js."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    final_results = load_json(root / "reports" / "final_results.json")
    grid_summary = load_json(root / "results" / "phase1_grid" / "grid_summary.json")
    dim_ablations = load_json(root / "results" / "phase3_ablations" / "dimension_ablations.json")

    phase3_traces = load_jsonl(root / "results" / "phase3_ablations" / "traces.jsonl")
    phase4_traces = load_jsonl(root / "results" / "phase4_adaptive" / "traces.jsonl")
    phase5_traces = load_jsonl(root / "results" / "phase5_donor" / "traces.jsonl")
    phase6_traces = load_jsonl(root / "results" / "phase6_selective" / "traces.jsonl")
    phase78_traces = load_jsonl(root / "results" / "phase7_8_controller_test" / "traces.jsonl")

    ridge_ckpt = load_json(root / "artifacts" / "controllers" / "tebdil_ridge_v1.json")
    mlp_ckpt = load_json(root / "artifacts" / "controllers" / "tebdil_mlp_v1.json")
    audit_report = load_json(root / "artifacts" / "evidence" / "audit_phase7_8_report.json")
    split_manifest = load_json(root / "artifacts" / "counterfactuals" / "split_manifest.json")

    payload = {
        "meta": {
            "project": "TEBDİL",
            "title": "Self-Directed & Cooperative Latent Excursions for Recurrent Reasoning",
            "model_id": "ByteDance/Ouro-1.4B-Thinking",
            "model_revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
            "architecture": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "total_ut_steps": 4,
                "rmsnorms_per_step": 96,
                "vocab_size": 49152,
            },
            "total_records_analyzed": final_results.get("total_records_analyzed", 1408),
        },
        "final_results": final_results,
        "grid_summary": grid_summary,
        "dimension_ablations": dim_ablations,
        "controllers": {
            "ridge": {
                "controller_type": ridge_ckpt.get("controller_type"),
                "metrics": ridge_ckpt.get("metrics", {}),
                "feature_names": ridge_ckpt.get("feature_names", []),
            },
            "mlp": {
                "controller_type": mlp_ckpt.get("controller_type"),
                "metrics": mlp_ckpt.get("metrics", {}),
                "feature_names": mlp_ckpt.get("feature_names", []),
            },
            "split_manifest": split_manifest,
        },
        "audit_report": audit_report,
        "traces": {
            "test_phase7_8": phase78_traces,
            "validation_phase5": phase5_traces,
            "validation_phase6": phase6_traces,
            "test_phase4": phase4_traces,
            "validation_phase3": phase3_traces,
        },
    }

    out_dir = root / "site" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tebdil_data.js"
    js_content = "window.TEBDIL_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    out_file.write_text(js_content, encoding="utf-8")
    print(f"Wrote {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
