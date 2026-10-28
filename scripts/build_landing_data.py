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


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    final_results = json.loads((root / "reports" / "final_results.json").read_text(encoding="utf-8"))
    grid_summary = json.loads((root / "results" / "phase1_grid" / "grid_summary.json").read_text(encoding="utf-8"))
    dim_ablations = json.loads(
        (root / "results" / "phase3_ablations" / "dimension_ablations.json").read_text(encoding="utf-8")
    )
    phase3_traces = load_jsonl(root / "results" / "phase3_ablations" / "traces.jsonl")
    phase4_traces = load_jsonl(root / "results" / "phase4_adaptive" / "traces.jsonl")

    payload = {
        "meta": {
            "project": "TEBDİL",
            "title": "Cooperative Latent Excursions for Recurrent Reasoning",
            "model_id": "ByteDance/Ouro-1.4B-Thinking",
            "model_revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
            "architecture": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "total_ut_steps": 4,
                "rmsnorms_per_step": 96,
                "vocab_size": 49152,
            },
            "total_records_analyzed": final_results.get("total_records_analyzed", 634),
        },
        "final_results": final_results,
        "grid_summary": grid_summary,
        "dimension_ablations": dim_ablations,
        "traces": {
            "validation_phase3": phase3_traces,
            "test_phase4": phase4_traces,
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
