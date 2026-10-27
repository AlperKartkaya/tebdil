"""Phase 1C: Systematic alpha x beta excursion grid and identity verification on Ouro-1.4B-Thinking."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.metrics import compare_logits, compute_token_logprob
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks import generate_task_suite


def main() -> int:
    out_dir = REPO_ROOT / "results" / "phase1_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "ByteDance/Ouro-1.4B-Thinking",
        revision=PINNED_OURO_THINKING_REV,
        local_files_only=True,
    )
    adapter = OuroRecurrentAdapter.from_pretrained(
        device="mps" if torch.backends.mps.is_available() else "cpu",
        dtype=torch.bfloat16,
        local_files_only=True,
    )

    problems = generate_task_suite("arithmetic", split="pilot", num_problems=6, seed=42)
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    betas = [0.0, 0.25, 0.50, 1.0]

    grid_rows = []
    for alpha in alphas:
        for beta in betas:
            jsds = []
            max_diffs = []
            curvs = []
            dlps = []
            flips = 0
            all_finite = True

            cfg = ExcursionConfig(
                alpha=alpha,
                beta=beta,
                max_relative_norm=0.35,
                visit_step=2,
                token_scope="full",
            )
            for prob in problems:
                ids = tokenizer.encode(prob.prompt, return_tensors="pt").to(adapter.device)
                gold_ids = tokenizer.encode(prob.gold_answer, add_special_tokens=False)

                s0 = run_scheduled_trajectory(adapter, ids, method="S0", seed=42, record_pca=False)
                e1 = run_scheduled_trajectory(
                    adapter,
                    ids,
                    method="E1",
                    perturbation_relative_norm=0.20,
                    excursion_config=cfg,
                    seed=42,
                    record_pca=False,
                )

                cmp = compare_logits(s0.primary_logits, e1.primary_logits)
                lp_s0 = compute_token_logprob(s0.primary_logits, gold_ids[0])
                lp_e1 = compute_token_logprob(e1.primary_logits, gold_ids[0])

                jsds.append(cmp.js_divergence)
                max_diffs.append(cmp.max_abs_diff)
                dlps.append(lp_e1 - lp_s0)
                if not cmp.top1_match:
                    flips += 1
                if e1.excursion_telemetry is not None:
                    curvs.append(e1.excursion_telemetry.curvature_frob)
                    if not e1.excursion_telemetry.post_finite:
                        all_finite = False

            row = {
                "alpha": alpha,
                "beta": beta,
                "mean_max_abs_logit_diff": round(float(sum(max_diffs) / len(max_diffs)), 6),
                "mean_js_divergence": round(float(sum(jsds) / len(jsds)), 6),
                "mean_curvature_frob": round(float(sum(curvs) / len(curvs)), 6) if curvs else 0.0,
                "mean_delta_gold_logprob": round(float(sum(dlps) / len(dlps)), 6),
                "top1_flip_count": flips,
                "num_problems": len(problems),
                "all_finite": all_finite,
            }
            grid_rows.append(row)

    summary_path = out_dir / "grid_summary.json"
    summary_path.write_text(json.dumps({"grid": grid_rows}, indent=2), encoding="utf-8")
    print(f"Wrote Phase 1C grid results to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
