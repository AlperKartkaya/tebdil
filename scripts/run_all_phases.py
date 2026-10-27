"""Execute Phases 1-5 on ByteDance/Ouro-1.4B-Thinking in a single model session."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tebdil.cli import cmd_analyze
from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.metrics import compare_logits, compute_token_logprob
from tebdil.evaluation.runner import (
    _decode_method_on_problem,
    AdaptivePolicyConfig,
    run_experiment_from_config,
)
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks import generate_task_suite


def run_phase1_grid_with_adapter(
    adapter: OuroRecurrentAdapter,
    tokenizer: Any,
) -> None:
    out_dir = REPO_ROOT / "results" / "phase1_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = generate_task_suite("arithmetic", split="pilot", num_problems=6, seed=42)
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    betas = [0.0, 0.25, 0.50, 1.0]

    # Cache S0 and step-2 B per problem
    s0_map = {}
    for prob in problems:
        ids = tokenizer.encode(prob.prompt, return_tensors="pt").to(adapter.device)
        gold_ids = tokenizer.encode(prob.gold_answer, add_special_tokens=False)
        s0 = run_scheduled_trajectory(adapter, ids, method="S0", seed=42, record_pca=False)
        s0_map[prob.problem_id] = (ids, gold_ids[0], s0)

    grid_rows = []
    for alpha in alphas:
        for beta in betas:
            jsds, max_diffs, curvs, dlps = [], [], [], []
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
                ids, gold_id0, s0 = s0_map[prob.problem_id]
                e1 = run_scheduled_trajectory(
                    adapter,
                    ids,
                    method="E1",
                    perturbation_relative_norm=0.25,
                    excursion_config=cfg,
                    seed=42,
                    record_pca=False,
                )
                cmp = compare_logits(s0.primary_logits, e1.primary_logits)
                lp_s0 = compute_token_logprob(s0.primary_logits, gold_id0)
                lp_e1 = compute_token_logprob(e1.primary_logits, gold_id0)
                jsds.append(cmp.js_divergence)
                max_diffs.append(cmp.max_abs_diff)
                dlps.append(lp_e1 - lp_s0)
                if not cmp.top1_match:
                    flips += 1
                if e1.excursion_telemetry is not None:
                    curvs.append(e1.excursion_telemetry.curvature_frob)
                    if not e1.excursion_telemetry.post_finite:
                        all_finite = False

            grid_rows.append(
                {
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
            )
    (out_dir / "grid_summary.json").write_text(
        json.dumps({"grid": grid_rows}, indent=2),
        encoding="utf-8",
    )


def run_dimension_ablations(
    adapter: OuroRecurrentAdapter,
    tokenizer: Any,
) -> None:
    """Ablate visit_step in {1, 2} and token_scope in {'last_token', 'full'} on validation problems."""
    out_dir = REPO_ROOT / "results" / "phase3_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = generate_task_suite("all", split="validation", num_problems=9, seed=42)
    pol_cfg = AdaptivePolicyConfig()

    rows = []
    for visit_step in (1, 2):
        for scope in ("last_token", "full"):
            cfg = ExcursionConfig(
                alpha=0.65,
                beta=0.65,
                max_relative_norm=0.35,
                visit_step=visit_step,
                token_scope=scope,
            )
            correct = 0
            curvs = []
            for p in probs:
                res = _decode_method_on_problem(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    problem=p,
                    method="E1",
                    seed=42,
                    recurrent_steps=4,
                    fork_after_step=1,
                    branch_count=2,
                    perturbation_method="normalized_gaussian",
                    perturbation_relative_norm=0.25,
                    excursion_config=cfg,
                    policy_config=pol_cfg,
                    max_output_tokens=2,
                )
                if res["is_correct"]:
                    correct += 1
                curvs.append(float(res["curvature_frob"]))
            rows.append(
                {
                    "visit_step": visit_step,
                    "token_scope": scope,
                    "accuracy": round(correct / len(probs), 4),
                    "mean_curvature_frob": round(float(sum(curvs) / len(curvs)), 4),
                    "num_problems": len(probs),
                }
            )
    (out_dir / "dimension_ablations.json").write_text(
        json.dumps({"ablations": rows}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    t0 = time.perf_counter()
    model_id = "ByteDance/Ouro-1.4B-Thinking"
    model_rev = PINNED_OURO_THINKING_REV
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_rev, local_files_only=True)
    adapter = OuroRecurrentAdapter.from_pretrained(
        repo_id=model_id,
        revision=model_rev,
        dtype=torch.bfloat16,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        local_files_only=True,
    )
    override = (adapter, tokenizer, model_id, model_rev)
    shared_cache: Dict[str, Dict[str, Any]] = {}

    for cfg_name in (
        "phase1_smoke.yaml",
        "phase2_recovery.yaml",
        "phase3_ablations.yaml",
        "phase4_adaptive.yaml",
    ):
        t_phase = time.perf_counter()
        summary = run_experiment_from_config(
            config_path=REPO_ROOT / "configs" / cfg_name,
            repo_root=REPO_ROOT,
            adapter_override=override,
            shared_eval_cache=shared_cache,
        )
        print(
            f"[DONE] {cfg_name} ({time.perf_counter() - t_phase:.1f}s) -> "
            + ", ".join(
                f"{m}: acc={v['accuracy']:.3f} (rescue={v['rescue_rate']:.3f}, deg={v['degradation_rate']:.3f})"
                for m, v in summary["rescue_matrix"]["methods"].items()
            ),
            flush=True,
        )

    t_grid = time.perf_counter()
    run_phase1_grid_with_adapter(adapter, tokenizer)
    print(f"[DONE] phase1_grid ({time.perf_counter() - t_grid:.1f}s)", flush=True)

    t_dim = time.perf_counter()
    run_dimension_ablations(adapter, tokenizer)
    print(f"[DONE] dimension_ablations ({time.perf_counter() - t_dim:.1f}s)", flush=True)

    class _Args:
        runs = "results/"
        output = "reports/"

    cmd_analyze(_Args())
    print(f"All phases completed in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
