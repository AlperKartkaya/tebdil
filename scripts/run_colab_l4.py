"""Fast KV-cached TEBDIL runner optimized for Google Colab NVIDIA L4 / A100 GPUs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.scheduler import generate_tebdil_cached
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.reasoning import encode_reasoning_prompt, split_prompt_started_reasoning
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks.reasoning import evaluate_reasoning_task

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/colab_l4_tebdil_run.json")
    parser.add_argument("--partition", choices=("development", "reserved"), default="development")
    parser.add_argument("--max-tasks", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--methods", nargs="+", default=["S0", "M1", "E1", "E2"])
    parser.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa" if torch.cuda.is_available() else "eager")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else device

    print(f"[TEBDIL] Device={device} ({gpu_name}) | dtype={dtype} | attn={args.attn}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "ByteDance/Ouro-1.4B-Thinking",
        revision=PINNED_OURO_THINKING_REV,
    )
    adapter = OuroRecurrentAdapter.from_pretrained(
        dtype=dtype,
        device=device,
        attn_implementation=args.attn,
    )

    tasks_data = json.loads((ROOT / "configs/reasoning_v1_tasks.json").read_text(encoding="utf-8"))
    tasks = tasks_data["partitions"][args.partition][: args.max_tasks]

    exc_cfg = ExcursionConfig(
        alpha=0.65,
        beta=0.65,
        max_relative_norm=0.25,
        visit_step=2,
        token_scope="last_token",
        donor_strategy="competitive_logit",
        update_space="pre_norm",
        excursion_window_tokens=4,
    )
    pol_cfg = AdaptivePolicyConfig(
        min_relative_divergence=0.002,
        max_relative_divergence=0.65,
        min_entropy_trigger=1.10,
        max_margin_trigger=0.85,
        stagnation_ratio_threshold=0.35,
        donor_selection="confidence_gain",
    )

    records = []
    t_run_start = time.perf_counter()
    for task in tasks:
        prompt_ids = encode_reasoning_prompt(tokenizer, task["prompt"]).to(adapter.device)
        for method in args.methods:
            print(f"  Running {task['problem_id']} | method={method} ...", flush=True)
            gen = generate_tebdil_cached(
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                method=method,
                max_new_tokens=args.max_new_tokens,
                recurrent_steps=4,
                fork_after_step=1,
                branch_count=2,
                perturbation_method="competitive_logit",
                perturbation_relative_norm=0.12,
                excursion_config=exc_cfg,
                policy_config=pol_cfg,
                seed=args.seed,
            )
            split_info = split_prompt_started_reasoning(
                gen["decoded_text"], tokenizer.eos_token, completed=gen["completed"]
            )
            row = {
                "problem_id": task["problem_id"],
                "family": task["family"],
                "level": task["level"],
                **gen,
                **split_info,
            }
            eval_res = evaluate_reasoning_task(task, row)
            row["evaluation"] = eval_res
            records.append(row)
            print(
                f"    -> tokens={len(gen['generated_token_ids'])} | "
                f"status={eval_res['outcome']} | excursions={gen['excursion_count']} | "
                f"calls={gen['recurrent_block_calls']} | time={gen['wall_seconds']}s",
                flush=True,
            )

    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu_name": gpu_name,
        "dtype": str(dtype),
        "attn_implementation": args.attn,
        "partition": args.partition,
        "max_new_tokens": args.max_new_tokens,
        "excursion_config": asdict(exc_cfg),
        "policy_config": asdict(pol_cfg),
        "total_wall_seconds": round(time.perf_counter() - t_run_start, 3),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[TEBDIL] Saved results to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
