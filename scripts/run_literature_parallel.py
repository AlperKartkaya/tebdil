"""Parallel multi-worker literature reasoning benchmark runner for Google Colab L4 / A100 GPUs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch

from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.scheduler import generate_tebdil_cached
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.reasoning import (
    encode_reasoning_prompt,
    split_prompt_started_reasoning,
)
from tebdil.models.ouro_adapter import (
    OuroForCausalLM,
    OuroRecurrentAdapter,
    PINNED_OURO_THINKING_REV,
)
from tebdil.tasks.literature_reasoning import evaluate_literature_reasoning

ROOT = Path(__file__).resolve().parents[1]


def _worker_loop(
    worker_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    max_new_tokens: int,
    exc_cfg: ExcursionConfig,
    pol_cfg: AdaptivePolicyConfig,
    seed: int,
    model_id: str = "ByteDance/Ouro-1.4B-Thinking",
    revision: str = PINNED_OURO_THINKING_REV,
    attn_implementation: str = "sdpa",
) -> None:
    try:
        from transformers import AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        t_load0 = time.perf_counter()
        model = OuroForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        model.eval()

        adapter = OuroRecurrentAdapter(model=model, readout_mode="last_step")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        mem_gb = (
            torch.cuda.memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        print(
            f"[Worker {worker_id}] Ready on {device} ({mem_gb:.2f} GB VRAM) in {time.perf_counter() - t_load0:.2f}s",
            flush=True,
        )

        while True:
            job = job_queue.get()
            if job is None:
                break
            task, method = job
            t0 = time.perf_counter()
            print(
                f"[Worker {worker_id}] START {task['problem_id']} ({task['family']}) | method={method} ...",
                flush=True,
            )
            prompt_ids = encode_reasoning_prompt(tokenizer, task["prompt"]).to(device)
            gen = generate_tebdil_cached(
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                method=method,
                max_new_tokens=max_new_tokens,
                recurrent_steps=4,
                fork_after_step=1,
                branch_count=2,
                perturbation_method="competitive_logit",
                perturbation_relative_norm=0.14,
                excursion_config=exc_cfg,
                policy_config=pol_cfg,
                seed=seed,
            )
            split_info = split_prompt_started_reasoning(
                gen["decoded_text"], tokenizer.eos_token, completed=gen["completed"]
            )
            row = {
                "problem_id": task["problem_id"],
                "family": task["family"],
                "template_id": task["template_id"],
                "source_benchmark": task.get("source_benchmark", ""),
                "reasoning_pathology": task.get("reasoning_pathology", ""),
                "difficulty": task["difficulty"],
                "gold_answer": task["gold_answer"],
                "trap_answers": task.get("trap_answers", []),
                "generated_tokens": len(gen["generated_token_ids"]),
                **gen,
                **split_info,
            }
            eval_res = evaluate_literature_reasoning(row, task["gold_answer"])
            eval_res["hit_trap_answer"] = (
                eval_res["extracted_answer"] in set(task.get("trap_answers", []))
                if eval_res["extracted_answer"] is not None
                else False
            )
            row["evaluation"] = eval_res
            row["worker_id"] = worker_id
            wall = time.perf_counter() - t0
            print(
                f"[Worker {worker_id}] FINISH {task['problem_id']} | method={method} -> "
                f"tokens={len(gen['generated_token_ids'])} | status={eval_res['status']} | "
                f"ans={eval_res['extracted_answer']} (gold={task['gold_answer']}, trap={eval_res['hit_trap_answer']}) | "
                f"exc={gen['excursion_count']} | calls={gen['recurrent_block_calls']} | time={wall:.1f}s",
                flush=True,
            )
            result_queue.put(row)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result_queue.put({"error": str(exc), "worker_id": worker_id})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[
            "lit_dev_graph_001",
            "lit_test_crt_000",
            "lit_test_kk_000",
            "lit_test_gsm_000",
            "lit_test_nt_000",
            "lit_test_zebra_000",
            "lit_test_graph_000",
        ],
        help="Specific task IDs to run (searches all partitions).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["S0", "E2"],
        help="Inference methods to run (e.g. S0 E2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent worker processes on the GPU (default: 2).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3000,
        help="Maximum generated tokens budget (default: 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/colab_l4_lit_remaining_3000.json",
        help="Output report JSON file.",
    )
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation.",
    )
    args = parser.parse_args()

    lit_path = ROOT / "configs/literature_reasoning_v1_tasks.json"
    lit_data = json.loads(lit_path.read_text(encoding="utf-8"))
    all_tasks = lit_data["partitions"]["development"] + lit_data["partitions"]["reserved_test"]
    task_lookup = {t["problem_id"]: t for t in all_tasks}

    selected_tasks = []
    for tid in args.task_ids:
        if tid not in task_lookup:
            raise KeyError(f"Task ID '{tid}' not found in literature reasoning dataset!")
        selected_tasks.append(task_lookup[tid])

    jobs = [(t, m) for t in selected_tasks for m in args.methods]
    total_jobs = len(jobs)
    print(
        f"[TEBDIL Parallel Runner] {total_jobs} total jobs ({len(selected_tasks)} tasks x {len(args.methods)} methods) "
        f"across {args.workers} workers | max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    exc_cfg = ExcursionConfig(
        alpha=0.65,
        beta=0.65,
        visit_step=2,
        token_scope="last_token",
        donor_strategy="competitive_logit",
        update_space="pre_norm",
        excursion_window_tokens=1,
    )
    pol_cfg = AdaptivePolicyConfig(
        min_relative_divergence=0.002,
        max_relative_divergence=0.85,
        min_entropy_trigger=1.20,
        max_margin_trigger=0.80,
        stagnation_ratio_threshold=0.35,
        donor_selection="confidence_gain",
    )

    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for job in jobs:
        job_queue.put(job)
    for _ in range(args.workers):
        job_queue.put(None)

    workers = []
    for wid in range(args.workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                wid + 1,
                job_queue,
                result_queue,
                args.max_new_tokens,
                exc_cfg,
                pol_cfg,
                args.seed,
                "ByteDance/Ouro-1.4B-Thinking",
                PINNED_OURO_THINKING_REV,
                args.attn,
            ),
        )
        p.start()
        workers.append(p)

    records: List[Dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for i in range(total_jobs):
        item = result_queue.get()
        if "error" in item:
            print(f"[TEBDIL Parallel Runner] ERROR from worker {item.get('worker_id')}: {item['error']}", flush=True)
            continue
        records.append(item)
        # Checkpoint atomically after every job
        checkpoint_data = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "workers": args.workers,
            "max_new_tokens": args.max_new_tokens,
            "jobs_completed": len(records),
            "jobs_total": total_jobs,
            "partial_literature_records": records,
        }
        args.output.write_text(
            json.dumps(checkpoint_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    for p in workers:
        p.join()

    # Sort records for reproducible reporting
    records.sort(key=lambda r: (r["problem_id"], r["method"]))
    total_wall = time.perf_counter() - t_start

    final_report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "max_new_tokens": args.max_new_tokens,
        "total_jobs": total_jobs,
        "total_wall_seconds": round(total_wall, 2),
        "excursion_config": asdict(exc_cfg),
        "policy_config": asdict(pol_cfg),
        "literature_reasoning_records": records,
    }
    args.output.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[TEBDIL Parallel Runner] ALL {len(records)}/{total_jobs} JOBS COMPLETE in {total_wall:.1f}s!", flush=True)
    print(f"[TEBDIL Parallel Runner] Saved final report to {args.output}\n", flush=True)

    # Print summary table
    print(f"{'Task ID':<20} | {'Method':<6} | {'Tokens':<7} | {'Status':<16} | {'Ans':<10} | {'Gold':<8} | {'Trap':<6} | {'Exc':<4} | {'Calls':<6}")
    print("-" * 96)
    for r in records:
        ev = r.get("evaluation", {})
        print(
            f"{r['problem_id']:<20} | {r['method']:<6} | {r['generated_tokens']:<7} | "
            f"{ev.get('status', 'unknown'):<16} | {str(ev.get('extracted_answer')):<10} | "
            f"{str(r['gold_answer']):<8} | {str(ev.get('hit_trap_answer')):<6} | "
            f"{r['excursion_count']:<4} | {r['recurrent_block_calls']:<6}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
