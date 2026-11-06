"""Fast KV-cached TEBDIL runner optimized for Google Colab NVIDIA L4 / A100 GPUs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoTokenizer

from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.scheduler import generate_tebdil_cached
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.reasoning import (
    encode_reasoning_prompt,
    evaluate_reasoning_integer,
    split_prompt_started_reasoning,
)
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks.literature_reasoning import evaluate_literature_reasoning
from tebdil.tasks.reasoning import evaluate_reasoning_task

ROOT = Path(__file__).resolve().parents[1]


def run_stall_recovery_suite(
    adapter: OuroRecurrentAdapter,
    tokenizer,
    methods: list[str],
    exc_cfg: ExcursionConfig,
    pol_cfg: AdaptivePolicyConfig,
    continuation_tokens: int = 32,
    seed: int = 42,
) -> list[dict]:
    passive_dir = ROOT / "reports/passive_reasoning_v1"
    report = json.loads((passive_dir / "report.json").read_text(encoding="utf-8"))
    raw_by_id = {
        r["problem_id"]: json.loads((passive_dir / r["raw_file"]).read_text(encoding="utf-8"))["record"]
        for r in report["records"]
    }

    checkpoints = [
        ("reasoning_v1_development_cnf_0_0", 1637, "clause_9_implication_translation"),
        ("reasoning_v1_development_cnf_0_0", 2005, "variable_assignment_synthesis"),
        ("reasoning_v1_development_key_planning_0_0", 255, "key_requirement_enumeration"),
        ("reasoning_v1_development_key_planning_0_0", 382, "room_3_outgoing_edge_check"),
        ("reasoning_v1_development_key_planning_0_0", 1023, "room_1_dead_end_recovery"),
    ]

    suite_rows = []
    for prob_id, tok_idx, label in checkpoints:
        rec = raw_by_id[prob_id]
        prefix_list = rec["prompt_token_ids"] + rec["generated_token_ids"][:tok_idx]
        prefix_ids = torch.tensor([prefix_list], dtype=torch.long, device=adapter.device)
        prefix_tail = tokenizer.decode(prefix_list[-25:], skip_special_tokens=False)
        print(f"  [Stall Suite] {prob_id} @ k={tok_idx} ({label}) | L={len(prefix_list)}", flush=True)

        method_outputs: dict[str, dict] = {}
        for m in methods:
            gen = generate_tebdil_cached(
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_ids=prefix_ids,
                method=m,
                max_new_tokens=continuation_tokens,
                recurrent_steps=4,
                fork_after_step=1,
                branch_count=2,
                perturbation_method="competitive_logit",
                perturbation_relative_norm=0.25,
                excursion_config=exc_cfg,
                policy_config=pol_cfg,
                seed=seed,
            )
            method_outputs[m] = gen
            print(
                f"    {m:4s} -> exc={gen['excursion_count']:2d} | calls={gen['recurrent_block_calls']:3d} | "
                f"{gen['wall_seconds']:.2f}s | {gen['decoded_text']!r}",
                flush=True,
            )

        s0_toks = method_outputs["S0"]["generated_token_ids"] if "S0" in method_outputs else []
        m1_toks = method_outputs["M1"]["generated_token_ids"] if "M1" in method_outputs else []
        for m, gen in method_outputs.items():
            toks = gen["generated_token_ids"]
            diff_vs_s0 = sum(1 for a, b in zip(toks, s0_toks) if a != b) if s0_toks else 0
            diff_vs_m1 = sum(1 for a, b in zip(toks, m1_toks) if a != b) if m1_toks else 0
            first_diff_s0 = next(
                (i for i, (a, b) in enumerate(zip(toks, s0_toks)) if a != b), None
            )
            gen["changed_tokens_vs_s0"] = diff_vs_s0
            gen["changed_tokens_vs_m1"] = diff_vs_m1
            gen["first_diff_token_offset_vs_s0"] = first_diff_s0

        suite_rows.append(
            {
                "problem_id": prob_id,
                "checkpoint_token_index": tok_idx,
                "checkpoint_label": label,
                "prefix_length": len(prefix_list),
                "prefix_tail": prefix_tail,
                "continuation_tokens": continuation_tokens,
                "methods": method_outputs,
            }
        )
    return suite_rows


def run_completion_reasoning_suite(
    adapter: OuroRecurrentAdapter,
    tokenizer,
    methods: list[str],
    exc_cfg: ExcursionConfig,
    pol_cfg: AdaptivePolicyConfig,
    max_tasks: int = 3,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> list[dict]:
    comp_data = json.loads((ROOT / "configs/completion_v1_tasks.json").read_text(encoding="utf-8"))
    tasks = comp_data["partitions"]["development"][:max_tasks]
    records = []

    comp_exc_cfg = replace(exc_cfg, excursion_window_tokens=1)
    for task in tasks:
        prompt_ids = encode_reasoning_prompt(tokenizer, task["prompt"]).to(adapter.device)
        for m in methods:
            print(f"  [Completion Suite] {task['problem_id']} ({task['template_id']}) | method={m} ...", flush=True)
            gen = generate_tebdil_cached(
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                method=m,
                max_new_tokens=max_new_tokens,
                recurrent_steps=4,
                fork_after_step=1,
                branch_count=2,
                perturbation_method="competitive_logit",
                perturbation_relative_norm=0.25,
                excursion_config=comp_exc_cfg,
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
                "difficulty": task["difficulty"],
                "gold_answer": task["gold_answer"],
                "generated_tokens": len(gen["generated_token_ids"]),
                **gen,
                **split_info,
            }
            eval_res = evaluate_reasoning_integer(row, task["gold_answer"])
            if (
                eval_res["status"] == "invalid_format"
                and row["completed"]
                and row["reasoning_status"] == "closed"
                and isinstance(row["final_text"], str)
            ):
                m_eq = re.fullmatch(r"\s*[a-zA-Z]\s*=\s*([+-]?[0-9]+)\s*\.?\s*", row["final_text"])
                if m_eq:
                    ans_val = m_eq.group(1)
                    is_ok = ans_val == str(task["gold_answer"])
                    eval_res = {
                        "answer_extraction": "prompt_started_reasoning_equation_v1",
                        "extracted_answer": ans_val,
                        "format_valid": True,
                        "status": "correct" if is_ok else "incorrect",
                        "is_correct": is_ok,
                    }
            row["evaluation"] = eval_res
            records.append(row)
            print(
                f"    -> tokens={len(gen['generated_token_ids'])} | status={eval_res['status']} | "
                f"ans={eval_res['extracted_answer']} (gold={task['gold_answer']}) | "
                f"exc={gen['excursion_count']} | calls={gen['recurrent_block_calls']} | "
                f"time={gen['wall_seconds']}s",
                flush=True,
            )
    return records


def run_literature_reasoning_suite(
    adapter: OuroRecurrentAdapter,
    tokenizer,
    methods: list[str],
    exc_cfg: ExcursionConfig,
    pol_cfg: AdaptivePolicyConfig,
    partition: str = "development",
    max_tasks: int = 12,
    task_ids: list[str] | None = None,
    max_new_tokens: int = 640,
    seed: int = 42,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    lit_data = json.loads((ROOT / "configs/literature_reasoning_v1_tasks.json").read_text(encoding="utf-8"))
    all_tasks = lit_data["partitions"][partition]
    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in all_tasks if t["problem_id"] in id_set]
    else:
        tasks = all_tasks[:max_tasks]

    records = []
    lit_exc_cfg = replace(exc_cfg, excursion_window_tokens=1, beta=0.65)
    lit_pol_cfg = replace(
        pol_cfg,
        min_entropy_trigger=1.20,
        max_margin_trigger=0.80,
        stagnation_ratio_threshold=0.35,
    )
    for task in tasks:
        prompt_ids = encode_reasoning_prompt(tokenizer, task["prompt"]).to(adapter.device)
        for m in methods:
            print(
                f"  [Literature Suite] {task['problem_id']} ({task['family']}) | method={m} ...",
                flush=True,
            )
            gen = generate_tebdil_cached(
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                method=m,
                max_new_tokens=max_new_tokens,
                recurrent_steps=4,
                fork_after_step=1,
                branch_count=2,
                perturbation_method="competitive_logit",
                perturbation_relative_norm=0.14,
                excursion_config=lit_exc_cfg,
                policy_config=lit_pol_cfg,
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
            records.append(row)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps({"partial_literature_records": records}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            print(
                f"    -> tokens={len(gen['generated_token_ids'])} | status={eval_res['status']} | "
                f"ans={eval_res['extracted_answer']} (gold={task['gold_answer']}, trap={eval_res['hit_trap_answer']}) | "
                f"exc={gen['excursion_count']} | calls={gen['recurrent_block_calls']} | "
                f"time={gen['wall_seconds']}s",
                flush=True,
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/colab_l4_tebdil_run.json")
    parser.add_argument(
        "--suite",
        choices=("all", "stall", "completion", "reasoning_v1", "literature"),
        default="all",
    )
    parser.add_argument("--partition", choices=("development", "reserved_test"), default="development")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=32)
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
        cache_mode="branched_kv",
    )

    exc_cfg = ExcursionConfig(
        alpha=0.75,
        beta=0.80,
        max_relative_norm=0.35,
        visit_step=2,
        token_scope="last_token",
        donor_strategy="competitive_logit",
        update_space="pre_norm",
        excursion_window_tokens=4,
    )
    pol_cfg = AdaptivePolicyConfig(
        min_relative_divergence=0.002,
        max_relative_divergence=0.85,
        min_entropy_trigger=0.95,
        max_margin_trigger=1.20,
        stagnation_ratio_threshold=0.45,
        donor_selection="confidence_gain",
    )

    t_run_start = time.perf_counter()
    stall_records = []
    completion_records = []
    reasoning_v1_records = []
    literature_records = []

    if args.suite in ("all", "stall"):
        stall_records = run_stall_recovery_suite(
            adapter=adapter,
            tokenizer=tokenizer,
            methods=args.methods,
            exc_cfg=exc_cfg,
            pol_cfg=pol_cfg,
            continuation_tokens=args.continuation_tokens,
            seed=args.seed,
        )

    if args.suite in ("all", "completion"):
        completion_records = run_completion_reasoning_suite(
            adapter=adapter,
            tokenizer=tokenizer,
            methods=args.methods,
            exc_cfg=exc_cfg,
            pol_cfg=pol_cfg,
            max_tasks=args.max_tasks,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )

    if args.suite in ("all", "literature"):
        literature_records = run_literature_reasoning_suite(
            adapter=adapter,
            tokenizer=tokenizer,
            methods=args.methods,
            exc_cfg=exc_cfg,
            pol_cfg=pol_cfg,
            partition=args.partition,
            max_tasks=args.max_tasks,
            task_ids=args.task_ids,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            checkpoint_path=args.output,
        )

    if args.suite == "reasoning_v1":
        tasks_data = json.loads((ROOT / "configs/reasoning_v1_tasks.json").read_text(encoding="utf-8"))
        tasks = tasks_data["partitions"][args.partition][: args.max_tasks]
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
                    perturbation_relative_norm=0.25,
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
                    "level": task.get("structural_level", task.get("level")),
                    **gen,
                    **split_info,
                }
                eval_res = evaluate_reasoning_task(task, row)
                row["evaluation"] = eval_res
                reasoning_v1_records.append(row)
                print(
                    f"    -> tokens={len(gen['generated_token_ids'])} | "
                    f"status={eval_res['status']} | excursions={gen['excursion_count']} | "
                    f"calls={gen['recurrent_block_calls']} | time={gen['wall_seconds']}s",
                    flush=True,
                )

    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu_name": gpu_name,
        "dtype": str(dtype),
        "attn_implementation": args.attn,
        "suite": args.suite,
        "excursion_config": asdict(exc_cfg),
        "policy_config": asdict(pol_cfg),
        "total_wall_seconds": round(time.perf_counter() - t_run_start, 3),
        "stall_recovery_records": stall_records,
        "completion_reasoning_records": completion_records,
        "reasoning_v1_records": reasoning_v1_records,
        "literature_reasoning_records": literature_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[TEBDIL] Saved results to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
