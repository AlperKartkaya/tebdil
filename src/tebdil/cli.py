"""Command-line interface for TEBDIL model inspection, adapter validation, experiments, and analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List
import psutil
import torch
import yaml

from tebdil.analysis import (
    analyze_curvature,
    analyze_step_progression,
    compute_pareto_frontier,
    compute_rescue_matrix,
)
from tebdil.evaluation.metrics import compare_logits
from tebdil.evaluation.runner import (
    load_adapter_and_tokenizer,
    run_experiment_from_config,
)
from tebdil.evaluation.trace import read_trace_records
from tebdil.models.ouro_adapter import PINNED_OURO_THINKING_REV
from tebdil.tasks import generate_task_suite


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_inspect_model(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    vendor_dir = repo_root / "vendor" / "ouro_1_4b_thinking"
    cfg_file = vendor_dir / "config.json"

    hw_info = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    model_info: Dict[str, Any] = {
        "model_id": args.model,
        "pinned_revision": PINNED_OURO_THINKING_REV,
        "vendor_config_present": cfg_file.exists(),
    }
    if cfg_file.exists():
        cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
        model_info.update(
            {
                "architectures": cfg_data.get("architectures"),
                "hidden_size": cfg_data.get("hidden_size"),
                "intermediate_size": cfg_data.get("intermediate_size"),
                "num_hidden_layers": cfg_data.get("num_hidden_layers"),
                "num_attention_heads": cfg_data.get("num_attention_heads"),
                "total_ut_steps": cfg_data.get("total_ut_steps"),
                "early_exit_threshold": cfg_data.get("early_exit_threshold"),
                "vocab_size": cfg_data.get("vocab_size"),
                "torch_dtype": cfg_data.get("torch_dtype"),
            }
        )

    report = {"hardware": hw_info, "model": model_info}
    print(json.dumps(report, indent=2))
    return 0


def cmd_validate_adapter(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_cfg = raw_cfg.get("model", {})

    adapter, tokenizer, model_id, model_rev = load_adapter_and_tokenizer(model_cfg)
    problems = generate_task_suite(
        family=str(raw_cfg.get("evaluation", {}).get("task_family", "arithmetic")),
        split="pilot",
        num_problems=min(5, int(raw_cfg.get("evaluation", {}).get("num_problems", 5))),
        seed=int(raw_cfg.get("execution", {}).get("seed", 42)),
    )

    device = getattr(adapter, "device", torch.device("cpu"))
    results = []
    all_passed = True

    for prob in problems:
        input_ids = tokenizer.encode(prob.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(adapter.model, "early_exit_threshold"):
                native_kwargs = {"input_ids": input_ids, "use_cache": False}
                if adapter.readout_mode == "last_step":
                    native_kwargs["exit_at_step"] = adapter.total_steps - 1
                ref_out = adapter.model(**native_kwargs)
                ref_logits = ref_out.logits
            else:
                ref_logits = adapter.model(input_ids, num_steps=adapter.total_steps)

        st = adapter.initialize(input_ids)
        for _ in range(adapter.total_steps):
            st = adapter.transition(st)
        adap_logits = adapter.readout_logits(st)

        cmp = compare_logits(ref_logits, adap_logits)
        passed = (cmp.max_abs_diff <= 1e-4) and cmp.top1_match
        if not passed:
            all_passed = False
        results.append(
            {
                "problem_id": prob.problem_id,
                "seq_len": int(input_ids.shape[1]),
                "passed": passed,
                **asdict(cmp),
            }
        )

    payload = {
        "model_id": model_id,
        "model_revision": model_rev,
        "device": str(device),
        "total_ut_steps": adapter.total_steps,
        "readout_mode": getattr(adapter, "readout_mode", "last_step"),
        "all_passed": all_passed,
        "checks": results,
    }
    print(json.dumps(payload, indent=2))
    return 0 if all_passed else 1


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path

    summary = run_experiment_from_config(config_path=cfg_path, repo_root=repo_root)
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "output_dir": summary["manifest"]["output_dir"],
                "num_total_records": summary["num_total_records"],
                "methods": {
                    m: {
                        "accuracy": round(v["accuracy"], 4),
                        "paired_accuracy_diff": round(v["paired_accuracy_diff"], 4),
                        "rescue_rate": round(v["rescue_rate"], 4),
                        "degradation_rate": round(v["degradation_rate"], 4),
                        "mean_recurrent_calls": round(v["mean_recurrent_calls"], 2),
                    }
                    for m, v in summary["rescue_matrix"]["methods"].items()
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    runs_dir = Path(args.runs)
    if not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(runs_dir.glob("*/traces.jsonl"))
    if runs_dir.is_file() and runs_dir.name.endswith(".jsonl"):
        trace_files = [runs_dir]

    aggregated_runs: Dict[str, Any] = {}
    all_records: List[Dict[str, Any]] = []

    for tf in trace_files:
        run_name = tf.parent.name
        recs = read_trace_records(tf)
        if not recs:
            continue
        all_records.extend(recs)
        rm = compute_rescue_matrix(recs, baseline_method="S0")
        curv = analyze_curvature(recs)
        pareto = compute_pareto_frontier(rm["methods"])
        step_prog = analyze_step_progression(recs)
        aggregated_runs[run_name] = {
            "num_records": len(recs),
            "rescue_matrix": rm,
            "curvature": curv,
            "pareto_frontier": pareto,
            "step_progression": step_prog,
        }

    overall_rescue = compute_rescue_matrix(all_records, baseline_method="S0") if all_records else {}
    overall_curv = analyze_curvature(all_records) if all_records else {}
    overall_pareto = (
        compute_pareto_frontier(overall_rescue.get("methods", {}))
        if overall_rescue
        else {}
    )

    final_json = {
        "total_runs_analyzed": len(aggregated_runs),
        "total_records_analyzed": len(all_records),
        "runs": aggregated_runs,
        "overall_rescue_matrix": overall_rescue,
        "overall_curvature": overall_curv,
        "overall_pareto_frontier": overall_pareto,
    }
    json_path = out_dir / "final_results.json"
    json_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote analysis summary to {json_path} ({len(aggregated_runs)} runs, {len(all_records)} records).")
    return 0


def cmd_serve_site(args: argparse.Namespace) -> int:
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    site_dir = _repo_root() / "site"
    handler = partial(SimpleHTTPRequestHandler, directory=str(site_dir))
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"Serving TEBDIL landing page at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from tebdil.audit import audit_run

    report = audit_run(
        run_id=args.run_id,
        repo_root=_repo_root(),
        replay_problem_id=args.replay_problem_id,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tebdil", description="TEBDIL CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_insp = sub.add_parser("inspect-model", help="Inspect hardware and Ouro checkpoint config")
    p_insp.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    p_insp.set_defaults(func=cmd_inspect_model)

    p_val = sub.add_parser("validate-adapter", help="Validate exact equivalence of stepwise adapter")
    p_val.add_argument("--config", required=True)
    p_val.set_defaults(func=cmd_validate_adapter)

    p_run = sub.add_parser("run", help="Run experiment from YAML configuration")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=cmd_run)

    p_ana = sub.add_parser("analyze", help="Analyze completed runs and generate paired statistics")
    p_ana.add_argument("--runs", default="results/")
    p_ana.add_argument("--output", default="reports/")
    p_ana.set_defaults(func=cmd_analyze)

    p_aud = sub.add_parser("audit", help="Audit run artifacts and optionally replay on real model")
    p_aud.add_argument("--run-id", required=True)
    p_aud.add_argument("--replay-problem-id", default=None)
    p_aud.set_defaults(func=cmd_audit)

    p_site = sub.add_parser("serve-site", help="Serve the interactive academic landing page UI")
    p_site.add_argument("--host", default="127.0.0.1")
    p_site.add_argument("--port", type=int, default=8080)
    p_site.set_defaults(func=cmd_serve_site)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
