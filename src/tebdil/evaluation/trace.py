"""Reproducible run manifests and JSONL event trace persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, Iterable, List
import psutil
import yaml


@dataclass
class RunManifest:
    run_id: str
    created_at_utc: str
    git_commit: str
    model_id: str
    model_revision: str
    software_lock_hash: str
    hardware: str
    seed: int
    task_family: str
    split: str
    decode_policy: str
    recurrent_steps: int
    branch_count: int
    intervention: str
    intervention_step: int
    alpha: float
    beta: float
    norm_clip: float
    cache_policy: str
    precision: str
    config_path: str
    output_dir: str
    tokenizer_revision: str = "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"
    controller_checkpoint: str = "none"
    recurrent_forward_calls: int = 0
    controller_calls: int = 0
    runtime_seconds: float = 0.0
    result_artifacts: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.result_artifacts is None:
            self.result_artifacts = ["manifest.yaml", "traces.jsonl", "summary.json"]


def get_git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "uncommitted"
    except Exception:
        return "uncommitted"


def get_software_lock_hash(repo_root: Path) -> str:
    lock_file = repo_root / "uv.lock"
    if lock_file.exists():
        return hashlib.sha256(lock_file.read_bytes()).hexdigest()[:16]
    pyproj = repo_root / "pyproject.toml"
    if pyproj.exists():
        return hashlib.sha256(pyproj.read_bytes()).hexdigest()[:16]
    return "unknown"


def get_hardware_summary() -> str:
    mem_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    proc = platform.processor() or platform.machine()
    return f"{platform.system()}-{platform.machine()}-{proc}-{mem_gb}GB"


def create_run_manifest(
    run_id: str,
    repo_root: Path,
    model_id: str,
    model_revision: str,
    seed: int,
    task_family: str,
    split: str,
    decode_policy: str,
    recurrent_steps: int,
    branch_count: int,
    intervention: str,
    intervention_step: int,
    alpha: float,
    beta: float,
    norm_clip: float,
    cache_policy: str,
    precision: str,
    config_path: str,
    output_dir: str,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(repo_root),
        model_id=model_id,
        model_revision=model_revision,
        software_lock_hash=get_software_lock_hash(repo_root),
        hardware=get_hardware_summary(),
        seed=seed,
        task_family=task_family,
        split=split,
        decode_policy=decode_policy,
        recurrent_steps=recurrent_steps,
        branch_count=branch_count,
        intervention=intervention,
        intervention_step=intervention_step,
        alpha=alpha,
        beta=beta,
        norm_clip=norm_clip,
        cache_policy=cache_policy,
        precision=precision,
        config_path=config_path,
        output_dir=output_dir,
    )


def save_run_manifest(manifest: RunManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(manifest), f, sort_keys=False)


def append_trace_record(record: Dict[str, Any], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_trace_records(records: Iterable[Dict[str, Any]], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_trace_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
