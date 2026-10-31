"""Verify the pinned checkpoint's fixed-prefix execution contract, without task claims.

uv run python scripts/verify_ouro_execution.py --output site/static/evidence/ouro_execution_contract.json
The output must not already exist; preserve earlier evidence rather than overwrite it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from huggingface_hub import snapshot_download
import torch

from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.models.validation import validate_ouro_execution

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "ByteDance/Ouro-1.4B-Thinking"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execution_cases():
    """Token fixtures, not benchmark questions; include left and right padding."""
    ids = torch.tensor([[1, 14, 29, 55, 8, 99, 20, 4]])
    return [
        ("one_token", ids[:, :1], None, None),
        ("masked_batch", ids.expand(2, -1).clone(),
         torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]]), None),
        ("offset_positions", ids, None, torch.arange(7, 15).unsqueeze(0)),
    ]


@torch.no_grad()
def precision_snapshot(adapter, ids, mask, positions):
    state = adapter.initialize(ids, mask, positions)
    for _ in range(adapter.total_steps):
        state = adapter.transition(state)
    hidden = state.hidden.detach().float().cpu()
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    # Compare readout at each row's last unmasked position, not at padding.
    last = torch.arange(ids.shape[1]).expand_as(ids).masked_fill(~valid, -1).max(-1).values
    selected = state.hidden[torch.arange(ids.shape[0], device=adapter.device), last.to(adapter.device)]
    logits = adapter.model.lm_head(selected).float().cpu()
    return hidden[valid], logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new evidence path")

    started = time.monotonic()
    provenance = json.loads((ROOT / "site/static/evidence/model_provenance.json").read_text())
    snapshot = Path(snapshot_download(MODEL_ID, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    checkpoint_files = {}
    for name, expected in provenance["weight_hashes"].items():
        actual = sha256(snapshot / name)
        if actual != expected["sha256"]:
            raise ValueError(f"Pinned checkpoint provenance mismatch: {name}")
        checkpoint_files[name] = actual

    cases = execution_cases()
    fp32_snapshots = {}
    validations = []
    precision = []
    for dtype in (torch.float32, torch.bfloat16):
        adapter = OuroRecurrentAdapter.from_pretrained(
            dtype=dtype, device=args.device, local_files_only=True,
        )
        for name, ids, mask, positions in cases:
            print(f"Checking {dtype} {name} on {adapter.device}", flush=True)
            result = validate_ouro_execution(adapter, ids, mask, positions)
            result.update({
                "case": name, "input_ids": ids.tolist(),
                "attention_mask": mask.tolist() if mask is not None else None,
                "position_ids": positions.tolist() if positions is not None else None,
            })
            validations.append(result)
            hidden, logits = precision_snapshot(adapter, ids, mask, positions)
            if dtype == torch.float32:
                fp32_snapshots[name] = (hidden, logits)
            else:
                ref_hidden, ref_logits = fp32_snapshots[name]
                precision.append({
                    "case": name,
                    "hidden_max_abs_diff": (hidden - ref_hidden).abs().max().item(),
                    "hidden_mean_abs_diff": (hidden - ref_hidden).abs().mean().item(),
                    "logits_max_abs_diff": (logits - ref_logits).abs().max().item(),
                    "last_valid_token_top1_match_by_row": (logits.argmax(-1) == ref_logits.argmax(-1)).tolist(),
                    "fp32_top1_ids": ref_logits.argmax(-1).tolist(),
                    "bf16_top1_ids": logits.argmax(-1).tolist(),
                })
        del adapter
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    sources = [
        "src/tebdil/models/ouro_adapter.py", "src/tebdil/models/validation.py",
        "src/tebdil/cli.py", "src/tebdil/core/state.py",
        "scripts/verify_ouro_execution.py", "tests/test_adapter_equivalence.py",
        "vendor/ouro_1_4b_thinking/modeling_ouro.py",
        "vendor/ouro_1_4b_thinking/configuration_ouro.py", "uv.lock",
    ]
    report = {
        "schema_version": 1,
        "purpose": "Real-checkpoint execution parity, not reasoning efficacy or benchmark reproduction",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_identity": "Per-file SHA-256 includes changes not yet committed at recording time",
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "model_id": MODEL_ID, "model_revision": PINNED_OURO_THINKING_REV,
        "checkpoint_sha256": checkpoint_files,
        "torch_version": torch.__version__, "platform": platform.platform(),
        "cache_mode": "disabled", "attention_backend": "eager",
        "all_passed": all(v["all_passed"] for v in validations),
        "validations": validations, "cross_precision_diagnostic": precision,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Three short token fixtures, not task prompts or generation-quality measurements",
            "Exact parity is within one dtype/device; FP32/BF16 differences are diagnostic, not required to be zero",
            "No KV-cache execution, long-context or CUDA parity claim unless explicitly recorded",
            "No intervention efficacy, controller utility or historical cost-provenance claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"all_passed={report['all_passed']}; evidence: {args.output}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
