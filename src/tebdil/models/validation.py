"""Unintervened, fixed-prefix Ouro parity checks; not a reasoning benchmark."""

from __future__ import annotations

import torch
from torch import Tensor

from tebdil.core.state import hash_tensor
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


@torch.no_grad()
def validate_ouro_execution(
    adapter: OuroRecurrentAdapter,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> dict:
    """Compare every depth with native forward; restore configuration even on failure.

    Native gate is checked at the configured threshold and at 0.5 so a checkpoint
    whose default selects the last state also exercises earlier-state selection.
    All comparisons are exact within one device/dtype, including padded positions.
    """
    model = adapter.model
    base = model.model
    ids = input_ids.to(adapter.device)
    kwargs = {"input_ids": ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(adapter.device)
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to(adapter.device)
    total_steps = adapter.total_steps
    original_mode = adapter.readout_mode
    original_threshold = model.early_exit_threshold
    calls_before = adapter.recurrent_block_calls
    _, native_hidden, native_gates = base(**kwargs)
    state = adapter.initialize(ids, attention_mask, position_ids)
    steps = []

    def difference(a: Tensor, b: Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    try:
        for depth in range(1, total_steps + 1):
            state = adapter.transition(state)
            base.total_ut_steps = depth
            adapter.readout_mode = "last_step"
            forced_native = model(**kwargs, exit_at_step=depth - 1).logits
            forced_adapter = adapter.readout_logits(state)
            entry = {
                "depth": depth,
                "hidden_sha256": hash_tensor(state.hidden, length=64),
                "hidden_max_abs_diff": difference(native_hidden[depth - 1], state.hidden),
                "gate_max_abs_diff": difference(native_gates[depth - 1], state.gate_history[-1]),
                "forced_logits_max_abs_diff": difference(forced_native, forced_adapter),
                "native_gate_checks": [],
            }
            adapter.readout_mode = "native_gate"
            for threshold in dict.fromkeys([original_threshold, 0.5]):
                model.early_exit_threshold = threshold
                native = model(**kwargs).logits
                actual = adapter.readout_logits(state)
                entry["native_gate_checks"].append({
                    "threshold": threshold,
                    "max_abs_diff": difference(native, actual),
                    "top1_match": bool(torch.equal(native.argmax(-1), actual.argmax(-1))),
                })
            entry["passed"] = (
                entry["hidden_max_abs_diff"] == 0.0
                and entry["gate_max_abs_diff"] == 0.0
                and entry["forced_logits_max_abs_diff"] == 0.0
                and all(c["max_abs_diff"] == 0.0 and c["top1_match"] for c in entry["native_gate_checks"])
            )
            steps.append(entry)
    finally:
        base.total_ut_steps = total_steps
        adapter.readout_mode = original_mode
        model.early_exit_threshold = original_threshold
    return {
        "all_passed": all(s["passed"] for s in steps),
        "dtype": str(adapter.dtype),
        "device": str(adapter.device),
        "input_ids_sha256": hash_tensor(ids, length=64),
        "mask_hash": state.context.mask_hash,
        "positions_hash": state.context.positions_hash,
        "batch_size": int(ids.shape[0]),
        "sequence_length": int(ids.shape[1]),
        "native_forced_step": model.early_exit_step,
        "adapter_recurrent_calls": adapter.recurrent_block_calls - calls_before,
        "steps": steps,
    }
