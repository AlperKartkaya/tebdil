"""Exact forward equivalence tests for OuroRecurrentAdapter."""

import torch
import pytest

from tebdil.evaluation.metrics import compare_logits
from tebdil.models.ouro_adapter import OuroRecurrentAdapter
from tebdil.models.validation import validate_ouro_execution


@pytest.mark.parametrize("readout_mode", ["last_step", "native_gate"])
def test_ouro_mini_adapter_exact_equivalence(readout_mode: str):
    adapter = OuroRecurrentAdapter.from_mini_config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        total_ut_steps=4,
        dtype=torch.float32,
        device="cpu",
        seed=42,
        readout_mode=readout_mode,
    )
    input_ids = torch.tensor([[1, 14, 29, 55, 8, 99]], dtype=torch.long)

    with torch.no_grad():
        if readout_mode == "last_step":
            ref_out = adapter.model(input_ids=input_ids, use_cache=False, exit_at_step=3)
        else:
            ref_out = adapter.model(input_ids=input_ids, use_cache=False)

    adapter.reset_call_counter()
    state = adapter.initialize(input_ids)
    for _ in range(adapter.total_steps):
        state = adapter.transition(state)

    assert adapter.recurrent_block_calls == 4
    adap_logits = adapter.readout_logits(state)
    cmp = compare_logits(ref_out.logits, adap_logits)
    assert cmp.max_abs_diff <= 1e-6
    assert cmp.top1_match is True
    assert cmp.topk_overlap == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["short", "masked_batch", "offset_positions"])
def test_per_depth_execution_contract(dtype, case):
    adapter = OuroRecurrentAdapter.from_mini_config(dtype=dtype)
    ids = torch.tensor([[1, 4, 9, 12, 15, 20]])
    mask = positions = None
    if case == "short":
        ids = ids[:, :1]
    elif case == "masked_batch":
        ids = ids.expand(2, -1).clone()
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    else:
        positions = torch.arange(7, 13).unsqueeze(0)
    report = validate_ouro_execution(adapter, ids, mask, positions)
    assert report["all_passed"], report
    assert len(report["steps"]) == 4
    assert report["adapter_recurrent_calls"] == 4
    assert adapter.total_steps == 4
    assert adapter.readout_mode == "last_step"


@pytest.mark.parametrize("threshold", [None, 0.5, 1.0])
@pytest.mark.parametrize("forced_step", [None, 0, 2])
def test_native_gate_respects_native_readout_configuration(threshold, forced_step):
    adapter = OuroRecurrentAdapter.from_mini_config(readout_mode="native_gate")
    adapter.model.early_exit_threshold = threshold
    adapter.model.early_exit_step = forced_step
    ids = torch.tensor([[1, 4, 9]])
    with torch.no_grad():
        native = adapter.model(input_ids=ids, use_cache=False).logits
    state = adapter.initialize(ids)
    for _ in range(adapter.total_steps):
        state = adapter.transition(state)
    assert torch.equal(native, adapter.readout_logits(state))


def test_position_context_and_cache_boundaries():
    from dataclasses import replace

    adapter = OuroRecurrentAdapter.from_mini_config()
    ids = torch.tensor([[1, 4, 9]])
    a = adapter.initialize(ids)
    b = adapter.initialize(ids, position_ids=torch.tensor([[7, 8, 9]]))
    assert a.context.positions_hash != b.context.positions_hash
    with pytest.raises(ValueError, match="Positions hash"):
        adapter.validate_compatibility(a, b)
    with pytest.raises(ValueError, match="cache"):
        adapter.transition(replace(a, context=replace(a.context, cache_mode="enabled")))
    with pytest.raises(ValueError, match="step"):
        adapter.transition(replace(a, recurrent_step=1))
    with pytest.raises(ValueError, match="position_ids"):
        adapter.initialize(ids, position_ids=torch.tensor([[0, 1]]))


def test_execution_validation_detects_transition_drift(monkeypatch):
    adapter = OuroRecurrentAdapter.from_mini_config()
    original = adapter.transition

    def drift(state):
        result = original(state)
        result.hidden = result.hidden + 0.01
        return result

    monkeypatch.setattr(adapter, "transition", drift)
    report = validate_ouro_execution(adapter, torch.tensor([[1, 4, 9]]))
    assert not report["all_passed"]
    assert not report["steps"][0]["passed"]
