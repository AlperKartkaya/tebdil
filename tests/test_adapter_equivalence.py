"""Exact forward equivalence tests for OuroRecurrentAdapter."""

import torch
import pytest

from tebdil.evaluation.metrics import compare_logits
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


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
