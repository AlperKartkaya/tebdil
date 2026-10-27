"""Unit tests for ToyRecurrentAdapter stepwise unrolling and call tracking."""

import torch
import pytest

from tebdil.models.toy_adapter import ToyLoopedModel, ToyRecurrentAdapter


def test_toy_adapter_matches_model_forward():
    model = ToyLoopedModel(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, total_steps=4, seed=123)
    adapter = ToyRecurrentAdapter(model=model)
    input_ids = torch.tensor([[2, 7, 15, 31, 9]], dtype=torch.long)

    with torch.no_grad():
        expected_logits = model(input_ids, num_steps=4)

    adapter.reset_call_counter()
    state = adapter.initialize(input_ids)
    assert state.recurrent_step == 0
    assert state.hidden.shape == (1, 5, 32)

    for step_idx in range(4):
        state = adapter.transition(state)
        assert state.recurrent_step == step_idx + 1
        assert state.context.step == step_idx + 1

    assert adapter.recurrent_block_calls == 4
    actual_logits = adapter.readout_logits(state)
    assert actual_logits.shape == expected_logits.shape
    max_diff = float((actual_logits - expected_logits).abs().max().item())
    assert max_diff <= 1e-6


def test_toy_adapter_compatibility_assertions():
    adapter = ToyRecurrentAdapter(seed=99)
    s1 = adapter.initialize(torch.tensor([[1, 2, 3]]))
    s2 = adapter.initialize(torch.tensor([[1, 2, 3]]))
    adapter.validate_compatibility(s1, s2)

    s2_next = adapter.transition(s2)
    with pytest.raises(ValueError, match="Recurrent step mismatch"):
        adapter.validate_compatibility(s1, s2_next)

    s_other_tok = adapter.initialize(torch.tensor([[1, 2, 4]]))
    with pytest.raises(ValueError, match="Token IDs hash mismatch"):
        adapter.validate_compatibility(s1, s_other_tok)
