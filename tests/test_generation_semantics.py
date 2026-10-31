"""Synthetic generation fixtures: prefix identity, refork lifetime and stop rules."""

import torch
import pytest

from tebdil.core import scheduler
from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.evaluation.runner import ByteLevelToyTokenizer, _decode_method_on_problem
from tebdil.models.ouro_adapter import OuroRecurrentAdapter
from tebdil.tasks import TaskProblem


def decode(adapter, tokenizer, **kwargs):
    problem = TaskProblem("fixture", "arithmetic", "pilot", "fixture", "test", "Answer: ", "7")
    return _decode_method_on_problem(
        adapter, tokenizer, problem, "E1", 42, 4, 1, 2,
        "normalized_gaussian", 0.25, ExcursionConfig(), AdaptivePolicyConfig(),
        max_output_tokens=3, **kwargs,
    )


def constant_readout(adapter, token_id):
    def readout(state):
        logits = torch.zeros((*state.hidden.shape[:2], 128))
        logits[..., token_id] = 10.0
        return logits
    adapter.readout_logits = readout


def test_each_generated_token_has_its_own_full_prefix_and_forks(monkeypatch):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    constant_readout(adapter, ord("7"))
    adapter.estimate_step_flops = lambda seq_len, batch_size: seq_len
    original = scheduler.fork_branches
    fork_inputs = []

    def record_fork(base_state, *args, **kwargs):
        fork_inputs.append((base_state.recurrent_step, base_state.hidden.shape[1]))
        return original(base_state, *args, **kwargs)

    monkeypatch.setattr(scheduler, "fork_branches", record_fork)
    record = decode(adapter, ByteLevelToyTokenizer())
    prompt = record["prompt_token_ids"]
    assert record["generated_token_ids"] == [ord("7")] * 3
    assert record["trajectory_semantics"] == "full_prefix_refork_per_generated_token"
    assert record["stop_reason"] == "max_output_tokens"
    assert fork_inputs == [(1, len(prompt) + k) for k in range(3)]
    for k, entry in enumerate(record["token_ledger"]):
        expected = torch.tensor([prompt + [ord("7")] * k])
        assert entry["prefix_token_ids_sha256"] == hash_tensor(expected, length=64)
        assert entry["prefix_length"] == len(prompt) + k
        assert entry["branch_seed"] == 42 + k * 131
    assert record["recurrent_block_calls"] == sum(e["recurrent_block_calls"] for e in record["token_ledger"])
    assert record["estimated_flops"] == sum(e["recurrent_block_calls"] * e["prefix_length"] for e in record["token_ledger"])


def test_eos_stops_even_when_it_decodes_to_empty_text():
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    constant_readout(adapter, 3)

    class Tokenizer(ByteLevelToyTokenizer):
        eos_token_id = 3

    record = decode(adapter, Tokenizer())
    assert len(record["generated_token_ids"]) == 1
    assert record["stop_reason"] == "eos_token"


def test_chat_template_is_explicit_and_does_not_stop_at_a_reasoning_newline():
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    constant_readout(adapter, ord("\n"))

    class Tokenizer(ByteLevelToyTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_tensors):
            assert messages[0]["role"] == "user"
            assert tokenize and add_generation_prompt and not enable_thinking
            return torch.tensor([[1, 2, 4]])

    record = decode(adapter, Tokenizer(), prompt_format="chat")
    assert record["prompt_format"] == "chat"
    assert record["prompt_token_ids"] == [1, 2, 4]
    assert len(record["generated_token_ids"]) == 3
    assert record["stop_reason"] == "max_output_tokens"
    with pytest.raises(ValueError, match="prompt_format"):
        decode(adapter, Tokenizer(), prompt_format="invented")


@pytest.mark.parametrize("gold", ["777", "888"])
def test_teacher_forced_full_answer_is_not_clipped_by_generation_budget(gold):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    constant_readout(adapter, ord("7"))
    problem = TaskProblem("fixture", "arithmetic", "pilot", "fixture", "test", "Answer: ", gold)
    record = _decode_method_on_problem(
        adapter, ByteLevelToyTokenizer(), problem, "S0", 42, 4, 1, 1,
        "none", 0.0, ExcursionConfig(), AdaptivePolicyConfig(), max_output_tokens=1,
    )
    assert record["generated_token_ids"] == [ord("7")]
    assert len(record["gold_token_logprobs"]) == 3
    assert record["recurrent_block_calls"] == 4
    assert record["scoring_recurrent_calls"] == 8
    assert adapter.recurrent_block_calls == 12
    assert record["sequence_gold_logprob"] == pytest.approx(sum(record["gold_token_logprobs"]), abs=2e-6)
