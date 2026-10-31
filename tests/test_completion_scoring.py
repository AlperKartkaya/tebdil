"""Synthetic final-answer fixtures; no gold answer is an extraction input."""

import pytest
import torch

from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.runner import ByteLevelToyTokenizer, _decode_method_on_problem
from tebdil.models.ouro_adapter import OuroRecurrentAdapter
from tebdil.tasks import TaskProblem
from tebdil.tasks.completion import extract_completed_integer


@pytest.mark.parametrize("text,completed,expected", [
    ("<think>123</think>**Answer:** 7<eos>", True, "7"),
    (r"<think>123</think>Final Result: \boxed{0}<eos>", True, "0"),
    (r"<think>123</think>Working: 48, then 50. $\boxed{50}$<eos>", True, "50"),
    ("<think>123</think>Final answer:\n-12<eos>", True, "-12"),
    ("7<eos>", True, "7"),
    ("<think>Answer: 7<eos>", True, None),
    ("<think>123</think>Answer: 7", False, None),
    ("Working: 7, 15, maybe 50<eos>", True, None),
    (r"\boxed{7} or \boxed{8}<eos>", True, None),
    (r"\boxed{7} but not final<eos>", True, None),
    ("Answer: 7 or 8<eos>", True, None),
])
def test_completed_final_answer_not_first_reasoning_integer(text, completed, expected):
    assert extract_completed_integer(text, "<eos>", completed=completed) == expected


@pytest.mark.parametrize("eos_id,correct", [(7, True), (8, False)])
def test_chat_decoder_scores_only_completed_final_answer(eos_id, correct):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)

    def readout(state):
        logits = torch.zeros((*state.hidden.shape[:2], 128))
        logits[..., 7] = 10.0
        return logits

    adapter.readout_logits = readout

    class Tokenizer(ByteLevelToyTokenizer):
        eos_token_id = eos_id
        eos_token = "<eos>"

        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1, 2, 4]])

        def decode(self, *args, **kwargs):
            return "<think>123 is not the answer</think>**Answer:** 7<eos>"

    problem = TaskProblem("fixture", "arithmetic", "pilot", "fixture", "test", "Answer: ", "7")
    record = _decode_method_on_problem(
        adapter, Tokenizer(), problem, "S0", 42, 4, 1, 1, "none", 0.0,
        ExcursionConfig(), AdaptivePolicyConfig(), max_output_tokens=1, prompt_format="chat",
    )
    assert record["is_correct"] is correct
    assert record["extracted_answer"] == ("7" if correct else "")
    assert record["answer_extraction"] == "completed_final_integer_v1"
