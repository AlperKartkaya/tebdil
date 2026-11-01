"""Mini-model instrumentation checks; no reserved questions or efficacy claims."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tebdil.models.ouro_adapter import OuroRecurrentAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('baseline_difficulty', ROOT / 'scripts/run_baseline_difficulty.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Tokenizer:
    eos_token_id = 2
    eos_token = '<eos>'
    pad_token_id = 2

    def __len__(self):
        return 128

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{'role': 'user', 'content': 'question'}]
        assert kwargs['enable_thinking'] is False
        return torch.tensor([[1, 4]])

    def decode(self, tokens, **kwargs):
        return r'<think>999</think>\(\boxed{7}\).<eos>'


def model():
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    with torch.no_grad():
        adapter.model.lm_head.weight.zero_()
    return adapter


@pytest.mark.parametrize('eos,budget,tokens', [(2, 2, [0, 0]), (0, 2, [0])])
def test_native_observer_matches_actual_tokens_calls_and_stops(eos, budget, tokens):
    adapter, tokenizer, events = model(), Tokenizer(), []
    tokenizer.eos_token_id = eos
    record = MODULE.run_native_completion(adapter, tokenizer, 'question', budget, events.append)
    assert record['generated_token_ids'] == tokens
    assert record['recurrent_block_calls'] == 4 * len(tokens)
    assert record['independent_forward_checks_passed']
    assert record['completed'] == (eos == 0)
    assert [e['greedy_readout_token_id'] for e in events] == tokens
    assert [e['prefix_length'] for e in events] == list(range(2, 2 + len(tokens)))
    assert all(e['recurrent_block_calls'] == 4 for e in events)
    assert 'gold_answer' not in record
    assert not adapter.model._forward_hooks and not adapter.model._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks


def test_observer_failure_preserves_event_and_cleans_all_hooks():
    adapter, events = model(), []

    def fail(entry):
        events.append(entry)
        raise RuntimeError('synthetic journal failure')

    with pytest.raises(RuntimeError, match='journal failure'):
        MODULE.run_native_completion(adapter, Tokenizer(), 'question', 2, fail)
    assert len(events) == 1 and events[0]['recurrent_block_calls'] == 4
    assert not adapter.model._forward_hooks and not adapter.model._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks


def test_nonfinite_readout_is_not_substituted_with_a_finished_answer():
    adapter = model()
    with torch.no_grad():
        adapter.model.lm_head.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='readout'):
        MODULE.run_native_completion(adapter, Tokenizer(), 'question', 2, lambda entry: None)
    assert not adapter.model._forward_hooks and not adapter.model.model.norm._forward_hooks


@pytest.mark.parametrize('counts,expected', [
    ({'correct': 12, 'incorrect': 0, 'invalid_format': 0, 'budget_exhausted': 0}, 'no_observed_rescue_opportunity'),
    ({'correct': 11, 'incorrect': 1, 'invalid_format': 0, 'budget_exhausted': 0}, 'recognized_baseline_failures'),
    ({'correct': 11, 'incorrect': 0, 'invalid_format': 1, 'budget_exhausted': 0}, 'headroom_unresolved_format_or_truncation'),
    ({'correct': 10, 'incorrect': 1, 'invalid_format': 0, 'budget_exhausted': 1}, 'headroom_unresolved_format_or_truncation'),
])
def test_difficulty_decision_does_not_call_format_or_budget_failure_wrong_arithmetic(counts, expected):
    assert MODULE.difficulty_decision(counts) == expected


def test_runner_refuses_existing_partial_directory(tmp_path):
    journal = tmp_path / 'journal.jsonl'
    journal.write_text('partial\n')
    run = subprocess.run([sys.executable, 'scripts/run_baseline_difficulty.py', '--output-dir', str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2 and 'preserve partial evidence' in run.stderr
    assert journal.read_text() == 'partial\n' and list(tmp_path.iterdir()) == [journal]
