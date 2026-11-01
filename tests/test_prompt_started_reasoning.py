"""Synthetic recording boundaries; not new Ouro reasoning outcomes."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tebdil.core.state import hash_tensor
from tebdil.evaluation.reasoning import (
    encode_reasoning_prompt, evaluate_reasoning_integer, split_prompt_started_reasoning,
    validate_reasoning_completion,
)
from tebdil.models.ouro_adapter import OuroRecurrentAdapter
from tebdil.tasks.completion_v2 import extract_completed_integer_v2

ROOT = Path(__file__).resolve().parents[1]
PROMPT = '<|im_start|>assistant\n<think>\n'


class Tokenizer:
    eos_token_id = 2
    eos_token = '<eos>'
    pad_token_id = 2
    prompt_text = PROMPT
    text = '999 is intermediate.\n</think>\nAnswer: 7<eos>'

    def __len__(self):
        return 128

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{'role': 'user', 'content': 'question'}]
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors='pt')
        return torch.tensor([[1, 4]])

    def decode(self, tokens, **kwargs):
        assert kwargs == {'skip_special_tokens': False}
        return self.prompt_text if tokens == [1, 4] else self.text


def fixture(text=None, tokens=None):
    tokenizer = Tokenizer()
    if text is not None:
        tokenizer.text = text
    tokens = [7, 2] if tokens is None else tokens
    prompt = [1, 4]
    ledger = []
    for k in range(len(tokens)):
        prefix = prompt + tokens[:k]
        ledger.append(dict(processed_input_ids=prefix, prefix_length=len(prefix), processed_sequence_length=len(prefix),
                           prefix_token_ids_sha256=hash_tensor(torch.tensor([prefix]), 64),
                           calls_before=4*k, recurrent_block_calls=4))
    return dict(tokenizer=tokenizer, prompt_ids=prompt, tokens=tokens, ledger=ledger, calls=4*len(tokens), budget=2, depth=4)


def test_true_prompt_is_explicit_and_old_parser_cannot_handle_its_generated_suffix():
    assert encode_reasoning_prompt(Tokenizer(), 'question').tolist() == [[1, 4]]
    assert extract_completed_integer_v2(Tokenizer.text, '<eos>', completed=True) is None
    row = validate_reasoning_completion(**fixture())
    assert row['enable_thinking'] is True and row['reasoning_open_in_prompt'] is True
    assert row['prompt_text_with_special_tokens'] == PROMPT
    assert row['decoded_text_with_special_tokens'] == Tokenizer.text
    assert not row['decoded_text_with_special_tokens'].startswith('<think>')
    assert row['reasoning_text'] == '999 is intermediate.\n'
    assert row['final_text'] == '\nAnswer: 7'
    assert row['reasoning_status'] == 'closed'
    original = deepcopy(row)
    assert evaluate_reasoning_integer(row, '7')['extracted_answer'] == '7'
    assert row == original and 'gold_answer' not in row
    assert evaluate_reasoning_integer(row, '7')['status'] == 'correct'
    assert evaluate_reasoning_integer(row, '8')['status'] == 'incorrect'


@pytest.mark.parametrize('text,completed,status,answer', [
    ('</think>7<eos>', True, 'closed', '7'),
    ('999</think>\\(\\boxed{7}\\).<eos>', True, 'closed', '7'),
    ('999</think>Answer: 7', False, 'closed', None),
    ('999 Answer: 7', False, 'open', None),
    ('999 Answer: 7<eos>', True, 'missing_close', None),
    ('<think>999</think>Answer: 7<eos>', True, 'malformed', None),
    ('999</think></think>Answer: 7<eos>', True, 'malformed', None),
    ('999</think>Answer: 7<eos><eos>', True, 'malformed', None),
    ('999</think>Answer: 7<eos>extra', True, 'malformed', None),
    ('999</think>Answer: 7', True, 'malformed', None),
    ('999</think>Answer: 7<eos>', False, 'malformed', None),
    ('999</think><eos>', True, 'closed', None),
    ('999</think>Answer: 7\nAnswer: 8<eos>', True, 'closed', None),
    ('999</think>x = 7<eos>', True, 'closed', None),
])
def test_reasoning_boundaries_do_not_promote_intermediate_or_unfinished_answers(text, completed, status, answer):
    parts = split_prompt_started_reasoning(text, '<eos>', completed=completed)
    assert parts['reasoning_status'] == status
    record = {'completed': completed, **parts}
    result = evaluate_reasoning_integer(record, '7')
    assert result['extracted_answer'] == answer
    assert result['status'] == ('budget_exhausted' if not completed else 'correct' if answer else 'invalid_format')


def test_wrong_prompt_or_corrupt_native_ledger_fails_closed():
    args = fixture()
    args['tokenizer'].prompt_text = '<|im_start|>assistant\n'
    with pytest.raises(ValueError, match='prompt'):
        validate_reasoning_completion(**args)
    with pytest.raises(ValueError, match='prompt'):
        encode_reasoning_prompt(args['tokenizer'], 'question')
    args = fixture()
    args['ledger'][0]['prefix_token_ids_sha256'] = 'invented'
    with pytest.raises(ValueError, match='ledger'):
        validate_reasoning_completion(**args)


def runner():
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('thinking_smoke', ROOT / 'scripts/verify_prompt_started_reasoning.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('eos,tokens', [(2, [0, 0]), (0, [0])])
def test_mini_native_tokens_and_physical_calls_match_observer_and_hooks_are_removed(eos, tokens):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    with torch.no_grad():
        adapter.model.lm_head.weight.zero_()
    tokenizer, events = Tokenizer(), []
    tokenizer.eos_token_id = eos
    if eos == 2:
        tokenizer.text = 'still reasoning'
    row = runner().run_reasoning_completion(adapter, tokenizer, 'question', 2, events.append)
    assert row['generated_token_ids'] == tokens and row['recurrent_block_calls'] == 4*len(tokens)
    assert [e['greedy_readout_token_id'] for e in events] == tokens
    assert row['independent_forward_checks_passed']
    assert not adapter.model._forward_hooks and not adapter.model._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks


def test_forward_failure_cleans_hooks_and_does_not_create_completed_output():
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    events = []

    def fail(entry):
        events.append(entry)
        raise RuntimeError('synthetic journal failure')

    with pytest.raises(RuntimeError, match='journal failure'):
        runner().run_reasoning_completion(adapter, Tokenizer(), 'question', 2, fail)
    assert len(events) == 1
    assert not adapter.model._forward_hooks and not adapter.model._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks


def test_cli_refuses_existing_evidence(tmp_path):
    journal = tmp_path / 'existing.jsonl'
    journal.write_text('partial\n')
    run = subprocess.run([sys.executable, 'scripts/verify_prompt_started_reasoning.py', '--output', str(journal.with_suffix('.json'))], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2 and 'preserve' in run.stderr
    assert journal.read_text() == 'partial\n' and not journal.with_suffix('.json').exists()


def test_nonfinite_native_readout_aborts_and_removes_hooks():
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    with torch.no_grad():
        adapter.model.lm_head.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='readout'):
        runner().run_reasoning_completion(adapter, Tokenizer(), 'question', 2, lambda entry: None)
    assert not adapter.model._forward_hooks and not adapter.model._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks


@pytest.mark.parametrize('question', ['', ' ', None])
def test_question_must_be_nonempty_text(question):
    with pytest.raises(ValueError, match='question'):
        encode_reasoning_prompt(Tokenizer(), question)


def test_cli_rejects_report_journal_path_collision_before_model_loading(tmp_path):
    path = tmp_path / 'collision.jsonl'
    run = subprocess.run([sys.executable, 'scripts/verify_prompt_started_reasoning.py', '--output', str(path)], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2 and 'distinct' in run.stderr
    assert not path.exists()
