"""Synthetic mini-native continuation contracts, not research outcomes."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tebdil.core.state import hash_tensor
from tebdil.models.ouro_adapter import OuroRecurrentAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('budget_runner',ROOT/'scripts/extend_reasoning_baselines.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 2
    eos_token = '<eos>'
    def decode(self, ids, **kwargs):
        assert kwargs == {'skip_special_tokens':False}
        return ' '.join(self.eos_token if i == self.eos_token_id else str(i) for i in ids)


def model():
    return OuroRecurrentAdapter.from_mini_config(vocab_size=128,hidden_size=16,
        intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,seed=11)


def original(tokens=(9,8,7)):
    return dict(prompt_token_ids=[1,4],generated_token_ids=list(tokens),completed=False,
                reasoning_status='open',recurrent_block_calls=4*len(tokens))


def test_resume_matches_uninterrupted_native_tokens_and_actual_prefixes():
    adapter, tok = model(), Tokenizer()
    weights = {k:v.clone() for k,v in adapter.model.state_dict().items()}
    full,_,_ = runner.native_generate(adapter,tok,torch.tensor([[1,4]]),6)
    assert len(full) == 6 and tok.eos_token_id not in full[:3]
    old = original(full[:3])
    events = []
    row = runner.resume_baseline(adapter,tok,old,3,events.append)
    assert row['generated_token_ids'] == full
    assert row['prior_generated_tokens'] == row['new_generated_tokens'] == 3
    assert row['total_generated_tokens'] == 6
    assert row['new_generation_calls'] == row['reused_baseline_calls'] == 12
    for k,e in enumerate(row['continuation_token_ledger']):
        prefix = old['prompt_token_ids'] + full[:3+k]
        assert e['prefix_length'] == len(prefix)
        assert e['prefix_token_ids_sha256'] == hash_tensor(torch.tensor([prefix]),64)
        assert e['calls_before'] == 4*k and e['recurrent_block_calls'] == 4
        assert e['token_index'] == 3+k and e['token_id'] == full[3+k]
        assert 'processed_input_ids' not in e  # IDs are reconstructible; avoid quadratic storage.
    assert row['new_recurrent_token_positions'] == sum(e['prefix_length']*4 for e in events)
    assert row['reasoning_status'] == 'open' and not row['completed']
    assert all(torch.equal(v,weights[k]) for k,v in adapter.model.state_dict().items())
    assert not adapter.model._forward_hooks and not adapter.model.model.norm._forward_hooks


def test_terminal_parity_is_exact_and_mismatch_fails_closed():
    adapter, tok = model(), Tokenizer()
    states = []
    handle = adapter.model.model.norm.register_forward_hook(lambda m,a,o:states.append(hash_tensor(o[0,-1],64)))
    try:
        tokens,_,_ = runner.native_generate(adapter,tok,torch.tensor([[1,4]]),3)
    finally:
        handle.remove()
    case = dict(record=original(tokens),observations=[dict(depth_states=[dict(normalized_sha256=states[-1])])])
    check = runner.verify_last_proposal(adapter,tok,case)
    assert check['token_id'] == tokens[-1] and check['recurrent_calls'] == 4
    assert check['recurrent_token_positions'] == 4*4
    case['observations'][-1]['depth_states'][-1]['normalized_sha256'] = 'wrong'
    with pytest.raises(ValueError,match='parity'):
        runner.verify_last_proposal(adapter,tok,case)
    assert not adapter.model.model.norm._forward_hooks


def test_native_eos_stops_suffix_without_inventing_closed_reasoning():
    adapter, tok = model(), Tokenizer()
    tok.eos_token_id = tok.pad_token_id = 0
    with torch.no_grad():
        adapter.model.lm_head.weight.zero_()
    row = runner.resume_baseline(adapter,tok,original(),5,lambda e:None)
    assert row['generated_token_ids'] == [9,8,7,0]
    assert row['new_generated_tokens'] == 1 and row['new_generation_calls'] == 4
    assert row['completed'] and row['stop_reason'] == 'eos_token'
    assert row['reasoning_status'] == 'missing_close' and row['final_text'] is None


def test_failed_journal_callback_removes_native_and_observer_hooks():
    adapter = model()
    def fail(entry):
        raise RuntimeError('synthetic journal failure')
    with pytest.raises(RuntimeError,match='synthetic journal'):
        runner.resume_baseline(adapter,Tokenizer(),original(),3,fail)
    assert not adapter.model._forward_hooks and not adapter.model._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks


def test_nonfinite_readout_aborts_and_removes_observer_hooks():
    adapter = model()
    def corrupt(module,args,output):
        output.logits.fill_(float('nan'))
    handle = adapter.model.register_forward_hook(corrupt)
    try:
        with pytest.raises(ValueError,match='readout'):
            runner.resume_baseline(adapter,Tokenizer(),original(),3,lambda e:None)
        assert len(adapter.model._forward_hooks) == 1
        assert not adapter.model._forward_pre_hooks and not adapter.model.model.norm._forward_hooks
    finally:
        handle.remove()


@pytest.mark.parametrize('changes,budget',[
    ({'completed':True},3),({'generated_token_ids':[9,2]},3),
    ({'reasoning_status':'closed'},3),({},0),
])
def test_completed_or_invalid_baselines_are_not_extended(changes,budget):
    adapter = model()
    with pytest.raises(ValueError,match='unfinished'):
        runner.resume_baseline(adapter,Tokenizer(),{**original(),**changes},budget,lambda e:None)
    assert not adapter.model._forward_hooks


def test_runner_refuses_existing_output_before_model_load(tmp_path):
    keep = tmp_path/'journal.jsonl'
    keep.write_text('untouched\n')
    result = subprocess.run([sys.executable,'scripts/extend_reasoning_baselines.py',
        '--output-dir',str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 2 and 'preserve' in result.stderr
    assert keep.read_text() == 'untouched\n'
