"""Mini-native checks only; these fixtures are not experimental outcomes."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.models.ouro_adapter import OuroRecurrentAdapter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('reasoning_probe',ROOT/'scripts/probe_reasoning_cot.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Tokenizer:
    eos_token_id=2
    pad_token_id=0
    eos_token='<eos>'
    def decode(self,ids,**kwargs):
        return ''.join('<eos>' if v==2 else f'[{v}]' for v in ids)


def model():
    return OuroRecurrentAdapter.from_mini_config(vocab_size=128)


@pytest.mark.parametrize('beta',[0.,.65])
def test_probe_native_parity_cost_isolation_and_identity(beta):
    adapter=model()
    ids=torch.tensor([[1,4,9]])
    weights={k:v.clone() for k,v in adapter.model.state_dict().items()}
    with torch.no_grad():
        native=adapter.initialize(ids)
        for _ in range(4):
            native=adapter.transition(native)
        logits=adapter.readout_logits(native)[0,-1]
    row=probe.probe_prefix(adapter,ids,ExcursionConfig(alpha=.65,beta=beta,max_relative_norm=.1),12)
    assert row['measurements'][0]['next_token_id']==int(logits.argmax())
    assert row['measurements'][0]['final_last_position_hash']==hash_tensor(native.hidden[0,-1],64)
    assert row['prefix_sha256']==hash_tensor(ids,64)
    assert row['recurrent_calls']==(16 if beta else 15)
    assert row['recurrent_calls']==sum(e['recurrent_calls'] for e in row['stages'])
    assert row['recurrent_token_positions']==3*row['recurrent_calls']
    assert row['inputs_unchanged']
    assert [m['method'] for m in row['measurements']]==list(probe.METHODS)
    assert row['measurements'][0]['logit_l2_vs_S0']==0
    if beta==0:
        assert row['random_target_norm']==0
        assert all(m['next_token_id']==row['measurements'][0]['next_token_id'] for m in row['measurements'])
        # M1 uses FP32 subtract/add; its no-op can retain rounding, not a changed policy.
        assert all(m['shift_vs_ordinary'] <= (1e-6 if m['method']=='M1' else 0.) for m in row['measurements'])
    assert all(torch.equal(v,weights[k]) for k,v in adapter.model.state_dict().items())
    assert not adapter.model.model.norm._forward_hooks


def test_probe_failure_removes_physical_hook():
    adapter=model()
    def fail(*args):
        raise RuntimeError('synthetic readout failure')
    adapter.readout_logits=fail
    with pytest.raises(RuntimeError,match='synthetic'):
        probe.probe_prefix(adapter,torch.tensor([[1,4]]),ExcursionConfig(),1)
    assert not adapter.model.model.norm._forward_hooks


def test_short_window_is_native_unedited_prefix_and_not_a_wrong_final():
    adapter=model()
    original=dict(prompt_token_ids=[1,4],generated_token_ids=[9,8,7])
    events=[]
    row=probe.continue_window(adapter,Tokenizer(),original,1,6,3,events.append)
    assert row['generated_token_ids'][0]==6
    assert len(events)==len(row['generated_token_ids'])-1
    assert row['new_recurrent_calls']==4*len(events)
    if events:
        assert events[0]['prefix_sha256']==hash_tensor(torch.tensor([[1,4,9,6]]),64)
        assert [e['token_id'] for e in events]==row['generated_token_ids'][1:]
    assert row['stop_reason'] in ('eos_token','window_exhausted')
    assert 'status' not in row and 'extracted_answer' not in row
    assert not adapter.model._forward_hooks and not adapter.model.model.norm._forward_hooks


def test_window_eos_proposal_requires_no_extra_model_call():
    adapter=model()
    events=[]
    row=probe.continue_window(adapter,Tokenizer(),dict(prompt_token_ids=[1,4],generated_token_ids=[9]),0,2,3,events.append)
    assert row['completed'] and row['stop_reason']=='eos_token'
    assert row['generated_token_ids']==[2] and row['new_recurrent_calls']==0
    assert not events


def test_runner_refuses_existing_evidence_before_loading_checkpoint(tmp_path):
    p=tmp_path/'journal.jsonl'
    p.write_text('keep me\n')
    result=subprocess.run([sys.executable,'scripts/probe_reasoning_cot.py','--output-dir',str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==2 and 'preserve' in result.stderr
    assert p.read_text()=='keep me\n'
