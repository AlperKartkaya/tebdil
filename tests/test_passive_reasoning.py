"""Passive observation is not an intervention or semantic stall annotation."""
import copy
import json

import pytest
import torch

from tebdil.core.state import hash_tensor
from tebdil.evaluation.monitoring import passive_observer, repeat_features
from tebdil.models.ouro_adapter import OuroRecurrentAdapter
from test_prompt_started_reasoning import Tokenizer, runner


class TextTokenizer:
    def decode(self, tokens, **kwargs):
        assert kwargs == {'skip_special_tokens': False}
        return ''.join('</think>' if t==9 else '<think>' if t==10 else 'x' for t in tokens)


def test_repeat_screen_is_causal_and_not_a_semantic_claim():
    tok=TextTokenizer()
    tokens=list(range(128))
    tokens=[v+20 for v in tokens]
    before=copy.deepcopy(tokens)
    features=repeat_features(tok,tokens)
    assert features['phase']=='reasoning' and not features['candidate_repeat_flag']
    assert tokens==before
    repeated=repeat_features(tok,[3]*128)
    assert repeated['prior_exact_8gram_matches']==120 and repeated['candidate_repeat_flag']
    assert not repeat_features(tok,[3]*127)['candidate_repeat_flag']
    assert not repeat_features(tok,[3]*128+[9])['candidate_repeat_flag']
    assert repeat_features(tok,[10]+[3]*128)['phase']=='malformed'
    assert repeat_features(tok,[9,9]+[3]*128)['phase']=='malformed'
    assert not any('gold' in key or 'correct' in key or 'stall' in key for key in repeated)


def test_native_passive_observer_does_not_change_tokens_calls_or_weights():
    adapter=OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    with torch.no_grad():
        adapter.model.lm_head.weight.zero_()
    tok=Tokenizer()
    tok.text='still reasoning'
    plain=runner().run_reasoning_completion(adapter,tok,'question',3,lambda e:None)
    weights={k:v.clone() for k,v in adapter.model.state_dict().items()}
    entries=[]
    with passive_observer(adapter,tok,entries.append) as observer:
        observed=runner().run_reasoning_completion(adapter,tok,'question',3,observer)
    assert observed==plain
    assert all(torch.equal(v,adapter.model.state_dict()[k]) for k,v in weights.items())
    assert len(entries)==3 and all(len(e['depth_states'])==4 for e in entries)
    assert [e['generated_prefix_token_count'] for e in entries]==[0,1,2]
    for k,e in enumerate(entries):
        assert e['prefix_token_ids_sha256']==observed['token_ledger'][k]['prefix_token_ids_sha256']
        for depth,state in enumerate(e['depth_states']):
            assert state['depth']==depth and state['position']==e['prefix_length']-1
            assert state['postnorm_update_l2']>=0
            assert 'input_vector' in state if k==0 else 'input_vector' not in state
            if depth:
                assert state['input_sha256']==e['depth_states'][depth-1]['normalized_sha256']
            if k==0:
                for name in ('input','raw','normalized'):
                    vector=torch.tensor(state[name+'_vector'],dtype=torch.float32)
                    assert hash_tensor(vector,64)==state[name+'_sha256']
                inp=torch.tensor(state['input_vector'])
                out=torch.tensor(state['normalized_vector'])
                assert abs(float(torch.linalg.vector_norm(out-inp))-state['postnorm_update_l2'])<1e-6
    assert not adapter.model.model.layers[0]._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks and not adapter.model._forward_hooks
    json.dumps(entries,allow_nan=False)


def test_observer_failure_and_nonfinite_states_remove_all_hooks():
    adapter=OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    tok=Tokenizer()
    def fail(entry):
        raise RuntimeError('journal failure')
    with pytest.raises(RuntimeError,match='journal failure'):
        with passive_observer(adapter,tok,fail) as observer:
            runner().run_reasoning_completion(adapter,tok,'question',1,observer)
    assert not adapter.model.model.layers[0]._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks and not adapter.model._forward_hooks
    with torch.no_grad():
        adapter.model.model.embed_tokens.weight.fill_(float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):
        with passive_observer(adapter,tok,lambda e:None) as observer:
            runner().run_reasoning_completion(adapter,tok,'question',1,observer)
    assert not adapter.model.model.layers[0]._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks and not adapter.model._forward_hooks


def test_recorder_refuses_existing_directory_before_model_load(tmp_path):
    from pathlib import Path
    import subprocess
    root=Path(__file__).resolve().parents[1]
    run=subprocess.run(['uv','run','python','scripts/record_passive_reasoning.py','--output-dir',str(tmp_path)],
                       cwd=root,text=True,capture_output=True)
    assert run.returncode != 0 and 'preserve all partial evidence' in run.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.real_model
def test_real_checkpoint_passive_true_prefix_parity():
    from transformers import AutoTokenizer
    from tebdil.models.ouro_adapter import PINNED_OURO_THINKING_REV
    tok=AutoTokenizer.from_pretrained('ByteDance/Ouro-1.4B-Thinking', revision=PINNED_OURO_THINKING_REV, local_files_only=True)
    adapter=OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16,local_files_only=True)
    prompt='Reason briefly: if P implies Q, and P is true, is Q true?'
    plain=runner().run_reasoning_completion(adapter,tok,prompt,3,lambda e:None)
    entries=[]
    with passive_observer(adapter,tok,entries.append) as observer:
        observed=runner().run_reasoning_completion(adapter,tok,prompt,3,observer)
    assert observed==plain and observed['recurrent_block_calls']==12
    assert len(entries)==3
    for state in entries[0]['depth_states']:
        assert state['dtype']=='torch.bfloat16'
        for key in ('input','raw','normalized'):
            vector=torch.tensor(state[key+'_vector'],dtype=torch.bfloat16)
            assert hash_tensor(vector,64)==state[key+'_sha256']
    assert not adapter.model.model.layers[0]._forward_pre_hooks
    assert not adapter.model.model.norm._forward_hooks and not adapter.model._forward_hooks
