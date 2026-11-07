import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('colab_runner', Path(__file__).parents[1] / 'scripts/run_colab_l4.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

@pytest.mark.parametrize('extra', [['--model', 'ByteDance/Ouro-2.6B-Thinking'], ['--revision', 'main']])
def test_requires_immutable_model_revision(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['runner'] + extra)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_stall_comparison_counts_unmatched_suffix(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    folder = tmp_path / 'reports/passive_reasoning_v1'
    folder.mkdir(parents=True)
    ids = ['reasoning_v1_development_cnf_0_0', 'reasoning_v1_development_key_planning_0_0']
    records = []
    for index, problem_id in enumerate(ids):
        filename = f'{index}.json'
        (folder / filename).write_text(json.dumps({'record': {'prompt_token_ids': [1], 'generated_token_ids': [2]}}))
        records.append({'problem_id': problem_id, 'raw_file': filename})
    (folder / 'report.json').write_text(json.dumps({'records': records}))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    def generate(**kwargs):
        tokens = [1] if kwargs['method'] == 'S0' else [1, 2]
        return {'generated_token_ids': tokens, 'excursion_count': 0, 'recurrent_block_calls': 1, 'wall_seconds': 0, 'decoded_text': ''}
    monkeypatch.setattr(runner, 'generate_tebdil_cached', generate)
    rows = runner.run_stall_recovery_suite(SimpleNamespace(device='cpu'), SimpleNamespace(decode=lambda *a, **k: ''), ['S0', 'E1'], None, None)
    assert rows[0]['methods']['E1']['changed_tokens_vs_s0'] == 1
    assert rows[0]['methods']['E1']['first_diff_token_offset_vs_s0'] == 1
