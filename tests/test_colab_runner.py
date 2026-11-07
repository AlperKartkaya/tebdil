import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('colab_runner', Path(__file__).parents[1] / 'scripts/run_colab_l4.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

@pytest.mark.parametrize('extra', [['--model', 'ByteDance/Ouro-2.6B-Thinking'], ['--revision', 'main']])
def test_requires_immutable_model_revision(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', ['runner', '--output', str(tmp_path / 'summary.json')] + extra)
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
    rows = runner.run_stall_recovery_suite(SimpleNamespace(device='cpu'), SimpleNamespace(decode=lambda *a, **k: ''), ['S0', 'E1'], None, None, generator=generate)
    assert rows[0]['methods']['E1']['changed_tokens_vs_s0'] == 1
    assert rows[0]['methods']['E1']['first_diff_token_offset_vs_s0'] == 1


def test_generation_checkpoint_reuses_and_rejects_corruption(monkeypatch, tmp_path):
    import json
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return {'generated_token_ids': [1], 'completed': True}
    monkeypatch.setattr(runner, 'generate_tebdil_cached', generate)
    cached = runner.checkpointed_generator(tmp_path, {'model': 'fixture'})
    assert cached(method='S0')['checkpoint_reused'] is False
    assert cached(method='S0')['checkpoint_reused'] is True
    assert len(calls) == 1
    path = next(tmp_path.rglob('generation.json'))
    record = json.loads(path.read_text())
    record['generation']['generated_token_ids'] = [2]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='checksum'):
        cached(method='S0')


def test_checkpoint_failure_is_not_reused(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise RuntimeError('runtime lost')
    monkeypatch.setattr(runner, 'generate_tebdil_cached', fail)
    cached = runner.checkpointed_generator(tmp_path, {})
    with pytest.raises(RuntimeError):
        cached(method='S0')
    assert not list(tmp_path.iterdir())


def test_checkpoint_publication_failure_leaves_no_reusable_record(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'generate_tebdil_cached', lambda **kwargs: {'generated_token_ids': [1]})
    def fail(*args):
        raise OSError('simulated interrupted publish')
    monkeypatch.setattr(runner.os, 'rename', fail)
    with pytest.raises(OSError):
        runner.checkpointed_generator(tmp_path, {})(method='S0')
    assert not list(tmp_path.iterdir())


def test_checkpoint_identity_change_does_not_reuse(monkeypatch, tmp_path):
    calls = []
    def generate(**kwargs):
        calls.append(1)
        return {'generated_token_ids': [1]}
    monkeypatch.setattr(runner, 'generate_tebdil_cached', generate)
    for identity in [{'revision': 'a'}, {'revision': 'b'}]:
        assert not runner.checkpointed_generator(tmp_path, identity)(method='S0')['checkpoint_reused']
    assert len(calls) == 2


@pytest.mark.parametrize('extra', [['--max-tasks', '0'], ['--max-new-tokens', '-1'], ['--methods', 'S0', 'S0']])
def test_invalid_run_contract_rejected_before_model_loading(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', ['runner', '--output', str(tmp_path / 'summary.json')] + extra)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_unknown_task_id_is_not_an_empty_success(monkeypatch, tmp_path):
    import json
    folder = tmp_path / 'configs'
    folder.mkdir()
    (folder / 'literature_reasoning_v1_tasks.json').write_text(json.dumps({'partitions': {'development': []}}))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Unknown task IDs'):
        runner.run_literature_reasoning_suite(None, None, [], None, None, task_ids=['typo'])


def test_checkpoint_survives_a_fresh_process(tmp_path):
    import subprocess
    source = str(Path(runner.__file__).resolve())
    code = f'''import runpy
from pathlib import Path
namespace = runpy.run_path({source!r})
factory = namespace['checkpointed_generator']
factory.__globals__['generate_tebdil_cached'] = lambda **kwargs: {{'generated_token_ids': [1]}}
print(factory(Path({str(tmp_path)!r}), {{'revision': 'same'}})(method='S0')['checkpoint_reused'])
'''
    first = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    second = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert first.stdout.strip() == 'False'
    assert second.stdout.strip() == 'True'


def test_interrupted_temporary_directory_is_not_a_checkpoint(monkeypatch, tmp_path):
    pending = tmp_path / '.pending-crashed'
    pending.mkdir()
    (pending / 'generation.json').write_text('{')
    monkeypatch.setattr(runner, 'generate_tebdil_cached', lambda **kwargs: {'generated_token_ids': [1]})
    assert not runner.checkpointed_generator(tmp_path, {})(method='S0')['checkpoint_reused']
    assert pending.exists()  # Unowned crash evidence is preserved.


@pytest.mark.parametrize('extra', [
    ['--suite', 'completion', '--partition', 'reserved_test'],
    ['--suite', 'stall', '--partition', 'reserved_test'],
    ['--suite', 'all', '--partition', 'reserved_test'],
    ['--suite', 'completion', '--task-ids', 'completion_dev_000'],
    ['--suite', 'reasoning_v1', '--task-ids', 'some-task'],
    ['--suite', 'all', '--task-ids', 'some-task'],
    ['--suite', 'literature', '--task-ids'],
])
def test_ignored_selection_flags_are_rejected(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', ['runner', '--output', str(tmp_path / 'summary.json')] + extra)
    monkeypatch.setattr(runner.AutoTokenizer, 'from_pretrained', lambda *a, **k: pytest.fail('model loading must not start'))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize('answer', ['+029', '029', '29'])
def test_equation_fallback_compares_integer_values(monkeypatch, answer):
    import torch
    from types import SimpleNamespace
    monkeypatch.setattr(runner, 'encode_reasoning_prompt', lambda *a: torch.tensor([[1]]))
    def generate(**kwargs):
        return {'decoded_text': f'calculation</think>x = {answer}<eos>',
                'completed': True, 'generated_token_ids': [2],
                'excursion_count': 0, 'recurrent_block_calls': 4, 'wall_seconds': 0}
    rows = runner.run_completion_reasoning_suite(
        SimpleNamespace(device='cpu'), SimpleNamespace(eos_token='<eos>'),
        ['S0'], runner.ExcursionConfig(), runner.AdaptivePolicyConfig(),
        max_tasks=1, generator=generate,
    )
    assert rows[0]['evaluation']['status'] == 'correct'
    assert rows[0]['excursion_config']['excursion_window_tokens'] == 1
    assert rows[0]['policy_config'] == runner.asdict(runner.AdaptivePolicyConfig())


def test_literature_records_effective_configuration(monkeypatch):
    import torch
    from types import SimpleNamespace
    monkeypatch.setattr(runner, 'encode_reasoning_prompt', lambda *a: torch.tensor([[1]]))
    monkeypatch.setattr(runner, 'evaluate_literature_reasoning', lambda *a: {'extracted_answer': None, 'status': 'budget_exhausted'})
    def generate(**kwargs):
        return {'decoded_text': 'unfinished', 'completed': False, 'generated_token_ids': [2],
                'excursion_count': 0, 'recurrent_block_calls': 4, 'wall_seconds': 0}
    rows = runner.run_literature_reasoning_suite(
        SimpleNamespace(device='cpu'), SimpleNamespace(eos_token='<eos>'),
        ['S0'], runner.ExcursionConfig(), runner.AdaptivePolicyConfig(),
        max_tasks=1, generator=generate,
    )
    assert rows[0]['excursion_config']['excursion_window_tokens'] == 1
    assert rows[0]['excursion_config']['beta'] == 0.65
    assert rows[0]['policy_config']['min_entropy_trigger'] == 1.20


def test_deceptive_suite_preserves_historical_settings(monkeypatch):
    import torch
    from types import SimpleNamespace
    seen = []
    monkeypatch.setattr(runner, 'encode_reasoning_prompt', lambda *a: torch.tensor([[1]]))
    def generate(**kwargs):
        seen.append(kwargs)
        return {'decoded_text': 'unfinished', 'completed': False, 'generated_token_ids': [2], 'excursion_count': 0, 'recurrent_block_calls': 4, 'wall_seconds': 0}
    rows = runner.run_literature_reasoning_suite(SimpleNamespace(device='cpu'), SimpleNamespace(eos_token='<eos>'), ['E3'], runner.ExcursionConfig(), runner.AdaptivePolicyConfig(), task_suite='deceptive', task_ids=['trap_hotel_empty_rooms'], generator=generate)
    assert len(rows) == 1
    assert rows[0]['problem_id'] == 'trap_hotel_empty_rooms'
    assert seen[0]['excursion_config'].alpha == 0.65
    assert seen[0]['excursion_config'].beta == 0.65
    assert seen[0]['excursion_config'].excursion_window_tokens == 1
    assert seen[0]['perturbation_relative_norm'] == 0.14
    assert seen[0]['method'] == 'E3'
    assert rows[0]['evaluation']['status'] == 'budget_exhausted'
