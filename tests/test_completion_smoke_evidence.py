"""A bounded development smoke is not a benchmark or compute-matched study."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FOLDER = ROOT / 'reports/completion_smoke'


@pytest.mark.parametrize('size', ['14b', '26b'])
def test_smoke_records_match_frozen_checkpoint_contracts(size):
    summary = json.loads((FOLDER / f'{size}.json').read_text())
    rows = summary['completion_reasoning_records']
    assert [row['method'] for row in rows] == ['S0', 'E1']
    assert summary['dtype'] == 'torch.float32'
    assert summary['attn_implementation'] == 'sdpa'
    assert summary['seed'] == 42
    checkpoints = list((FOLDER / f'{size}-checkpoints').glob('*/generation.json'))
    assert len(checkpoints) == 2
    for path in checkpoints:
        checkpoint = json.loads(path.read_text())
        contract, generation = checkpoint['contract'], checkpoint['generation']
        assert hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest() == path.parent.name
        assert hashlib.sha256(json.dumps(generation, sort_keys=True, allow_nan=False).encode()).hexdigest() == checkpoint['sha256']
        assert contract['identity'] == summary['execution_identity']
        parameters = contract['parameters']
        assert parameters['max_new_tokens'] == 256
        assert parameters['excursion_config']['excursion_window_tokens'] == 1
        row = next(r for r in rows if r['method'] == parameters['method'])
        assert row['problem_id'] == 'completion_dev_000'
        assert not row['checkpoint_reused']
        assert all(row[key] == value for key, value in generation.items())
        assert len(generation['generated_token_ids']) <= 256
        assert generation['completed'] == (generation['stop_reason'] == 'eos_token')
        if not generation['completed']:
            assert len(generation['generated_token_ids']) == 256
            assert row['evaluation']['status'] == 'budget_exhausted'
            assert not row['evaluation']['is_correct']
        elif row['evaluation']['status'] == 'correct':
            assert row['reasoning_status'] == 'closed'
            assert int(row['evaluation']['extracted_answer']) == int(row['gold_answer']) == 29
        assert generation['excursion_count'] == len(generation['excursion_token_indices'])
        if parameters['method'] == 'S0':
            assert generation['excursion_count'] == 0
        assert generation['recurrent_block_calls'] >= 4 * len(generation['generated_token_ids'])


def test_smoke_transferred_bytes_and_runner_source():
    manifest = json.loads((FOLDER / 'transfer.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == digest
    runner = (FOLDER / 'run_colab_l4.py').read_bytes()
    for size in ['14b', '26b']:
        summary = json.loads((FOLDER / f'{size}.json').read_text())
        assert hashlib.sha256(runner).hexdigest() == summary['execution_identity']['sources']['scripts/run_colab_l4.py']
