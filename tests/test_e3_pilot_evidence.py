"""Validate the fixed E3 pilot's transfer, pairing and exact checkpoint contracts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1] / 'reports/e3_pilot'


def test_e3_pilot_records_are_paired_and_bound_to_checkpoints():
    summary = json.loads((ROOT / 'summary.json').read_text())
    rows = summary['deceptive_traps_records']
    expected = {(pid, method) for pid in ('trap_hotel_empty_rooms', 'trap_monty_hall_4doors') for method in ('S0', 'E2', 'E3')}
    assert len(rows) == len(expected) == 6
    assert {(r['problem_id'], r['method']) for r in rows} == expected
    assert summary['dtype'] == 'torch.float32'
    assert summary['attn_implementation'] == 'sdpa'
    checkpoints = [json.loads(p.read_text()) for p in (ROOT / 'checkpoints').glob('*/generation.json')]
    assert len(checkpoints) == 6
    prompts = {}
    for row in rows:
        matches = [c for c in checkpoints if c['generation'] == {k: row[k] for k in c['generation']}]
        assert len(matches) == 1
        checkpoint = matches[0]
        assert checkpoint['contract']['identity'] == summary['execution_identity']
        parameters = checkpoint['contract']['parameters']
        assert parameters['method'] == row['method']
        assert parameters['max_new_tokens'] == 1024
        assert parameters['seed'] == 42
        assert parameters['excursion_config'] == row['excursion_config']
        assert parameters['policy_config'] == row['policy_config']
        prompt = parameters['prompt_ids']
        assert prompts.setdefault(row['problem_id'], prompt) == prompt
        assert not row['checkpoint_reused']
        assert row['excursion_count'] == len(row['excursion_token_indices'])
        rejected = row['rejected_return_token_indices']
        assert not set(rejected) & set(row['excursion_token_indices'])
        assert all(0 <= k < row['generated_tokens'] for k in rejected)
        if row['method'] != 'E3':
            assert rejected == []
        assert row['generated_tokens'] == len(row['generated_token_ids']) <= 1024
        if not row['completed']:
            assert row['generated_tokens'] == 1024
            assert row['evaluation']['status'] == 'budget_exhausted'
        assert row['recurrent_block_calls'] >= 4 * row['generated_tokens']
    for path in (ROOT / 'checkpoints').glob('*/generation.json'):
        record = json.loads(path.read_text())
        assert hashlib.sha256(json.dumps(record['contract'], sort_keys=True, allow_nan=False).encode()).hexdigest() == path.parent.name
        assert hashlib.sha256(json.dumps(record['generation'], sort_keys=True, allow_nan=False).encode()).hexdigest() == record['sha256']


def test_e3_pilot_remote_transfer_hashes():
    manifest = json.loads((ROOT / 'transfer.json').read_text())
    assert len(manifest['sha256']) == 7
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
