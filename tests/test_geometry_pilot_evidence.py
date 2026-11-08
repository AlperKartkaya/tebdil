"""Frozen development pilot: verify transfer, pairing, scoring and abstention."""
import hashlib
import json
from pathlib import Path
from tebdil.tasks.deceptive_reasoning import evaluate_deceptive_reasoning

ROOT = Path(__file__).parents[1]
REPORT = ROOT / 'reports/geometry_pilot'


def test_geometry_pilot_evidence():
    manifest = json.loads((REPORT / 'transfer.json').read_text())
    assert len(manifest['sha256']) == 15  # 12 checkpoints, summary and two logs.
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == digest
    summary = json.loads((REPORT / 'summary.json').read_text())
    rows = summary['deceptive_traps_records']
    ids = {'trap_hotel_empty_rooms', 'trap_monty_hall_4doors', 'trap_bat_ball_glove', 'trap_speed_harmonic_mean'}
    assert len(rows) == 12
    assert {(r['problem_id'], r['method']) for r in rows} == {(p, m) for p in ids for m in ('S0', 'E2', 'geometry')}
    assert summary['dtype'] == 'torch.float32' and summary['attn_implementation'] == 'sdpa'
    tasks = {t['problem_id']: t for t in json.loads((ROOT / 'configs/deceptive_traps_v1_tasks.json').read_text())['tasks']}
    checkpoints = []
    for p in (REPORT / 'checkpoints').glob('*/generation.json'):
        record = json.loads(p.read_text())
        assert hashlib.sha256(json.dumps(record['contract'], sort_keys=True, allow_nan=False).encode()).hexdigest() == p.parent.name
        assert hashlib.sha256(json.dumps(record['generation'], sort_keys=True, allow_nan=False).encode()).hexdigest() == record['sha256']
        assert record['contract']['identity'] == summary['execution_identity']
        assert record['contract']['parameters']['max_new_tokens'] == 1024
        assert record['contract']['parameters']['seed'] == 42
        checkpoints.append(record)
    assert len(checkpoints) == 12
    for row in rows:
        matches = [c for c in checkpoints if c['generation']['method'] == row['method']
                   and c['generation'] == {k: row[k] for k in c['generation']}]
        assert len(matches) == 1
        assert matches[0]['contract']['parameters']['excursion_config'] == row['excursion_config']
        assert row['gold_answer'] == tasks[row['problem_id']]['gold_answer']
        assert evaluate_deceptive_reasoning(row, row['gold_answer'])['status'] == row['evaluation']['status']
        assert row['evaluation']['status'] == ('budget_exhausted' if row['problem_id'] == 'trap_monty_hall_4doors' else 'correct')
        if row['method'] == 'geometry':
            native = next(r for r in rows if r['method'] == 'S0' and r['problem_id'] == row['problem_id'])
            assert row['generated_token_ids'] == native['generated_token_ids']
            assert row['recurrent_block_calls'] == native['recurrent_block_calls']
            assert row['excursion_count'] == 0
            records = row['geometry']['records']
            assert len(records) == row['geometry']['extra_readout_projections'] == row['generated_tokens']
            for r in records:
                assert not r['eligible'] and not r['attempted'] and r['selected'] is None
                assert all(pos < r['token_position'] for pos in r['bank_positions'])
                assert r['motion']['reason'] in ('progress', 'insufficient_or_invalid_geometry')
