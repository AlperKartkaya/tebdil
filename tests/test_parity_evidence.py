"""Keep diagnostic records attached to executed sources and honest gate outcomes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_parity_records_have_exact_sources_and_consistent_verdicts():
    paths = list((ROOT / 'reports').glob('cached_parity_*.json'))
    assert len(paths) == 8
    conditions = set()
    for path in paths:
        record = json.loads(path.read_text())
        conditions.add((record['model'], record['dtype'], record['attention']))
        digest = record['script_sha256']
        source = ROOT / 'reports/parity_sources' / (digest + '.py')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        assert len(record['rows']) == 6
        expected = all(row['finite'] and row['top1_match'] and row['cache_lengths_match'] and row['max_abs_difference'] <= record['absolute_logit_tolerance'] for row in record['rows'])
        assert record['passed'] is expected
    assert len(conditions) == 8


def test_extended_parity_records_match_sources_and_gates():
    for size in ['14', '26']:
        record = json.loads((ROOT / f'reports/extended_cache_parity_{size}b.json').read_text())
        source = ROOT / 'reports/parity_sources' / (record['script_sha256'] + '.py')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record['script_sha256']
        assert [row['prefix_length'] for row in record['rows']] == [128, 129, 130, 512, 513, 514]
        assert record['passed']
        assert all(row['finite'] and row['top1_match'] and row['cache_lengths_match'] and row['max_abs_difference'] <= record['absolute_logit_tolerance'] for row in record['rows'])


def test_exported_checkpoint_is_identical_after_restore():
    folder = ROOT / 'reports/checkpoint_recovery'
    transfer = json.loads((folder / 'transfer.json').read_text())
    raw = (folder / 'generation.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == transfer['sha256']
    checkpoint = json.loads(raw)
    contract = json.dumps(checkpoint['contract'], sort_keys=True, allow_nan=False).encode()
    assert hashlib.sha256(contract).hexdigest() == transfer['checkpoint_key']
    payload = json.dumps(checkpoint['generation'], sort_keys=True, allow_nan=False).encode()
    assert hashlib.sha256(payload).hexdigest() == checkpoint['sha256']
    original = json.loads((folder / 'original_summary.json').read_text())['completion_reasoning_records']
    restored = json.loads((folder / 'restored_summary.json').read_text())['completion_reasoning_records']
    assert len(original) == len(restored) == 1
    assert original[0].pop('checkpoint_reused') is False
    assert restored[0].pop('checkpoint_reused') is True
    assert original == restored
    assert not original[0]['completed']  # Four-token smoke, not a scored answer.
