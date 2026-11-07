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
