import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('audit_e2', ROOT / 'scripts/audit_e2_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('jobs,correct,rescues,regressions,missing', [
    (32, (11, 12), 2, 1, 0),
    (76, (26, 25), 4, 5, 32),
])
def test_historical_e2_recount(jobs, correct, rescues, regressions, missing):
    report = json.loads((ROOT / f'reports/colab_l4_deceptive_traps_3000_{jobs}jobs.json').read_text())
    original = copy.deepcopy(report)
    result = audit.summarize(report)
    assert tuple(result['methods'][m]['stored_correct'] for m in ('S0', 'E2')) == correct
    assert len(result['rescues']) == rescues
    assert len(result['regressions']) == regressions
    assert sum(len(m['missing_raw_token_ids']) for m in result['methods'].values()) == missing
    assert report == original


@pytest.mark.parametrize('damage', ['duplicate', 'missing_pair', 'token_count', 'wrong_method', 'non_boolean'])
def test_invalid_pairing_or_records_fail_closed(damage):
    report = json.loads((ROOT / 'reports/colab_l4_deceptive_traps_3000_32jobs.json').read_text())
    rows = report['deceptive_traps_records']
    if damage == 'duplicate':
        rows.append(rows[0])
    elif damage == 'missing_pair':
        rows.pop()
    elif damage == 'token_count':
        rows[0]['generated_tokens'] += 1
    elif damage == 'wrong_method':
        rows[0]['method'] = 'E1'
    else:
        rows[0]['evaluation']['is_correct'] = 'true'
    with pytest.raises(ValueError):
        audit.summarize(report)
