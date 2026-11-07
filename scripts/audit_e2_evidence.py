"""Recount historical E2 reports without treating stored scores as verified accuracy."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(report: dict) -> dict:
    pairs = {}
    for row in report['deceptive_traps_records']:
        method = row['method']
        if method not in ('S0', 'E2'):
            raise ValueError(f'Unexpected method: {method}')
        pair = pairs.setdefault(row['problem_id'], {})
        if method in pair:
            raise ValueError('Duplicate task/method observation')
        if 'generated_token_ids' in row and len(row['generated_token_ids']) != row['generated_tokens']:
            raise ValueError('Generated-token count mismatch')
        if type(row['evaluation']['is_correct']) is not bool:
            raise ValueError('Expected Boolean stored correctness')
        pair[method] = row
    if not pairs or any(set(pair) != {'S0', 'E2'} for pair in pairs.values()):
        raise ValueError('Expected nonempty complete S0/E2 pairs')
    methods = {}
    for method in ('S0', 'E2'):
        rows = [pair[method] for pair in pairs.values()]
        methods[method] = {
            'tasks': len(rows),
            'stored_correct': sum(row['evaluation']['is_correct'] for row in rows),
            'stored_statuses': dict(sorted(Counter(row['evaluation']['status'] for row in rows).items())),
            'generated_tokens': sum(row['generated_tokens'] for row in rows),
            'recurrent_block_calls': sum(row['recurrent_block_calls'] for row in rows),
            'missing_raw_token_ids': sorted(row['problem_id'] for row in rows if 'generated_token_ids' not in row),
        }
    return {
        'methods': methods,
        'rescues': sorted(pid for pid, pair in pairs.items() if pair['E2']['evaluation']['is_correct'] and not pair['S0']['evaluation']['is_correct']),
        'regressions': sorted(pid for pid, pair in pairs.items() if pair['S0']['evaluation']['is_correct'] and not pair['E2']['evaluation']['is_correct']),
        'interpretation': 'Stored-score recount only; raw-token presence does not prove provenance or scoring correctness.',
    }


def main() -> None:
    for jobs in (32, 76):
        path = ROOT / f'reports/colab_l4_deceptive_traps_3000_{jobs}jobs.json'
        print(json.dumps({'report': path.name, **summarize(json.loads(path.read_text()))}, indent=2))


if __name__ == '__main__':
    main()
