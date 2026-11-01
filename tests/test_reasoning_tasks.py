"""Contract tests written before the new reasoning pool or verifier."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from tebdil.tasks.reasoning import (
    build_manifest, evaluate_reasoning_task, render_prompt, solve_planning,
    validate_manifest,
)
from tebdil.tasks.validators import evaluate_cnf_clauses

ROOT = Path(__file__).resolve().parents[1]


def record(final, *, completed=True, boundary='closed'):
    return dict(completed=completed, reasoning_status=boundary, final_text=final,
                reasoning_text='FINAL: {"assignment": [0, 0, 0, 0]}')


def cnf():
    return dict(family='cnf', instance={'num_vars': 3, 'clauses': [[1, 2], [-1, 3]]})


def planning():
    return dict(family='key_planning', instance=dict(num_rooms=4, start=0, target=3,
                room_keys=[0, 1, 0, 0], edges=[[0, 1, 0], [1, 0, 0],
                [0, 2, 1], [2, 3, 0], [1, 2, 1]]))


def test_all_satisfying_assignments_accepted_not_only_one_witness():
    for bits in itertools.product((0, 1), repeat=3):
        expected = evaluate_cnf_clauses(cnf()['instance']['clauses'], dict(enumerate(bits, 1)))
        got = evaluate_reasoning_task(cnf(), record('FINAL: '+json.dumps({'assignment': bits})))
        assert got['format_valid']
        assert got['status'] == ('correct' if expected else 'incorrect')


def test_planning_accepts_any_shortest_legal_route_and_respects_keys():
    task = planning()
    assert len(solve_planning(task['instance'])) == 4
    assert evaluate_reasoning_task(task, record('{"path":[0,1,2,3]}'))['status'] == 'correct'
    assert evaluate_reasoning_task(task, record('{"path":[0,2,3]}'))['status'] == 'incorrect'
    assert evaluate_reasoning_task(task, record('{"path":[0,1,0,2,3]}'))['status'] == 'incorrect'
    task['instance']['edges'].append([1, 3, 1])
    assert evaluate_reasoning_task(task, record('{"path":[0,1,3]}'))['status'] == 'correct'


def test_planning_requires_return_to_previous_room_with_new_inventory():
    task = planning()['instance']
    task['edges'].remove([1, 2, 1])
    assert solve_planning(task) == [0, 1, 0, 2, 3]
    task['room_keys'] = [0]*4
    assert solve_planning(task) is None


@pytest.mark.parametrize('text', [
    '', 'Answer: {"assignment":[1,0,1]}', '```json\n{"assignment":[1,0,1]}\n```',
    'FINAL: {"assignment":[1,0,1]}\nmore',
    'FINAL: {"assignment":[1,0,1]}\nFINAL: {"assignment":[1,0,1]}',
    '{"assignment":[1,0,1],"assignment":[1,0,1]}',
    '{"assignment":[true,0,1]}', '{"assignment":[1.0,0,1]}',
    '{"assignment":[2,0,1]}', '{"assignment":[1,0]}',
    '{"assignment":[1,0,1],"explanation":"x"}', '{"assignment":[NaN,0,1]}',
    '{"path":[0,1,3]}',
])
def test_strict_final_format(text):
    assert evaluate_reasoning_task(cnf(), record(text))['status'] == 'invalid_format'


@pytest.mark.parametrize('boundary', ['open', 'missing_close', 'malformed'])
def test_never_scores_thought_or_unclosed_reasoning(boundary):
    assert evaluate_reasoning_task(cnf(), record('{"assignment":[1,0,1]}', boundary=boundary))['status'] == 'invalid_format'
    assert evaluate_reasoning_task(cnf(), record('{"assignment":[1,0,1]}', completed=False))['status'] == 'budget_exhausted'


def test_terminal_marker_allows_explanation_without_reading_thought():
    assert evaluate_reasoning_task(cnf(), record('Checked the constraints.\nFINAL: {"assignment":[1,0,1]}'))['status'] == 'correct'
    assert evaluate_reasoning_task(cnf(), record('No final answer.'))['status'] == 'invalid_format'


@pytest.mark.parametrize('path', [[0,99,3], [1,2,3], [0,1,2], [0,3]])
def test_illegal_paths_are_formatted_but_wrong(path):
    assert evaluate_reasoning_task(planning(), record(json.dumps({'path':path})))['status'] == 'incorrect'


@pytest.mark.parametrize('path', [[False,1,3], [0.0,1,3], '0,1,3', []])
def test_malformed_paths(path):
    assert evaluate_reasoning_task(planning(), record(json.dumps({'path':path})))['status'] == 'invalid_format'


def test_frozen_manifest_reproduces_and_has_no_observations():
    manifest = build_manifest()
    assert manifest == json.loads((ROOT/'configs/reasoning_v1_tasks.json').read_text())
    validate_manifest(manifest)
    assert [len(manifest['partitions'][s]) for s in ('development','reserved_test')] == [6,12]
    for split, tasks in manifest['partitions'].items():
        for task in tasks:
            assert task['split'] == split and task['prompt'] == render_prompt(task)
            assert set(task) == {'problem_id','split','family','structural_level','instance','prompt'}
            if task['family'] == 'cnf':
                inst = task['instance']
                outcomes=[]
                for bits in itertools.product((0,1), repeat=inst['num_vars']):
                    outcome = evaluate_reasoning_task(task, record(json.dumps({'assignment':bits})))
                    expected = evaluate_cnf_clauses(inst['clauses'], dict(enumerate(bits,1)))
                    assert (outcome['status']=='correct') == expected
                    outcomes.append(expected)
                assert 1 <= sum(outcomes) <= 2**(inst['num_vars']-2)
                assert not outcomes[0] and not outcomes[-1]
            else:
                path = solve_planning(task['instance'])
                assert len(path) > len(set(path))
                assert evaluate_reasoning_task(task, record(json.dumps({'path':path})))['status']=='correct'


@pytest.mark.parametrize('mutation', ['duplicate', 'prompt', 'partition', 'malformed', 'missing'])
def test_manifest_rejects_drift(mutation):
    data=copy.deepcopy(build_manifest())
    rows=data['partitions']['development']
    if mutation=='duplicate':
        rows[1]=copy.deepcopy(rows[0])
    elif mutation=='prompt':
        rows[0]['prompt']+=' changed'
    elif mutation=='partition':
        rows[0]['split']='reserved_test'
    elif mutation=='missing':
        rows.pop()
    else:
        rows[0]['instance']['num_vars']=True
    with pytest.raises(ValueError):
        validate_manifest(data)


def test_public_prompts_are_development_only():
    public=json.loads((ROOT/'site/static/evidence/reasoning_v1_development.json').read_text())
    data=build_manifest()
    expected=[{k:t[k] for k in ('problem_id','family','structural_level','prompt')}
              for t in data['partitions']['development']]
    assert public == dict(protocol='reasoning_tasks_v1', status='unexecuted_task_prompts', tasks=expected)


def test_equal_length_alternative_plans_and_start_key():
    task=planning()
    task['instance']['edges'] += [[0,2,0], [1,3,0]]
    for path in ([0,1,3], [0,2,3]):
        assert evaluate_reasoning_task(task, record(json.dumps({'path':path})))['status']=='correct'
    task['instance']['room_keys']=[1,0,0,0]
    assert evaluate_reasoning_task(task, record('{"path":[0,2,3]}'))['status']=='correct'


def test_freezer_refuses_existing_output_without_mutation():
    import subprocess
    path=ROOT/'configs/reasoning_v1_tasks.json'
    before=path.read_bytes()
    run=subprocess.run(['uv','run','python','scripts/freeze_reasoning_tasks.py'],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode != 0 and 'never overwrite' in run.stderr
    assert path.read_bytes()==before
