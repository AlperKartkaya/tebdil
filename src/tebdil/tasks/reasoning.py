"""Frozen instance-held-out reasoning tasks; no model observations or oracle runtime."""
from collections import deque
import itertools
import json
import random
import re

from tebdil.tasks.validators import evaluate_cnf_clauses


def solve_planning(instance):
    """Independent BFS over (room, key inventory), returning one shortest witness."""
    start = instance['start']
    initial = (start, instance['room_keys'][start])
    queue, seen = deque([(initial, [start])]), {initial}
    while queue:
        (room, keys), path = queue.popleft()
        if room == instance['target']:
            return path
        for source, destination, required in instance['edges']:
            if source != room or required & keys != required:
                continue
            state = destination, keys | instance['room_keys'][destination]
            if state not in seen:
                seen.add(state)
                queue.append((state, path + [destination]))
    return None


def _cnf(rng, level):
    n = 4 + 2*level
    for _ in range(10000):
        clauses = set()
        while len(clauses) < 3*n:
            variables = rng.sample(range(1, n+1), 3)
            clauses.add(tuple(sorted(v*rng.choice((-1, 1)) for v in variables)))
        clauses = [list(c) for c in sorted(clauses)]
        solutions = [bits for bits in itertools.product((0, 1), repeat=n)
                     if all(any(bits[abs(lit)-1] == int(lit > 0) for lit in c) for c in clauses)]
        if (1 <= len(solutions) <= 2**(n-2) and (0,)*n not in solutions and (1,)*n not in solutions
                and {abs(lit) for c in clauses for lit in c} == set(range(1, n+1))):
            return dict(num_vars=n, clauses=clauses)
    raise ValueError('CNF structural rejection limit reached')


def _planning(rng, level):
    n, key_count = 6+2*level, 1+level
    for _ in range(10000):
        keys = [0]*n
        for bit, room in enumerate(rng.sample(range(1, n-1), key_count)):
            keys[room] = 1 << bit
        edges = [[u,v,rng.choice([0,0]+[1 << k for k in range(key_count)])]
                 for u in range(n) for v in range(n) if u != v and rng.random() < .25]
        instance = dict(num_rooms=n, start=0, target=n-1, room_keys=keys, edges=edges)
        path = solve_planning(instance)
        unlocked = solve_planning({**instance, 'edges': [[u,v,0] for u,v,_ in edges]})
        no_keys = solve_planning({**instance, 'room_keys': [0]*n})
        if (path and len(path)-1 >= 4+2*level and len(path) > len(set(path))
                and unlocked and len(unlocked) < len(path) and no_keys is None):
            return instance
    raise ValueError('Planning structural rejection limit reached')


def render_prompt(task):
    """Only this question string is supplied to generation; no solution/witness."""
    inst = task['instance']
    if task['family'] == 'cnf':
        formula = ' AND '.join('('+' OR '.join(f'{"NOT " if v < 0 else ""}x{abs(v)}' for v in c)+')'
                               for c in inst['clauses'])
        question = (f"Find any satisfying Boolean assignment for x1 through x{inst['num_vars']}. "
                    f"1 means true, 0 means false. Formula: {formula}. "
                    'The assignment array must list every variable in increasing index order.')
        example = '{"assignment": [0, 1, ...]}'
    else:
        rooms = '; '.join(f'{room}: '+(', '.join(f'K{k+1}' for k in range(3) if mask & (1 << k)) or 'none')
                          for room, mask in enumerate(inst['room_keys']))
        edges = '; '.join(f'{u}->{v} (requires '+(', '.join(f'K{k+1}' for k in range(3) if mask & (1 << k)) or 'none')+')'
                          for u,v,mask in inst['edges'])
        question = (f"Find a shortest legal route from room {inst['start']} to room {inst['target']}. "
                    f"Rooms are 0 through {inst['num_rooms']-1}. Each directed move costs 1. "
                    'Start with no keys. Collect keys automatically on entering a room, including the start. '
                    'Keys are reusable and never consumed. You may revisit rooms. '
                    f'Keys in rooms: {rooms}. All allowed directed moves: {edges}. '
                    'List every visited room including start and target. Any shortest legal route is accepted.')
        example = '{"path": [0, 2, ...]}'
    return question+' Reason through the constraints. End your final answer with exactly one line FINAL: '+example+'. Replace the example with valid JSON using actual integers, no ellipsis, no extra keys or text after that line.'


def build_manifest():
    partitions = {}
    for split, count, seed in [('development', 1, 82451), ('reserved_test', 2, 174963)]:
        rows = []
        for level in range(3):
            for family, make in [('cnf', _cnf), ('key_planning', _planning)]:
                for replicate in range(count):
                    rng = random.Random(seed + level*100 + (50 if family == 'key_planning' else 0) + replicate)
                    task = dict(problem_id=f'reasoning_v1_{split}_{family}_{level}_{replicate}',
                                split=split, family=family, structural_level=level,
                                instance=make(rng, level))
                    task['prompt'] = render_prompt(task)
                    rows.append(task)
        partitions[split] = rows
    return dict(protocol='reasoning_tasks_v1', generator_seed=82451,
                split_unit='instance; shared generators, not template-held-out', partitions=partitions)


def validate_manifest(manifest):
    """Accept only the exact deterministic pool, including partition and prompt."""
    if manifest != build_manifest():
        raise ValueError('Reasoning manifest differs from frozen deterministic contract')
    rows = [t for tasks in manifest['partitions'].values() for t in tasks]
    identities = [json.dumps((t['family'],t['instance']), sort_keys=True) for t in rows]
    if len(set(identities)) != len(rows) or len({t['prompt'] for t in rows}) != len(rows):
        raise ValueError('Duplicate reasoning instance or prompt')


def development_prompts(manifest):
    return dict(protocol='reasoning_tasks_v1', status='unexecuted_task_prompts',
                tasks=[{k: t[k] for k in ('problem_id','family','structural_level','prompt')}
                       for t in manifest['partitions']['development']])


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def _unique_object(pairs):
    if len(dict(pairs)) != len(pairs):
        raise ValueError('Duplicate JSON key')
    return dict(pairs)


def evaluate_reasoning_task(task, record):
    """Offline final-only scoring; never pass this task/validator into a monitor."""
    result = dict(format_valid=False, extracted_answer=None, status='invalid_format')
    if not record['completed']:
        return {**result, 'status': 'budget_exhausted'}
    if record['reasoning_status'] != 'closed' or not isinstance(record['final_text'], str):
        return result
    text = record['final_text'].strip()
    if 'FINAL:' in text:
        match = re.search(r'(?:^|\n)FINAL: *(\{[^\n]*\})\s*$', text)
        if text.count('FINAL:') != 1 or not match:
            return result
        text = match.group(1)
    try:
        answer = json.loads(text, object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):
        return result
    family, inst = task['family'], task['instance']
    key = 'assignment' if family == 'cnf' else 'path'
    if not isinstance(answer, dict) or set(answer) != {key} or not isinstance(answer[key], list):
        return result
    values = answer[key]
    if not values or any(type(v) is not int for v in values):
        return result
    if family == 'cnf':
        if len(values) != inst['num_vars'] or any(v not in (0,1) for v in values):
            return result
        correct = evaluate_cnf_clauses(inst['clauses'], dict(enumerate(values, 1)))
    elif family == 'key_planning':
        correct = values[0] == inst['start'] and values[-1] == inst['target']
        keys = inst['room_keys'][inst['start']]
        for u,v in zip(values,values[1:]):
            legal = any(a == u and b == v and required & keys == required for a,b,required in inst['edges'])
            if not legal or not 0 <= v < inst['num_rooms']:
                correct = False
                break
            keys |= inst['room_keys'][v]
        shortest = solve_planning(inst)
        correct = correct and shortest is not None and len(values) == len(shortest)
    else:
        raise ValueError('Unknown reasoning family')
    return dict(format_valid=True, extracted_answer=answer, status='correct' if correct else 'incorrect')
