# Reasoning tasks v1 — frozen before model observations

Status: **task/verifier contract implemented; all 18 instances unexecuted**.
This replaces neither the old twelve development observations nor the sixty
reserved arithmetic questions. No difficulty, stall, recovery or efficacy result
follows from generating tasks or solving them with classical algorithms.

## Selection and separation

Six development instances: one per family at each structural level 0/1/2.
Twelve reserved-test instances: two per family/level. The statistical unit is the
problem instance, not a token, depth, candidate solution or seed. This small pool
is a measurement/development prerequisite, not a powered confirmatory benchmark.

`tasks/reasoning.py::build_manifest` uses independent `random.Random` seeds:
82451 (development), 174963 (reserved test), plus 100 × level, 50 for planning,
and replicate index. Freeze exact prompts and instances, not just seed names.
Reject only by the structural rules below, before any model execution. Keep all
subsequent attempts, including correct, wrong, malformed and unfinished outputs;
never select a pool by favorable model outcomes.

Partitions share generators and wording. They are **instance-held-out, not
template-held-out, sealed, or demonstrated out-of-distribution**. Exact instance
and prompt duplication is checked; general graph/Boolean isomorphism is not ruled
out. Public development download excludes reserved-test prompts. The repository
contains the reserved data for reproducibility; no model execution or outcome-based
selection on it is allowed during development.

## Two mechanically checkable families

- **CNF satisfaction:** 4/6/8 variables, 12/18/24 unique three-literal clauses,
  no repeated variable inside a clause. Enumerate all assignments before acceptance.
  Require 1–25% satisfying assignments, all variables present, and neither all-zero
  nor all-one a solution. Accept **any** complete satisfying assignment, not one
  planted/exact-string witness. Independent exhaustive comprehension checks the
  reused `evaluate_cnf_clauses` evaluator against every assignment in every instance.
- **Key/door planning:** 6/8/10 rooms, 1/2/3 key types, directed unit-cost moves.
  Collect keys on entering a room (including start); keys persist and are not consumed.
  BFS over `(room, inventory)` must find a shortest route with at least 4/6/8 moves
  and a room revisit. Without keys the goal is unreachable; removing locks permits
  a shorter route. These are structural screens, not observed search difficulty or
  a guarantee every listed key is needed. The offline verifier walks submitted
  edges with inventory, then checks length against independently computed BFS optimum.
  Accept **any** shortest legal route, including alternative optimal solutions.

There is no model-based grader or verifier feedback during generation. A future
runner receives only `task['prompt']`; instances, solution enumeration and BFS stay
in offline scoring. Prompts already contain the problem's necessary constraints,
not hidden solutions, gold labels, candidate ranking or optimum length.

## Frozen output/measurement grammar

Use the previously verified `enable_thinking=True` encoder and actual token/call
ledger: four depths, forced last-step, greedy, full prefix, cache disabled. Stop
at actual EOS or **2,048 newly generated tokens including EOS**. Do not hide a
budget exhaustion by treating an internal answer as completed.

Only `record.final_text` from a **completed, properly closed** prompt-started
reasoning section is eligible for scoring. Do not search emitted thoughts for a
correct answer. Accept either one whole JSON object, or a single terminal line
`FINAL: { ... }` after optional explanatory prose. That line is case-sensitive,
single-line JSON, unique, and must be last. Markdown fences, `Answer:` instead of
`FINAL:`, trailing prose, multiple markers, duplicate JSON keys, NaN/Infinity,
extra keys, floats and Booleans are rejected, not coerced.

- CNF: exactly `{"assignment": [0, 1, ...]}`, exactly N integer bits in x1…xN order.
- Planning: exactly `{"path": [0, 2, ...]}`, a nonempty list of actual integer rooms.
  Unknown rooms, missing endpoints, illegal/locked moves or a non-optimal route are
  **incorrect**, not format failures, if the JSON schema itself is valid.

Outcome classes are `correct`, `incorrect`, `invalid_format`, `budget_exhausted`.
No partial credit or post-hoc grammar repair. A malformed reasoning boundary is
invalid format; incomplete generation remains budget exhausted even if a closed
section and candidate answer were emitted. Any future relaxation needs a new
version, with original scores and outputs retained.

## Reproduce without inference

```bash
uv run python scripts/freeze_reasoning_tasks.py --check
uv run pytest -q tests/test_reasoning_tasks.py
```

The freezer refuses existing paths without `--check`. Regeneration is byte-checked
against committed files. Reference environment: locked Python 3.12; no new dependency.
[Full pool](../configs/reasoning_v1_tasks.json) SHA-256:
`016c4c8343fa3656825b63f3df6dc283d06f19ffaaba17714c8c163c10590e17`.
[Public development prompts](../site/static/evidence/reasoning_v1_development.json)
SHA-256: `3d5e116d2e47e4bd450fad1f5e9895b26d2dfaa68dd78a584390bc7276b9d002`.

## Gate before the next point

Pass extraction, alternate-valid-answer, exhaustive CNF, key persistence/revisit,
optimality, malformed data, partition/drift and overwrite-refusal tests; production
and public Pages checks must pass. Then implement passive token/depth recording
before new model observations. This contract does not implement or validate a
stall detector. Monitor thresholds and intervention rules are not tuned here.
