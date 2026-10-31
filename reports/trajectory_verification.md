# Point B — trajectory semantics and baseline calibration

2026-10-31. **In progress: completed-generation follow-up remains a gate.**
Point A was committed as `e0728b3`, deployed by [run 41219360287](https://github.com/AlperKartkaya/tebdil/actions/runs/41219360287),
and passed the full browser script on the public GitHub Pages URL.

## Completed checks

- New inference records identify encoded prompts, generated IDs, actual per-token
  prefix SHA-256/length, branch seed, branch lifetime, output budget and stop reason.
- Regression tests verify re-forking at depth 1 for each new prefix; a branch is not
  a persistent agent across generated tokens. EOS now stops even if decoded text
  is empty. An explicit diagnostic chat mode permits reasoning newlines.
- New FLOP estimates sum actual prefix lengths instead of charging all generated
  tokens at the initial prompt length. They remain estimates, not measured FLOPs.
- `uv run pytest -q -m 'not real_model'`: **102 passed, 1 deselected**. Scoped Ruff,
  JavaScript syntax, TypeScript, export checksums/pairing and diff checks pass.
- Real checkpoint calibration completed nine evaluations on the same three pilot
  problems: seed 20261031, BF16, four-depth forced-final readout. Checkpoint files
  were rehashed against committed provenance. All measured inference/scoring
  counters reconcile; committed tests reconstruct every recorded token prefix.
- Raw/2 and raw/64 each produced 1/3 correct integer-only responses. Arithmetic
  and graph kept the same wrong answers after reaching a newline. Chat/128
  produced explicit `<think>` content but no completed final answer on any case.
- Original experiment shards, scores, styles and seven pre-existing modified
  research artifacts were not regenerated or overwritten.

## Open completion gate

`scripts/complete_ouro_baseline.py` follows up **all three** unfinished chat cases,
not a correctness-selected subset. It uses native greedy, cache-free generation,
forced depth 4 and a 2048-token ceiling. Forward hooks count real normalization
boundaries and record input-prefix hashes. It compares the first recorded tokens
with the adapter and only accepts a final integer after EOS and outside thought.
Mini-model generation/parsing tests pass; those are not real completion evidence.

Until this follow-up is finished, inspected and published, point B is not complete
and point C is not started. The initial calibration is preserved unchanged in
[JSON](ouro_baseline_calibration.json); it is not rewritten to improve its outcomes.

## Limits

Three pilot problems cannot establish baseline competence, optimal prompting,
generalization, statistical improvement or intervention benefit. Native completion
and the old raw-prompt interventions must not be compared as a fair efficacy
experiment. New controller training and all method comparisons need one common
frozen prompt/completion/scoring protocol before publication.
