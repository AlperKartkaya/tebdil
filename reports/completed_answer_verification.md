# Gate 1: shared completed-answer evaluation

Status: gate 1 complete. Implementation frozen at `54b3381`; observation and
public evidence committed at `d74a61f`. Real smoke, release and public deployment
verification passed before moving to gate 2.

## What changed

- Reused native full-prefix generation and the existing final-integer parser;
  no historical outputs, scores, decoder, adapter or controller modified.
- Added a shared validator/scorer with explicit correct, incorrect,
  invalid-format and budget-exhausted outcomes. All attempts remain in accuracy's
  denominator. Prefix hashes and physical call ledgers must reconcile before
  scoring; no first-integer or gold-aware fallback.
- Froze twelve development and sixty reserved test arithmetic inputs, exact
  prompts/gold and split-disjoint template IDs. These use existing generator
  templates, not a new externally validated benchmark. Only the preselected first
  development case runs in this gate. No test-set model execution occurs.
- Frozen [protocol](../docs/completed_answer_protocol.md), fail-closed boundary
  fixtures, independent mathematical gold checks and mini native-engine coverage.
- New reports omit duplicated quadratic prefix token lists: exact prefixes are
  reconstructible from prompt/generated IDs and verified against measured hashes.

## Real-checkpoint observation

[Raw report](ouro_completed_answer_smoke.json), [append-only journal](ouro_completed_answer_smoke.jsonl).
One preselected development arithmetic question, `(5 * 6) + 6 - 7`, on the pinned
Ouro-1.4B-Thinking checkpoint. Apple MPS, BF16, four forced depths, no cache,
native greedy generation and the protocol's unchanged chat template/parser.

- **381 generated tokens**, EOS reached before the 2048-token ceiling.
- Completed final region ends with `Answer: 29`; parser recognizes integer 29,
  independently matching the exact arithmetic gold. Reasoning integers are not used.
- **1,524 physical recurrent calls**, two independent hook counters agree.
- **362,712 recurrent token positions**, accounting for actual growing prefixes.
- **541.4 seconds** generation time (about nine minutes). This is one case's
  hardware-specific timing, not a campaign throughput or cloud-speed estimate.
- All five evidence checks passed; frozen source/input and checkpoint identities
  match before/after execution. All 60 reserved test questions remain unexecuted.
- RSS and MPS allocation snapshots are in the raw row; neither is peak memory.

The report's all-attempts accuracy is mechanically 1/1 for this smoke. It is not an
estimate of general accuracy, nor evidence of baseline headroom or reasoning gains.
The verbose answer illustrates why first-integer matching would be inappropriate.

## Verification

Before observation: 165 full-suite tests passed, including real Ouro integration,
plus scoped Ruff, JS syntax and TypeScript. The new public-evidence disclosure test
was observed failing before adding its links, then passed. Final artifact/release
tests now pass: **167 passed**, including real Ouro integration and replay of the
new token/call/scoring evidence. Export checksums/pairing for all eight historical
runs, scoped Ruff, JS syntax, TypeScript and `git diff --check` pass.

Full local browser matrix passed: eight run selections, three recorded demo cases,
widths 320/390/768/1024/1440, keyboard, evidence downloads, missing context,
failure/retry/checksum handling and 116 rendered formulas. No page errors,
failed normal requests, external runtime requests or body overflow. Expanded
evidence and original-layout screenshots visually inspected. Original CSS,
section order and IDs remain unchanged. No WCAG conformance claim follows.

Manifest registration is append-only: old runs, metrics, evidence entries and file
identities are unchanged. Seven pre-existing dirty research artifacts remain
untouched; the exporter was checked, not rebuilt from those inputs.

[CI and Pages deployment passed](https://github.com/AlperKartkaya/tebdil/actions/runs/41245538443).
The full public browser matrix repeated the local pass, including the new recorded
smoke download. Public SHA-256 matches the committed index, app, manifest and
completion report. Protocol, task-manifest and verification GitHub source links
return HTTP 200. Only after these checks was gate 1 closed.

## Interpretation

This gate validates the evaluation instrument, **not TEBDİL's usefulness**. A single
development smoke cannot establish baseline competence, a useful task difficulty,
statistical power, held-out accuracy or an optimal prompt. The existing parser
originated in earlier post-hoc work; it is frozen before this new observation.

Gate 2 still needs fixed intervention/donor semantics, real completed answers,
appropriate controls and measured total costs. Donors initialized with Gaussian
perturbations are not demonstrated semantic alternatives. Equal token budgets
are not compute matching. Controller evaluation remains conditional on efficacy.
