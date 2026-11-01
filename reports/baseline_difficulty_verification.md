# Development baseline difficulty pilot

Status: **real serial run in progress; no completed campaign result yet**.
Measurement v2 was completed and publicly verified first. The protocol, runner
and tests were frozen at `0c1719e` before new model observations.

[Frozen protocol](../docs/baseline_difficulty_protocol.md). All twelve development
questions are selected in fixed order. `completion_dev_000` reuses its previously
observed native S0 output/cost; only the remaining eleven questions execute new
native generation. No intervention, controller or reserved-test inference occurs.

The existing pinned native engine, chat prompt, four-depth forced-last-step
readout, BF16 MPS, uncached greedy decoding and 2048-token ceiling stay unchanged.
V2 is applied offline; v1 evaluation is retained separately. Every attempt stays
in the denominator. Independent forward proposals/prefixes/call counts must match
the actual returned native tokens and ledger; a partial proposal is not an answer.

**241 tests passed** before observations, including nine new mini-model/CLI tests
for observer/native agreement, EOS/budget stops, nonfinite readout rejection,
failure-hook cleanup, difficulty decisions and preserving existing directories.
These fixtures do not verify the eleven real completions. Scoped Ruff, export,
JavaScript syntax, TypeScript and diff checks also passed.

Local directory: `.research/baseline-difficulty-20261101/`; sibling `.pid` and
`.log` identify the process. The append-only journal retains partial progress or
failure, and each finished question gets an exclusive JSON. Only a final report
and terminal `finish` event represent campaign completion. Do not restart, skip
failures or call partial results a baseline accuracy estimate. This may take hours.

After completion, audit every prefix/proposal, count, score, template summary and
source/checkpoint identity. Publish raw evidence and verify public Pages before
moving on. If all twelve answers are correct, there is no observed rescue
opportunity on this pool, not evidence that TEBDİL cannot work. Format/budget
failures do not automatically establish wrong arithmetic. This convenience
development pilot never establishes generalization, intervention utility or power.
