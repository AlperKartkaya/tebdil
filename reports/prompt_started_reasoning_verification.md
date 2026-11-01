# Explicit thinking-mode recording verification

Status: **implementation and synthetic tests complete; real smoke and public
release verification pending**. First sequential point only; no task-set creation,
passive stall monitor, intervention or controller experiment has begun.

[Protocol](../docs/prompt_started_reasoning_protocol.md). New
`evaluation/reasoning.py` explicitly encodes `enable_thinking=True`, verifies the
actual prompt ends in the assistant's `<think>` opening, and records prompt and
generated text separately. It reuses native ledger validation and the unchanged
v2 grammar only on the generated final section. The opening tag is never inserted
into raw output. Historical runners stay False and old results stay unchanged.

Synthetic fixtures cover the known opening-in-prompt parser mismatch, exact text
preservation, malformed/repeated tags/EOS, open or unfinished sections, empty
reasoning, gold-free boundary parsing, nonfinite readout, real mini-model token/
physical-call observer agreement, hook cleanup and existing-evidence preservation.
These fixtures are not new checkpoint outcomes or proof of CoT faithfulness.

The frozen smoke will execute only the already-observed development question 000
with True-mode native generation, exclusive JSON/journal and pre/post source and
checkpoint identities. No reserved test, paired False-mode comparison or stall
recovery is evaluated. A partial journal is not a completed answer. No task/verifier
point may begin until this gate's real evidence, tests, docs and Pages pass.

Before new observations: **268 tests passed**, including the existing cached
real-model integration and 25 new synthetic/mini-model/CLI cases. Scoped Ruff,
JavaScript syntax, TypeScript, export checksums/pairing and diff checks passed.
The full local browser matrix passed on a fresh run; an earlier isolated-page
navigation timed out and is not counted as a successful check. No application
change was needed for that retry. Original CSS/section identities remain protected.
