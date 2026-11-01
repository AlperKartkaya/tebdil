# Explicit thinking-mode recording verification

Status: **implementation, real smoke and raw audit complete; public release
verification pending**. First sequential point only; no new task pool, passive
stall monitor, intervention or controller experiment has begun.

## Frozen implementation

[Protocol](../docs/prompt_started_reasoning_protocol.md), new code and 25 synthetic/
mini-model/CLI tests were frozen at `1402630` before observations.
`evaluation/reasoning.py` explicitly encodes `enable_thinking=True`, verifies the
actual prompt ends in the assistant's `<think>` opening, and records input and
output separately. It reuses native ledger validation and unchanged v2 integer
grammar only on the final section. No opening tag is inserted into raw output.
Historical runners remain False; hash-bound sources, old scores and splits stay
unchanged. True adds a prompt prefix, not a guarantee of useful or faithful CoT.

Tests cover the opening-in-prompt parser mismatch, exact text preservation,
malformed/repeated tags/EOS, open and unfinished sections, empty reasoning,
offline final-only extraction, nonfinite readout, real mini-model token/call
observer agreement, failure-hook cleanup and overwrite/path-collision refusal.
These fixtures are not new checkpoint outcomes.

## One actual checkpoint execution

The exact pinned Ouro checkpoint/tokenizer, BF16 Apple MPS, four depths,
forced-last-step readout, eager attention and uncached native greedy generation
ran only **the already-observed development question 000**. No gold enters the
generator. No new task difficulty, reserved test, donor or controller was evaluated.

- Explicit opening is in the actual input; generated text starts `Okay, let's see.`,
  not a manufactured `<think>` wrapper.
- **379 generated tokens**, including EOS; `reasoning_status = closed`.
- Generated reasoning and final answer are separated at the actual `</think>`.
  Offline final-only integer assessment recognizes **29**, correct for this question.
- **1,516 physically observed recurrent calls**, **362,324 recurrent token positions**;
  **547.745 seconds** of instrumented generation/validation runtime. Positions are
  not FLOPs; runtime is not optimized inference throughput.
- All 21 source/input identities and checkpoint hashes matched before and after.
  Independent forward proposals, exact prefixes and physical counts reconcile with
  every returned native token and ledger entry. A terminal `finish` exists; no
  failure, retry, output replacement or favorable filtering occurred.

The newly executed continuation equals the earlier False-mode output after its two
opening-tag/newline tokens. This is an observation on one known question, not an
independent sample or evidence True improves reasoning. The old cost is not charged
again or combined into a purported new baseline comparison. No paired efficacy or
speed claim follows from the different prompt/output token accounting.

## Raw evidence

[Report](ouro_prompt_started_reasoning.json), SHA-256:
`ee018b2bb06567f5db76503221c7ee7c0f84cc60d9fe59dd49d88169fde26b57`.

[Append-only forward journal](ouro_prompt_started_reasoning.jsonl), SHA-256:
`33810843a5003c4f482988bb9c0f0cfcda6103a86aa01d7a319d9211321c711b`.

The public report is byte-identical, registered separately in the evidence
manifest. Existing eight run entries and all prior file/evidence metadata stay
unchanged. The release regression reconstructs every prefix and checks text
boundaries, score, journal, source identities and actual work. Recorded audits do
not establish clean-clone or cross-device generation reproducibility.

## Release gate

Before observations, **268 tests passed**, including existing cached real-Ouro
integration and the 25 new fixtures. Scoped Ruff, JavaScript syntax, TypeScript,
export and diff checks passed. The complete local browser matrix passed on a fresh
run; an earlier isolated-page navigation timeout is not counted as a success.
Original CSS and section identities remain protected. Post-run full tests and
local/public evidence verification are recorded below when complete.

The next task/verifier point must wait until this gate's real evidence, docs and
public Pages verification pass. There is still no implemented stall detector,
validated progress metric, recovery result or controller benefit.

After the real run: **269 tests passed**, including the complete raw-evidence
regression and cached real-model integration. Scoped Ruff, JS syntax, TypeScript,
export/diff checks and the full local browser matrix passed. The new download
recombines exact generated reasoning, closing tag, final section and EOS without
inventing output. Public deployment and provenance links remain to be verified.
