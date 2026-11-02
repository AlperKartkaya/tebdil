# Passive reasoning recording verification

Status: **real recording and offline raw audit complete; public release pending**.
The recorder is verified; the **semantic-monitor / controlled-continuation gate is
not cleared**. No new intervention or controller experiment has begun.

## Frozen execution, not a simulated trace

[Protocol](../docs/passive_reasoning_protocol.md), recorder and observer were frozen
at `95ea535` before running the exact first level-0 development CNF/planning pair.
Native True-mode Ouro-1.4B-Thinking, pinned checkpoint/tokenizer, BF16 MPS, four
depths, forced last-step, eager attention, cache-free greedy generation were reused.
No gold/verifier/instance or future output enters the observer. All 31 source/data
identities and checkpoint hashes agree before/after both attempts; a terminal
`finish` and both case finishes exist, no failure, retry, overwrite or filtering.

| Attempt | Tokens | Stop / boundary | Recurrent calls | Recurrent token positions | Candidate repeats | First flag k |
|---|---:|---|---:|---:|---:|---:|
| CNF | 2,048 | budget / reasoning open | 8,192 | 10,530,816 | 38 | 176 |
| Key/door planning | 2,048 | budget / reasoning open | 8,192 | 10,686,464 | 4 | 207 |

Neither attempt emitted EOS or an eligible final answer. Both are **budget
exhausted**, not recognized wrong final answers. Do not extract a successful answer
from their thoughts, relabel truncation as incorrect reasoning, or infer accuracy
from two unfinished development cases. No held-out task or higher-level development
case ran. The old 12/60 arithmetic data and scores remain unchanged.

Total: **16,384 physical recurrent calls**, **21,217,280 recurrent token positions**.
Instrumented generation/ledger-validation time is 15,654.014 + 15,850.606 =
31,504.620 seconds (about 8.75 h). This includes CPU observation transfers and
per-token journal I/O, not model-loading/hash/offline-scoring overhead. It is **not
optimized inference throughput** or FLOPs. No efficiency benefit is measured.

## Actual states and evidence audit

Each of 4,096 token proposals matches the actual native returned token and full
input-prefix ledger. Four last-position input/raw/normalized states are aligned
within that prefix. Normalized-output hashes equal the next depth's input hashes;
positions and typed snapshots at six fixed token indices reconcile exactly.
Only those fixed indices retain actual vectors; other steps retain hashes/scalars,
not all full tensors. Norms/update magnitudes include learned RMS normalization,
not semantic progress or persistent agents across output tokens.

Raw [summary](passive_reasoning_v1/report.json), SHA-256:
`1e63c7b3723a9e5c6347ac9498ca322b2b3d86e87a461fc4df4f1f0a35b749ba`.
It names and hashes both raw case files and append-only per-token journals.
The [run journal](passive_reasoning_v1/run.jsonl) ends with the summary hash.
All six public files are byte-identical copies, separately registered in the
manifest; all prior run/evidence/file metadata is preserved.

Two raw-evidence audits reconstruct every token prefix/position, physical cost,
state chain, typed snapshot hash, scalar norm/update, final score and journal.
The cached pinned tokenizer separately reconstructs raw text, prompt opening,
reasoning boundaries and causal prefix features/flags at every step. Scalars use
small FP32 cross-platform tolerance; hashes are exact. These audits verify the
recorded execution, not clean-clone or cross-device full-generation reproduction.

## What the candidate flags actually establish

[Prefix-only reviews](passive_process_review.md) were written before each final
outcome. Their first public commit `8367bb4` was after CNF completion but while
planning was still running. They are one code-agent exploratory review, **not**
independent human semantic labels, a false-positive-rate estimate or detector validation.

CNF's first flag is shared literal syntax while listing different clauses. Planning's
first flag lists the first ten of eleven real moves correctly: the eight-token suffix
omits the source room and conflates different edges. Neither boundary establishes
stalled search. Planning later repeatedly proposes a shortcut using absent edge
`3->0`; independent BFS and a synthetic offline verifier check expose that concrete
constraint error, without scoring thought text as an actual final answer.

Do not skip to a later, outcome-selected flag, change the frozen threshold, or call
a perturbation at input restatement "stall recovery". The pilot supplies observable
process errors and a warning about this lexical screen, not validated stall selection,
TEBDİL utility, statistical support or a learned controller. Causal monitoring and
baseline completion/budget adequacy need further development before the next gate.

## Verification and release

Before observations, **312 full tests passed**, including five new observer/CLI
checks and actual BF16 three-token passive/plain parity (12 calls per arm), plus the
existing real Ouro integration. Mini-native parity also verifies unchanged weights;
nonfinite states and failed journal callbacks remove all hooks. These are bounded
recorder checks, not full-answer or efficiency equivalence.

After recording, **315 full tests passed in 54.77 seconds**, including both
raw-evidence audits and a process-review regression that keeps a synthetic
constraint check separate from the actual unfinished model record. The hosted-CI
selection also passes locally: 312 tests, 3 real-model-marked tests deselected
(35.17 seconds). Scoped Ruff,
JavaScript syntax checks, TypeScript, frozen-task regeneration, committed-export
checksums/pairing and `git diff --check` pass. No full export regeneration used the
unrelated dirty historical artifacts.

The local `/tebdil/` browser matrix passes all eight historical run selectors,
actual new pilot downloads, selected traces/methods/steps, keyboard navigation,
three recorded comparison cases, error/retry/checksum handling and five widths
(320, 390, 768, 1024, 1440), with no page errors, external requests or failed
responses. Prior manifest contents are unchanged; only six files and one evidence
entry are added. Original CSS/sections and recorded stage stay intact.

Public deployment, deployed bytes and GitHub provenance checks remain pending;
local checks do not establish a successful public release.
