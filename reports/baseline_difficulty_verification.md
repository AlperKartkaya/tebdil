# Development baseline difficulty pilot

Status: **real serial run, raw-evidence audit and public release verification
complete**. Measurement v2 was completed and publicly verified first.
The [protocol](../docs/baseline_difficulty_protocol.md), runner and nine tests were
frozen at `0c1719e` before new observations. No subagents or cloud resources.

## What ran

All twelve existing development questions in fixed ID order. Question 000 reuses
its unchanged gate-1 native S0 output/cost; questions 001–011 are **eleven new real
native completions**, not twelve new runs. All reached EOS under the 2048-token
ceiling. No retry, outcome filtering, intervention, controller or reserved-test
inference occurred. All sixty reserved test questions remain unexecuted.

Pinned `ByteDance/Ouro-1.4B-Thinking`, revision
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, BF16 Apple MPS, four recurrent depths,
forced last-step readout, eager attention, uncached full-prefix greedy decoding.
Checkpoint files and 23 source/input hashes matched before and after. Generation
received the question, not the gold answer or task metadata. V2 scoring was offline;
v1 fields remain separately recorded. No historical output or score was changed.

## Outcomes, not an arithmetic-accuracy claim

| Template | Attempts / EOS | Recognized correct | Recognized incorrect | Invalid format | Budget exhausted |
|---|---:|---:|---:|---:|---:|
| Affine cascade | 4 / 4 | 4 | 0 | 0 | 0 |
| Linear equation | 4 / 4 | 3 | 0 | 1 | 0 |
| Modular product | 4 / 4 | 3 | 0 | 1 | 0 |
| **All preselected attempts** | **12 / 12** | **10** | **0** | **2** | **0** |

The stored `accuracy_all_attempts = 10/12` is the frozen completion-format-plus-
correctness score, **not arithmetic accuracy**. No attempt is dropped, and tokens
are not independent samples. No confidence interval or significance claim is made
from this convenience sample of twelve questions across three known templates.

Post-hoc manual format inspection, **not a rescoring**:

- `completion_dev_005` ends with prose: “Both methods confirm the remainder is
  **4**.” The value matches gold, but terminal prose is outside frozen v2 grammar.
- `completion_dev_010` ends with `x = 13`. This value matches gold, but assignments
  are outside frozen v2 grammar.

Both remain `invalid_format`. They are not demonstrated wrong arithmetic or
verified rescue opportunities. The parser was not broadened during or after this
run. No recognized incorrect answer was observed; this does not prove the pool has
no possible errors or that TEBDİL cannot help on harder reasoning tasks.

## Actual instrumented costs

| Cost | Eleven new completions | Reused question 000 | Combined recorded evidence |
|---|---:|---:|---:|
| Output tokens, including EOS | 6,667 | 381 | 7,048 |
| Physical recurrent-block calls | 26,668 | 1,524 | 28,192 |
| Recurrent token positions | 11,392,732 | 362,712 | 11,755,444 |
| Per-completion wall seconds | 16,168.987 | 541.415 | 16,710.402 |

The reused cost was incurred earlier, not again in this campaign. New per-question
runtime totals about 4.49 hours. Prefix reconstruction, observers and synchronous
journal writes add overhead; these are instrumented pipeline costs, not optimized
inference latency or FLOPs. Checkpoint loading and final auditing are not included
in the per-completion timing sum. The final modular question used 1,153 tokens;
none reached the budget ceiling.

## Immutable evidence and audit

- [Raw report: all twelve outputs, prefixes, scores and source/checkpoint hashes](ouro_baseline_difficulty.json), SHA-256
  `35bf17b08b46200d60944f8323627a9725ea363596784b7df84a4ed8f14e53ff`.
- [Append-only journal: all eleven new native forward observations and terminal finish](ouro_baseline_difficulty.jsonl), SHA-256
  `6cb37ce1629ac1a3eee1537b480576c74bc7ee5a0d85b6f55ec7f4e96c544eb2`.
- The public report is byte-identical and registered in the existing checksummed
  evidence manifest. Eight historical run entries and prior evidence stay intact.

The finished campaign has all twelve completion/reuse records, eleven attempt
starts, no failure event and a final `finish`. A release regression reconstructs
**every** prefix from the recorded prompt and generated IDs, verifies independent
greedy proposals equal actual returned native tokens, and reconciles each
four-depth call ledger, campaign total, v1/v2 score and per-template summary.
Mini-model fixtures are not counted as new checkpoint runs. Recorded ledger audits
are not a clean-clone or cross-device model-generation reproduction.

## Decision and stop boundary

The prespecified rule returns **`headroom_unresolved_format_or_truncation`** because
of two format misses; no truncation occurred. The baseline execution point is
finished, but it **does not clear the intervention-utility prerequisite**.

Do not spend on rescue comparisons or train a controller from this easy,
format-limited pool. First freeze a separate generation/measurement contract and
harder development tasks, with an independently reserved matching test set.
Preserve this run, all scores and the unexecuted existing test set. Establish
interpretable baseline failures before a rescue pilot, and retain correct cases
in later comparisons to measure degradation. Useful interventions must precede
learned-controller evaluation. No TEBDİL efficacy or publishable generalization
finding follows from this diagnostic.

**Task-selection correction:** these simple arithmetic cases did not test stalled
multi-step reasoning. Do not generalize their ceiling to TEBDİL's research question.
The [next study design](../docs/reasoning_stall_study.md) separates suspected CoT
stalls from latent depth-update norms, and recovery from efficiency at maintained
accuracy. A failed baseline is needed for a rescue claim, not for every possible
efficiency hypothesis. Monitoring and task/verifier implementation remain pending;
this interpretation does not amend the frozen pilot decision or any score.

## Release verification

241 tests passed before new model observations. After completion, **242 tests
passed**, including the cached real-Ouro excursion/negative-control integration
and the new complete journal/prefix audit. Scoped Ruff, JavaScript syntax,
TypeScript, static-export checksums/pairing and `git diff --check` passed.

The full local browser matrix passed eight recordings, three recorded demo cases,
the new twelve-record download, keyboard controls, retry/checksum failure paths
and 320–1440 px layouts, with no console, network or overflow errors. The expanded
evidence disclosure and mobile layout were inspected visually. Original CSS,
section identities, historical run entries and all previous evidence metadata are
unchanged.

Evidence release `56a4eab`: [CI and Pages deployment passed](https://github.com/AlperKartkaya/tebdil/actions/runs/41293024936).
The full public browser matrix passed independently, including the new baseline
download. Deployed HTML, app, manifest and baseline report match local bytes;
protocol, findings, raw report and journal provenance links return HTTP 200.
This closes the baseline execution/release point, **not** the intervention-utility
or publication-efficacy gate.
