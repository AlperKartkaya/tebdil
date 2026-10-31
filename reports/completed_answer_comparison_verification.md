# Gate 2: fixed-method completed-answer comparison

Status: **real technical comparison and public release verification complete**.
Gate 1 was publicly verified first. Code and [protocol](../docs/completed_answer_comparison_protocol.md)
were frozen at `c60dbaf` before observation. Gate 3 has not started; all 60 reserved
test questions remain unexecuted. No learned controller was run.

## Recorded outcome

Actual pinned `ByteDance/Ouro-1.4B-Thinking`, BF16 on MPS, four recurrent depths,
uncached full-prefix greedy generation, shared chat prompt and unchanged final
integer parser. Only `completion_dev_000` was used: `(5 * 6) + 6 - 7`, gold 29.
The already-correct native S0 completion is reused without alteration or rerun.

| Method | Frozen-parser outcome | Tokens | Recurrent calls | Recurrent token positions | Wall seconds |
|---|---|---:|---:|---:|---:|
| Native S0 (recorded gate 1) | Correct | 381 | 1,524 | 362,712 | 541.415 |
| Donor-only B_only | Invalid format | 365 | 2,555 | 587,650 | 989.070 |
| Direct state mixing M0 | Correct | 436 | 3,052 | 810,306 | 1,353.271 |
| Direct update mixing M1 | Correct | 381 | 2,667 | 634,746 | 1,075.983 |
| Random excursion E0 | Correct | 381 | 3,048 | 725,424 | 1,210.930 |
| Donor excursion E1 | Correct | 381 | 3,048 | 725,424 | 1,222.131 |

Every new method reached EOS. All five generated sequences differ from native S0,
including methods with the same token count. Eight additional S0 parity tokens
matched the recorded native prefix, costing 32 calls. Total newly executed calls:
**14,402**; reused S0's 1,524 calls are recorded separately, not treated as free.
No instability fallback was accepted. Physical hooks, scheduler and adapter
counters agreed; checkpoint and all 26 source/input identities stayed unchanged.

**Parser limitation, not demonstrated reasoning failure:** B_only's raw final
answer is `\boxed{29}`. The frozen parser does not recognize that format, so its
record remains `invalid_format`, with `is_correct: false` under that protocol.
It is not dropped or relabeled as a wrong arithmetic answer. This manual format
observation is post-hoc, not a replacement score. Before any utility pilot, freeze
and validate format handling separately; do not silently rescore this campaign.

## Evidence and reproducibility

- [Raw report](ouro_completed_answer_comparison.json): full outputs, token IDs,
  prefix/selected-state hashes, intervention telemetry, source/checkpoint identities.
  SHA-256: `1fee8ed944ab9f81fd698b8ac3010a828c85e79b6dacc386e7667e44bb57c3a2`.
- [Append-only journal](ouro_completed_answer_comparison.jsonl): start, parity,
  every token, all five method completions and terminal finish. No failed/unfinished
  arm or favorable subset has been substituted.
- The byte-identical public report is registered in the static manifest. Historical
  recordings and the original page layout/CSS stay unchanged.
- Reproduce with the [serial runner](../scripts/compare_completed_answers.py) and
  [instructions](../docs/reproduction.md#10-fixed-method-completed-answer-comparison-gate-2).
  A fresh output directory is required; partial evidence is never overwritten.
  This local five-arm run took approximately 98 minutes, not an interactive demo.

## What this does and does not establish

The new gold-free decoder reuses the existing scheduler/operators, chat formatter
and final-answer parser. It retains no earlier-token logits/states, performs no
teacher-forced gold scoring, and records actual per-token interventions and costs.
Ten mini-model/CLI tests cover counts, seeds, prefix hashes, EOS/budget handling,
no gold input, failure-hook cleanup and refusal to overwrite partial evidence.
A separate release test reconstructs every prefix and reconciles the real journal,
parser outcomes, source hashes, manifest and all costs. These checks validate
bounded execution and evidence consistency, not a clean-clone/cross-device rerun.

This is **one already-correct development question**, with no rescue opportunity,
no accuracy estimate, significance test or demonstrated excursion advantage.
E0 and E1 happened to have equal whole-answer calls/lengths here; the protocol does
not guarantee this after prefixes diverge. Recurrent token positions are not FLOPs.
Wall times include scheduler diagnostics/readouts and journaling; native S0 uses
a different instrumentation path. They are not optimized inference latency claims.

B_only emits the perturbed donor and charges both branches; it is not an optimized
independent full-answer sampling/selection baseline. E0 matches input direction
distance, not returned output norm. A later efficacy campaign still needs
completed-answer output-matched ablations, an appropriate independent-selection
baseline, frozen format coverage and a compute-matching design. Earlier fixed-prefix
controls do not substitute for these completed-answer comparisons.

Do not train/evaluate a controller or start reserved-test inference from this
smoke. Establish useful interventions on development data first, then freeze an
independent test and report uncertainty and all outcomes.

## Release verification

- **178 tests passed**, including the cached real-checkpoint adapter integration.
  The new evidence test first failed on the absent report, then passed on the
  actual immutable outputs, without changing any frozen generation/scoring source.
- Scoped Ruff, TypeScript, JavaScript syntax, export checksums and diff checks passed.
- Full local browser matrix passed: all eight recordings, three demos, new raw
  download, keyboard/retry/checksum handling and 320–1440 px layouts; no console,
  network or overflow failures. Expanded evidence was visually inspected.
- All historical manifest entries/run data remain byte-equivalent in content.
- Evidence commit `0dcbdc3`: [CI and Pages deployment passed](https://github.com/AlperKartkaya/tebdil/actions/runs/41254056115).
  The full public browser matrix passed separately. Deployed HTML, app, manifest
  and new raw report match local SHA-256 digests; protocol, findings and raw-journal
  source links return HTTP 200. This closes the technical gate, not the efficacy gate.
