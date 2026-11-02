# Reasoning CoT sensitivity: verification and findings

Status: actual probe, raw audit and public release verified. No rescue claim.
[Protocol](../docs/reasoning_probe_protocol.md), source and six mini-native tests
were frozen at `09b7f1f` before any new intervention observation. The full suite
passed 321 tests including real-checkpoint tests. Freeze CI/Pages passed
[41408860860](https://github.com/AlperKartkaya/tebdil/actions/runs/41408860860).
This is a separate diagnostic, not a claim that the earlier semantic-monitor gate
was cleared. No validated stall detector, held-out efficacy result or controller.

## What existed before this probe

Two real True-mode Ouro development attempts, each 2048 output tokens, both
unfinished/open reasoning without EOS/final answer. No reasoning intervention
comparison or observed verified stalled-to-solved case existed. Historical arithmetic
replays are not evidence of CoT recovery on these tasks. Earlier exact-repeat flags
were input organization, not established semantic stalls. Keep those results intact.

## Task suitability is still a question, not a conclusion

Independent task inspection, not new model inference:

- CNF: four variables, twelve clauses, exactly two satisfying assignments out of
  sixteen. Any satisfying assignment is accepted, not one particular gold witness.
- Planning: six rooms, eleven directed moves, one key. Independent BFS returns a
  five-move shortest route `0->3->1->0->4->5`. Any shortest legal route is accepted.

These are low-structural-size instances, not measurements of Ouro difficulty. A
long trace or unfinished 2048-token budget does not establish a semantic stall,
wrong final answer, excessive/easy difficulty or irrelevance of the task family.
Later planning text invents a `3->0` edge, demonstrating a process constraint error,
not a final-answer score or an oracle-approved trigger time.

Before scaling task size, establish ordinary completion/budget adequacy and a
causal process-selection contract on development. Preserve the frozen ladder and
reserved sets. Do not tune on held-out answers or silently shorten/coach a prompt
and compare it to the old original-question baseline. A new prompt/budget requires
separate versioned arms with their own baseline and costs.

## Acceptance before a before/after success story

A changed token/logit/CoT passage is not improved reasoning. Require an actual
EOS-terminated, verifier-accepted final answer, comparable ordinary continuation,
retained failures and degradation, and declared compute budgets. A selected
reused-development prefix cannot establish a general rescue rate. Short diagnostic
windows are not replacements for a full-answer efficacy campaign. If no next token
changes, report that scoped negative result and do not fabricate a continuation.

## Observed result (all predeclared contexts retained)

Twelve contexts: both development cases at k=0,127,255,511,1023,2047. All S0
proposals and last-position normalized state hashes exactly match the original
native recordings. All receiving/donor inputs remain unchanged by the operators.

| Method | Changed greedy next token vs S0 | Logit L2 change range |
|---|---:|---:|
| Direct state mixing M0 | 0 / 12 | 18.003–141.355 |
| Linear update mixing M1 | 0 / 12 | 11.018–32.366 |
| Excursion-and-return E1 | 0 / 12 | 12.535–73.327 |
| Output-matched random correction | 0 / 12 | 30.725–82.045 |

E1's actual hidden-state displacement from the ordinary depth-three state is
46.664–174.915 (whole-prefix Frobenius norm, **not semantic progress**). The visit
and returned states are real, finite and not instability fallbacks. Maximum
realized BF16 random output-norm mismatch is 0.0291%, retained rather than assuming
exact matching after casting. Logit L2 spans the vocabulary, not the winning margin.
The native baseline top-two logit margin at these points ranges 1.25–12.125.

The intervention is not a no-op on hidden states/logits, but the winning token
stays unchanged. In this cache-free greedy **single-intervention** setup, there is
no extra hidden state carried into the following token if the emitted token is
unchanged. This does not prove zero CoT change under every timing, donor, strength,
repeated intervention or stochastic decoding policy. A Gaussian latent donor is
not evidence of an independently generated alternative semantic reasoning strategy.
Do not tune those parameters on these twelve outcomes until a dramatic example appears.

The frozen conditional rule therefore selects **zero continuation windows**.
No new full CoT continuation or final-answer evaluation was run. There is no
stuck-to-solved observation, rescue rate, statistical test or efficacy conclusion.
Do not infer task unsuitability or a failed general TEBDİL hypothesis from this
small development sensitivity probe. In particular, these are two reused problems,
not twelve independent benchmark examples.

Actual new shared diagnostic work: **192 physical recurrent calls**,
**178,944 recurrent token positions**, **367.016 seconds** after model loading, including recording overhead
and final identity checks. Reference construction/output matching is charged;
these are not separately executed compute-matched arms. The 16,384 previously
executed baseline calls remain separately reported, not zero-cost inference.

Raw [report](reasoning_cot_probe_v1/report.json) and [append-only journal](reasoning_cot_probe_v1/journal.jsonl)
are copied byte-for-byte to Pages and registered in the checksum manifest. The
journal ends with the report hash. Source, input and checkpoint identities match
before/after. No held-out task, retried condition, new controller or outcome-selected
later prefix was used. All historical data and score artifacts remain untouched.

## Site and verification

The first recorded panel now exposes reasoning tasks, all six fixed token indices,
all comparison methods, raw shared CoT and actual paired token proposals. It
explicitly reports the absent rescue/continuation rather than synthesizing a
before/after success story. Historical arithmetic/graph replays remain accessible
below and original CSS is unchanged. The two record loaders fail/retry separately;
one failed recording cannot hide or overwrite the other evidence surface.

Verified before release:

- Full local suite: **324 passed** in 60.15 seconds, including cached real-checkpoint
  integration/parity tests and recorded-token decoding. The new actual intervention
  probe is the separately retained model execution above, not inferred from tests.
- Hosted-compatible suite: **320 passed, four real-model tests deselected**.
- Scoped Ruff, JavaScript syntax, TypeScript, task-freeze identity and static Pages
  export `--check` passed. No historical experiment export was regenerated.
- Local browser matrix: both tasks, all six indices and all four selectable
  comparisons match raw evidence; actual shared CoT, token IDs, keyboard disclosure,
  independent loader failure/retry and checksum rejection passed. All eight older
  run selectors and their comparison/download flows remain functional.
- Responsive checks passed at 320, 390, 768, 1024 and 1440 px; desktop and fresh
  390 px CoT screenshots were visually inspected. No horizontal overflow, page
  errors, unexpected external requests or failed normal requests were observed.
  Original CSS is unchanged.

Public release verified for evidence/viewer commit `10d3a69`:

- [CI/Pages run 41410963859](https://github.com/AlperKartkaya/tebdil/actions/runs/41410963859):
  verify, package and deploy all passed for that exact revision.
- The full browser matrix passed on
  [the deployed site](https://alperkartkaya.github.io/tebdil/), including all new
  task/index/method selections and both independent failure/retry/checksum paths.
  All five widths passed; no page errors, failed normal requests or unexpected
  external requests were observed.
- Thirteen deployed files match local committed bytes exactly: HTML, application
  JS, unchanged CSS, manifest, development prompts, both probe files and all six
  passive-recording files. The protocol, findings and frozen-source provenance
  links return HTTP 200.

This closes this diagnostic's release, not the research efficacy gate. No repeated
intervention, larger task, alternative donor or new decoding-policy sweep was run.
The next prerequisite remains baseline completion/budget adequacy and a separately
frozen decoder-visible intervention/causal-selection contract, not selecting a
more dramatic trace from this development pair.
