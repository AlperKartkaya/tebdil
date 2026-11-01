# Sequential research pass

The public page's original design and historical recordings stay unchanged. New
verification is separate from historical task-performance evidence. Existing
uncommitted controller, counterfactual and report artifacts are not inputs to this pass.

| Order | Gate before moving on | Status |
|---|---|---|
| A. Ouro execution contract | Per-depth real-checkpoint state/gate/logit parity; masks/positions; both readouts; FP32/BF16 diagnostic; documentation and Pages evidence | Complete: `e0728b3`; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41219360287) and public browser checks passed |
| B. Trajectory semantics | Explicit token/depth and prefix contracts; baseline-generation calibration | Complete: `22cc752`, 122 tests including real Ouro; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41239947652) and full public browser checks passed |
| C. Operator controls | Correctly named input noise; output-matched controls; algebraic invariants; fixed-prefix residual study | Complete: `65cd09b`, 135 tests including real Ouro; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41241237400), public hashes and full browser checks passed |
| D. Normalization diagnostics | Explicit radial/directional and norm-preserving ablations | Complete: `43e807d`, 142 tests including real Ouro; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41242354541), public hashes and full browser checks passed |
| Completed-answer gate 1 | Frozen prompt/stopping/scoring; real baseline smoke; docs and Pages | Complete: `54b3381`, `d74a61f`; 167 tests, real EOS completion, [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41245538443), public browser/hash checks passed |
| Completed-answer gate 2 | Fixed intervention/donor definitions, appropriate controls and measured completed-answer costs | Technical smoke complete: `c60dbaf`, `0dcbdc3`; five EOS completions, 14,402 new calls; 178 tests, [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41254056115), public browser/hash checks passed; boxed-format limit retained, no efficacy claim |
| Measurement v2 prerequisite | Versioned format repair, immutable v1 scores, offline diagnostic and Pages verification | Complete: `b315d22`, `2f9f96b`; 232 tests, [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41273817261), public browser/hash checks passed; no new model completions |
| Baseline difficulty prerequisite | All twelve development S0 attempts, versioned scoring, measured costs and raw evidence | Complete: `0c1719e` freeze, `56a4eab` evidence; 12 EOS, 10 recognized correct, 0 incorrect, 2 invalid format; 26,668 new calls, 1,524 reused; 242 tests. [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41293024936) and public browser/hash checks passed. [Headroom unresolved](../reports/baseline_difficulty_verification.md), so utility prerequisite remains unmet |
| Explicit thinking-mode recording | True-mode prompt provenance, reasoning/final boundaries, native smoke, documentation and Pages | Complete: `1402630` freeze, `e73273e` evidence; 379 tokens, 1,516 calls, closed reasoning/final boundary; 269 tests. [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41328226496), public browser/byte/provenance checks passed; [verification](../reports/prompt_started_reasoning_verification.md). Historical runners remain False |
| Reasoning-recovery design correction | Tasks that exercise multi-step reasoning; distinguish CoT signals from latent updates and correctness from efficiency | [Draft design](reasoning_stall_study.md); passive monitor and new task/verifier contract not implemented, no new inference |
| Completed-answer gate 3 | Development timing/utility pilot, then frozen independent test with uncertainty | Not started: simple arithmetic did not test reasoning stalls; recovery and efficiency require separately frozen endpoints; all 60 reserved test questions remain unexecuted |
| E. Controller evaluation | Unshaped and heuristic controls only after intervention utility is established | Prerequisite unmet: C/D establish sensitivity, not completed-answer utility; no new controller evaluation |
| Publication package | Frozen held-out campaign, complete costs and immutable reproducible evidence | Not started |

No later stage is considered complete because an earlier stage's tests pass.
Real-model execution is not proof of efficacy, statistical support or complete
historical provenance. Each completed stage gets its own validation record and commit.

C/D do not clear the efficacy gate. Before controller evaluation, a new held-out
intervention comparison needs one frozen completed-answer prompt/stopping/scoring
protocol, matched controls, measured costs and immutable outputs. The historical
raw two-token tables and these three reused pilot prefixes cannot supply that proof.

## Baseline decision

The twelve-question pilot finished without truncation. Its frozen rule returns
`headroom_unresolved_format_or_truncation` because two EOS answers are outside v2's
grammar, not because of a recognized wrong integer. Post-hoc inspection finds the
gold value in each format miss; original scores remain unchanged. This is neither
an arithmetic-accuracy estimate nor evidence against TEBDİL.

Do not proceed to intervention utility or controller training on these results.
The next prerequisite is a separately frozen task/verifier and passive-monitoring
contract on multi-step reasoning, with independently reserved matching test data.
The [reasoning-stall study correction](reasoning_stall_study.md) distinguishes
possible CoT stagnation from recurrent-depth update norms, and correctness recovery
from efficiency at maintained accuracy. It is a draft, not an executed experiment.
Keep all current development observations and the sixty unexecuted test questions
intact. A new rescue pilot needs interpretable baseline failures; an efficiency study
instead needs validated excess-compute/process endpoints and accuracy safeguards.
Neither follows from response length or repetition alone. Any later utility
comparison must retain correct cases and account for degradation and actual
compute, not select only failures.
