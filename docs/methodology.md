# Methodology and evidence contract

This describes the implemented system, not a claim of improved reasoning. Current outcomes and uncertainty are in [findings](findings.md); execution checks are in [MVP scientific verification](../reports/mvp_scientific_verification.md).

## Real recurrent computation

`OuroRecurrentAdapter` loads `ByteDance/Ouro-1.4B-Thinking` at revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Each transition executes the shared decoder layers and final normalization, retaining the recurrent-step and position context. The adapter uses `use_cache=False`; it does not share a mutable attention cache between branches. See [adapter](../src/tebdil/models/ouro_adapter.py) and [vendored upstream source](../vendor/ouro_1_4b_thinking/).

Branches are seeded perturbations of a shared prefix, not separate models. Distinct tensors do not imply distinct semantic strategies. The [operator specification](methods.md) distinguishes visited-state computation and update transport from switching or mixing.

## Three kinds of control

- **Fixed:** E1 intervenes at a configured step with configured donor, α and β.
- **Heuristic:** E2 uses uncertainty, stagnation and donor compatibility rules; it is not trained.
- **Learned scorer, usually hybrid:** E3 fits a ridge or MLP predictor to offline counterfactual advantages. Gated variants restrict action eligibility using hand-set receiving-logit margin/entropy thresholds. With the default thresholds, EXIT requires margin ≥ 0.72; EXCURSION requires margin ≤ 0.70 and entropy ≥ 1.20, plus feasibility/budget checks. These rules are **not Ouro's native exit gate**. The ungated ablation disables those eligibility gates, not all action-space or budget constraints.

At inference, [features](../src/tebdil/controller/features.py) observe hidden-state/logit summaries and donor compatibility, not answers. The state-action vector has 29 coordinates (`FEATURE_SCHEMA`), including action interactions; it is not simply a 29-dimensional latent state. CONTINUE's anchored vector and score are zero. Predicted advantages are not calibrated probabilities. See [controller](../src/tebdil/controller/learned_policy.py), [action space](../src/tebdil/controller/action_space.py), and [leakage tests](../tests/test_no_label_leakage.py).

## Offline training and split limits

[Counterfactual collection](../src/tebdil/controller/counterfactual.py) runs candidate continuations and computes an advantage relative to CONTINUE. The implemented utility combines correctness, a bounded change in gold-answer log probability, a compute penalty, degradation penalty and optional hand-written state shaping. Ground-truth supervision is permitted **offline**, never as an inference feature. The unshaped ablation removes the state-shaping term. `compute_action_utility` is the executable definition; a simple accuracy-minus-cost equation would omit implemented terms.

Controller files carry dataset/code/parameter hashes and training metadata. The exported [train](../site/static/evidence/train_counterfactuals.json), [validation](../site/static/evidence/val_counterfactuals.json), [split manifest](../site/static/evidence/split_manifest.json), and controller files are inspectable in the observatory. A split manifest is a historical artifact: it may describe an earlier collection generation and must not override the checkpoint's actual dataset hashes or records. Train and validation can share template families; their problem instances are separate. Held-out test template IDs/prompts are checked for overlap. This is not evidence of independent distributional generalization or preregistration.

## Paired evaluation

A trace is a method/problem/seed result. Pair every method with S0 using the same `(problem_id, seed)`; report accuracy difference, rescue among baseline failures, and degradation among baseline successes. Both controller evaluations use 18 problems × 3 seeds. Seeds are repeated measurements, not 54 independent problems.

Problem-cluster bootstrap intervals resample problems, retaining repeated seeds. The historical field `cluster_mcnemar_p_value` contains a **two-sided exact sign test on per-problem mean differences**, not ordinary row-level McNemar. It is not displayed as a probability of benefit. Multiple comparisons are exploratory and uncorrected. See [statistics implementation](../src/tebdil/evaluation/bootstrap.py) and [paired summaries](../src/tebdil/evaluation/metrics.py).

## Costs, audits and replay

Actual recurrent calls include donor/visit work where recorded ledgers reconcile. Calls at different sequence lengths are not equal FLOPs. Historical P0/P1 selection records combine output and budget provenance; mismatching costs are unverified. Future derived records explicitly label an idealized reference parallel budget, separate selected-output provenance and consulted sources. That reference is not verified total selection execution. No strict compute-matched conclusion follows.

The [audit](../src/tebdil/audit.py) checks provenance, split/leakage, trace/summary and controller integrity and can request real checkpoint replay. It fails closed on missing/inconsistent execution ledgers. Passing replay for selected cases does not certify every historical recording or negate an audit failure. See [reproduction](reproduction.md) for separate clean-clone static checks and real-model commands.
