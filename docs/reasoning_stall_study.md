# Reasoning-stall recovery: next research question

Status: **Draft study design; not a frozen protocol or an executed experiment**.
No new task set, stall detector, thresholds, controller or intervention results are
introduced by this note. Historical protocols, scores and reserved sets stay intact.

## What the arithmetic pilot did not test

The twelve simple arithmetic questions tested completed generation and measurement,
not recovery from stalled multi-step reasoning. Mathematical tasks can require
reasoning; the problem is this pool's limited demands, not mathematics as a category.
Long responses, repeated checking and the two format misses do not establish a stall.
The absence of recognized wrong answers here is not evidence against TEBDİL.

The revised question is: **when a causal, label-free monitor flags possible loss of
reasoning progress, does excursion-and-return improve verified outcomes or the
compute needed to reach them, compared with appropriate continuations and controls?**
Accuracy recovery and efficiency at maintained accuracy are separate hypotheses.
A ceiling on final correctness does not, by itself, rule out an efficiency study.

## Current implementation boundary

`core/policies.py::compute_stagnation_ratio` measures relative hidden-state update
norm across recurrent depths inside one prefix. It does not track CoT progress
across emitted tokens. The current adaptive trigger uses entropy and top-two margin;
the recorded stagnation ratio is not used to trigger a visit. Confidence is not a
correctness verifier. There is no validated CoT-stall detector in this pipeline.

There are two axes: generated token `k` and recurrent depth `t`. The current
cache-free engine re-embeds the full prefix at each token. A and B re-fork within
that computation; they are not persistent semantic agents. An excursion at `(k,t)`
returns an update to A and can affect future text through the emitted token.
Unchanged next-token output leaves no extra carried hidden memory at `k+1`.
Do not directly subtract growing full-prefix tensors across tokens and call their
norm a progress score. Same-depth observations need explicit token/context alignment.
See the [implemented trajectory contract](trajectory_semantics.md).

## Tasks before outcomes

Use a small, preselected difficulty ladder with mechanically checkable answers:
constraint satisfaction with interacting constraints, path/planning tasks requiring
backtracking, and multi-hop symbolic deduction. Include non-stalled successes and
failures, not only dramatic repetitive examples. Retain mathematical reasoning if
it presents comparable search or constraint demands. A family name is not evidence
of difficulty; the existing small Boolean/graph generators may also be too easy.

Freeze task generation, exact prompts, output grammar, EOS/budget handling and
independent development/test identities before new observations. Final answer format
must fit the task's verifier rather than forcing every task through the integer
parser. Establish verifier and extraction tests before inference. Tune difficulty
and monitoring on development only; do not inspect a reserved test set to select
cases. Keep the current twelve observations and sixty reserved questions unchanged.

## Observe before intervening

CoT repetition is a candidate signal, not proof of a reasoning stall. Start with
prefix-only textual repetition/revisiting diagnostics and real, aligned per-depth
state-update/readout observations. A long trace may still be making progress;
a confidently wrong trace may not repeat at all. Do not equate entropy, small latent
updates, token count or “wait” phrases with failed reasoning.

Preserve actual emitted CoT tokens, token/depth indices, prefix hashes, monitor
features, trigger reasons and measured costs. A future runtime monitor must not see
ground truth, verifier feedback, future tokens or eventual outcome. Final-answer
validators and task-specific progress annotation remain offline evaluation only.
Validate candidate flags against an explicit progress rubric, including false alarms,
missed failures and productive self-correction. Blind qualitative annotators to the
intervention condition where feasible. No LLM judge alone establishes progress.
Do not describe a new heuristic monitor as learned or a latent region as a proven
alternative reasoning strategy.

## Causal comparison at a suspected stall

Fork all arms from the **same observed prefix**, before their continuations diverge.
Choose the first eligible flag using only information available there, not a later
known failure or an oracle's favorite intervention time. Preserve that context's
mask, positions, four-depth contract and receiving/donor isolation. Do not paste
states from a different textual prefix into A and treat them as compatible donors.

Compare ordinary continuation, direct mixing, output-matched random intervention
and genuine donor excursion-and-return. A random/fixed timing control tests whether
the monitor adds value beyond generic perturbation. Include a continuation or
independent-restart control with a comparable actual compute allowance; define
budget matching and charge monitoring, donor creation and selection before running.
Equal output-token limits alone are not compute matching. Do not silently add
recurrent depths beyond the checkpoint's four-depth setting as a neutral control.

Primary endpoints are verified final correctness over all preselected tasks and
actual compute at a declared budget. Report incorrect-to-correct recovery,
correct-to-incorrect degradation, invalid formats and unfinished attempts separately.
Secondary process endpoints can include independently assessed progress after the
flag and repetition reduction; changed prose alone is not successful recovery.
Conditional flagged-case effects must be labeled conditional, not population gains.
A prefix-fork diagnostic is not yet a full online-policy evaluation.

## Next sequential gate

First implement and test the task/verifier contract and passive monitoring; freeze
its protocol and data before new baseline observations. Only after validating that
recording stage should the controlled continuation experiment run. Freeze endpoints,
controls, trigger rules, budgets and uncertainty analysis before that comparison.
Keep unsuccessful and non-triggered cases. Learn a controller only after evidence
that the underlying intervention is useful. This note clears none of those gates.

## Evidence informing this correction

[Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388) show that emitted
CoT explanations can misrepresent causes of model predictions. Text monitoring
therefore needs independent checks; their results are not measurements of Ouro.
[Chen et al., 2025](https://arxiv.org/abs/2412.21187) examine excess reasoning compute
and process/outcome efficiency in other reasoning models. This motivates a separate
efficiency question, not a claim that repetition reduction improves TEBDİL accuracy.
