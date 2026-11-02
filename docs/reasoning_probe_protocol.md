# Reasoning CoT sensitivity probe v1

Frozen before new intervention observations. This is a development mechanism/process
probe, **not a stall-recovery evaluation, learned policy or full-answer benchmark**.
Both existing baseline attempts are unfinished. Their known outcomes and process
reviews informed this diagnostic, so it is not preregistered held-out evidence.

## Scope and order

Use both recorded CNF/key-planning cases from passive_reasoning_v1, in that order.
Probe every fixed snapshot index k=0,127,255,511,1023,2047, not a hand-picked error,
repeat flag, confidence threshold or favorable outcome. Each input is the original
True-mode prompt plus exactly the actual baseline tokens [0:k]. Do not re-render
questions or insert a thought, hint, gold answer or text into the input.

Reuse the pinned BF16 MPS checkpoint, native masks/positions, four depths, eager
attention, last-step readout, cache disabled. Fork A/B after depth one with existing
normalized Gaussian perturbation .25, seed 20261102+131*k. At input depth two,
compare S0, M0 state mixing, M1 update mixing, E1 genuine excursion-and-return, and
R_output: seeded random output correction matched to the E1 displacement from S0
before casting (seed branch_seed+17). Reuse existing operators. Alpha=beta=.65,
full scope, trust radius=.1, no gates/projections/controller/parameter search.
Advance each candidate normally to depth four and take its actual greedy token.

S0 must match both the recorded token and dtype-aware last-position normalized
state hash at every selected prefix. Abort and preserve evidence on mismatch or
instability; never replace failed cases. Check receiving/donor input isolation.
Record actual state hashes, intervention telemetry, realized BF16 random norm error,
logit changes, token IDs and decoded text. Changed hidden states/logits do not imply
changed greedy text; changed text does not imply improvement. No latent geometry.

## Conditional short continuation window

After all twelve probes, choose the **first E1 next-token divergence in index order
for each case**, if one exists. This is an effect-conditioned diagnostic selection,
not a causal stall detector or unbiased efficacy estimate. Keep all non-divergent
probes too. If none exists, generate no continuation for that case.

At that shared prefix compare S0, M0, M1, E1 and R_output for a window of at most
min(128,2048-k) output tokens including the already measured proposal. Only the
first token receives the intervention. Subsequent tokens use unchanged native
greedy generation with no state/cache carried from the earlier forward.
Reuse the actual recorded S0 window, explicitly marked as previously executed,
not a new inference. If another arm has the same first token as S0, use no new
native continuation: report its continuation as **not executed**, not an observed
copy of the baseline. Equal token then implies no carried intervention state in
this cache-free single-intervention contract, but we do not relabel reused text
as new output. For a changed first token, record each actual native proposal,
prefix and physical boundary calls in an append-only journal. Stop on EOS/window.

A window ending without EOS is `window_exhausted`, not a recognized wrong answer
or a new full 2048-token attempt. The frozen task verifier may score only an actual
EOS-terminated closed final answer; never extract answers from thought text.
Primary descriptive outputs: twelve context-wise token changes, actual CoT window
text and stop/format/verified outcome if eligible. Do not select a success for the
website while hiding other arms. No significance tests or population accuracy:
the units are two reused problems, not twelve independent samples.

## Costs, evidence and release

References are shared: output matching requires E1 computation. Record every
construction and continuation call, actual full-prefix token positions and runtime;
not independent compute-matched arms or optimized latency/FLOPs. Previous baseline
cost is reported separately, not zero. Add no vocabulary/gold scoring passes.
Freeze source/protocol/tests/checkpoint/input hashes before execution; verify after.
Create a new output directory, preserve journals/partial failures, no retry or
parameter tuning on these outcomes. No reserved-test inference or training.

Pages may show these actual paired contexts/windows with a no-rescue warning,
while keeping the historical arithmetic replay accessible. Replace it with a
successful reasoning-recovery narrative only after a separately frozen controlled
full-answer comparison actually demonstrates that finding. Task suitability and
completion-budget adequacy remain unresolved by this probe.
