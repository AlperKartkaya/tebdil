# TEBDİL: deterministic geometry-guided excursions

Status: **isolated geometry primitives implemented after the user requested the next pass; no inference integration or efficacy validation**.
Scope: training-free detection, deterministic waypoint selection and bounded return
on supported recurrent/looped model adapters. This document does not authorize a GPU efficacy campaign. Do not reinterpret this proposal as a demonstrated improvement.

## 1. Research objective and exclusions

Improve completed-answer correctness at matched computation, or reduce computation
at matched correctness, by selectively changing unproductive recurrent dynamics.

The controller uses only the current request's causal computation. It does not use
training examples, an offline latent database, task labels, answer validators,
learned rewards, a trained router, random perturbations or candidate rollout search.
Output uncertainty can corroborate a dynamical warning, but is not the destination
objective and is not a correctness estimator.

Task-agnostic means no dataset-specific rules or learned task parameters. Portable
means one algorithm and one frozen, dimensionless parameter profile across supported
adapters. It does not mean parameter-free, representation-invariant under arbitrary
reparameterizations, or guaranteed to improve every model/task. Per-request online
normalization is observation, not weight training. Adapter correctness and numerical
tolerances still require model-specific verification.

The existing Ouro E2 competitive-logit path already chooses its perturbation
arithmetically from output-weight competitors; it is not simply random noise.
This proposal changes the source and selection criterion of the direction to
current-request trajectory geometry. It is not a claim to introduce determinism
where none existed. Missing geometric evidence must never invoke the existing
Gaussian fallback.

## 2. State and comparability contract

Let n index generated tokens and k the native recurrent boundary. h[n,k] is the
active token's hidden state at a declared comparable boundary. The adapter exposes:

- ordinary recurrent transitions, without adding native loop iterations;
- comparable state coordinates, a metric map phi, and a valid boundary retraction;
- residual-coordinate update transport and isolated/committable K/V branches;
- optional current readout probabilities, without extra recurrent evaluation.

Use q[n,k] = phi(h[n,k]) / ||phi(h[n,k])|| for geometry. Zero/nonfinite norms mean
abstention. A metric map can account for learned normalization gains; do not blindly
apply RMSNorm again to an already normalized state or mix raw/pre-norm vectors with
post-norm vectors. Identity phi is permissible only when its geometry contract has
been checked. Cosine is not invariant to arbitrary learned anisotropic scaling.

Within-token recurrent states may be compared only if the adapter declares their
boundaries comparable. Do not concatenate successive tokens' hidden states and call
that a trajectory under a fixed transition: each token changes the input/cache.
Unrelated layers in an ordinary feed-forward transformer are not automatically
comparable recurrent states. Black-box API models are outside this initial scope.

## 3. Detect a dynamical warning, not a wrong answer

At an eligible boundary k, use only already-observed states from this token:
q[k-m], ..., q[k], with m >= 2. Let d[i] = q[i] - q[i-1]. Define:

    L = sum_i ||d[i]||                       (path length)
    E = ||q[k] - q[k-m]|| / (L + epsilon)    (net movement / total movement)
    Z = ||d[k]|| / (sigma[k] + epsilon)      (relative movement)

sigma[k] is a robust reference movement scale from prior ordinary within-token
transitions at that boundary in this request. It is frozen before scoring the
current observation; do not normalize a stall only against itself. Insufficient
history or a degenerate reference scale means abstention.

Candidate warning:

    cycling = L > movement_floor AND E < theta_E
    flattening = Z < theta_Z

A low E with nontrivial movement is compatible with backtracking/cycling. A low Z
can be ordinary convergence and is not by itself a reason to intervene.

For the initial conservative specification, require BOTH:

1. cycling OR flattening;
2. persistent readout ambiguity, corroborated over K eligible observations at the
   same boundary on successive generated tokens.

Readout ambiguity is based on dimensionless normalized entropy H(p)/log(V), judged
against a causal per-request reference distribution, not raw logit margins. This
vocabulary normalization is not perfect cross-tokenizer calibration. Confidence is
used only as a conservative veto, never as the objective to maximize after a visit.

Only the warning statistics/persistence counts carry across tokens, not a claim
that their different hidden states form one fixed-context orbit. Reset suspicion
on recovered motion, EOS/final-answer boundary or an intervention attempt.

This targets a *dynamical-risk signature*. Stable correct convergence, useful
reconsideration and confident errors are counterexamples to stronger claims. A
confidently wrong but smoothly progressing model can evade this detector. Diagnostic
validation must include such counterexamples, not merely known successful rescues.

Ouro's four native passes provide few observations: initially allow only late
eligible boundaries with enough already-computed states, preserve remaining native
computation, and abstain rather than add passes to satisfy the detector.

## 4. Deterministic destination from the current request

Maintain a bounded causal bank B[k] of active-token representations recorded at the
same boundary k from earlier prefix positions. Populate it during ordinary prefill
and decoding; do not rerun the model to build it. Do not store external examples.
Retain only finite representations, original position/boundary identifiers and
recent ordinary transitions; never treat speculative visit states as native anchors.
These observations follow the actual current cache, not a hypothetical untouched
S0 trajectory after earlier interventions.

An anchor is **a steering reference, not a state to resume**. Never transplant its
old positional context or K/V cache. Across-position similarity can reflect syntax
or position rather than useful reasoning; this bank is a hypothesis to validate,
not proof of a semantically correct waypoint.

For current unit state q and candidate unit anchor a[j]:

    c[j] = dot(q, a[j])
    t[j] = a[j] - c[j] q
    p[j] = t[j] / ||t[j]||

Candidates must pass finite/norm checks and a fixed locality band
c_min <= c[j] <= c_max < 1. Reject degenerate tangent directions. Nearest-only
selection would repeat the current region; farthest-only selection could destroy
context. The locality band rules out both extremes.

Project recent within-token movements into the current tangent plane and normalize
the nondegenerate ones to obtain u[i]. Define:

    R[j] = max_i (dot(p[j], u[i]))^2
    S[j] = ((1 + c[j]) / 2) * (1 - R[j])
    j*   = argmax_j S[j]

The score favors a locally related anchor whose direction is not parallel or
antiparallel to recently traversed directions. It measures geometric suitability,
NOT expected correctness. Require a minimum score; break exact ties by original
position, then insertion index. No valid movements, no eligible anchor, or no
sufficiently distinct direction => ordinary continuation, not a random fallback.

Choose one waypoint before any counterfactual transition. Let

    r = min(r_max, kappa * sigma[k])
    v = Retract_h(r * p[j*])

Retraction maps the geometry-space displacement back into the adapter's valid native
boundary representation, preserving the active-token scope and norm constraints.
It is not an unverified Euclidean interpolation assumed to stay on a learned
manifold. The caps and scale controls are dimensionless and frozen across models.

A bounded bank can lack an escape direction. In a perfectly collapsed/collinear
history, geometry alone cannot invent a privileged orthogonal direction in a
rotation-equivariant way. Abstention is intentional, not an implementation failure.

## 5. Visit, transport the update, return

Conceptual operator, with x and v expressed in ONE adapter-declared residual space:

    u_A = G_k(x) - x
    u_V = G_k(v) - v
    delta = clip_relative(u_V - u_A, radius = eta * ||G_k(x)||)
    x_next = BoundaryMap(G_k(x) + beta * delta)

G_k is the transition under the CURRENT token/prefix/cache context for both states.
This formulation transports a difference of updates, rather than replacing the
receiver by the anchor or averaging unrelated final states. All coordinate maps
and the clipping denominator must be specified and tested by the adapter.

This is a proposed conceptual contract, not a claim that today's excursion operator
implements this exact simplified equation. Current code includes rho scaling,
curvature amplification and contraction compensation. An implementation proposal
must explicitly reconcile those terms; it must not silently mix formulas or remove
pre-norm transport. The first geometric experiment should hold return mechanics
fixed wherever possible so destination selection is not confounded with a new
transport operator.

There is no candidate lookahead: no scoring multiple possible future trajectories.
Evaluating the selected visit itself is indispensable TEBDİL work and must be counted.
If retaining the current operator also needs a donor transition, count that too;
"no search" does not mean "no extra computation."

Commit the receiver state and its consistent per-layer K/V, not the anchor's old
cache. Norm/finiteness/compatibility failure restores the normal transition. Do not
reintroduce E3's immediate entropy-improvement acceptance test. Apply cooldown after
EVERY attempted visit, including rollback, to bound repeated speculative work.

Initial scope: one visit/return at one eligible native boundary, then ordinary
reasoning. Multi-token visits, multiple candidate probes and learned policies are
excluded from the first implementation.

## 6. Explicit control state machine

    OBSERVE -> SUSPECT -> SELECT -> EXCURSE -> COOLDOWN -> OBSERVE

- OBSERVE: collect ordinary states/statistics; insufficient evidence => stay native.
- SUSPECT: require persistent risk signature; recovery => OBSERVE.
- SELECT: deterministic score on existing anchors; no suitable anchor => abstain.
- EXCURSE: one selected visit, bounded return, coherent cache commit or rollback.
- COOLDOWN: entered on either acceptance or rollback; no new probes during cooldown.
- EOS/final-answer boundary: disable interventions for the remainder of generation.

Selection itself has no recurrent calls. A cheap rejected selection need not claim
an accepted intervention. Log trigger inputs, causal bank membership, candidate
scores, selected anchor, radius, actual attempt/rollback and physical work.

## 7. What must be frozen after the user's verdict

The equations are a design family until a profile is fixed: history window m, bank
capacity/retention, warm-up, reference-scale update, ambiguity quantile, persistence K,
locality band, movement/score thresholds, radius and return caps, eligible boundaries,
and cooldown. Do not invent universal constants or fit these to the reported rescues.
Choose one small dimensionless profile and carry it unchanged across evaluated models.
Numerical epsilon/stability tolerances are separate from scientific policy thresholds.

Before efficacy claims: verify no future-state/label access; deterministic selection;
empty/degenerate-bank abstention; movement counterexamples; orthogonal-rotation and
uniform-scale behavior in declared metric coordinates; exact rollback/cache coherence;
and matching physical-cost accounting. These are implementation checks, not gains.

After approval, use paired native S0 and unchanged E2, common questions/precision/
budgets, and fixed held-out protocol. Necessary controls: geometry-trigger only,
random direction at the SAME trigger/attempt budget, and compute-matched alternatives.
Compare rescues AND regressions, completed-answer correctness AND cost; fewer probes,
higher novelty, lower entropy or more movement are not objectives by themselves.
Use the same policy profile on both Ouro sizes before asserting portability; other
architectures require independently validated adapters. A positive result on two
models still does not establish universal model independence.

## 8. Intellectual basis and evidence boundary

Recurrence analysis motivates measuring repeated motion rather than raw confidence:
https://arxiv.org/abs/2409.04110

Fixed/slow points can organize useful RNN computation, not just failure, which is
why this design must not equate low movement with bad reasoning:
https://barak.net.technion.ac.il/files/2012/11/sussillo_barak-neco.pdf

These references motivate diagnostics. They do not validate this waypoint score,
its transport, or an expected TEBDİL performance gain. All such claims remain
hypotheses pending the user's verdict and controlled evaluation.


## 9. Isolated implementation pass

`src/tebdil/core/geometry.py` implements motion diagnostics, causal-anchor validation,
deterministic waypoint scoring, and a per-request/per-boundary persistence/cooldown
gate. `tests/test_geometry.py` exercises synthetic progress, cycles, stable confident
convergence, degeneracy, causality violations, tie-breaking, rotation/scale behavior,
and absence of input mutation/RNG use. These fixtures are not reasoning benchmarks.

The provisional mechanics profile is fixed here before any efficacy observation:
net/path threshold 0.25, relative-motion threshold 0.10, cosine locality [0.50, 0.95],
minimum candidate score 0.05, radius multiplier 0.50, maximum metric radius 0.10,
persistence 3 eligible consecutive positions, cooldown 18 token positions.
Numerical epsilon is 1e-8, separate from scientific thresholds. These constants are
not claimed optimal or portable; they are a reproducible starting profile, not
values fitted to task answers. Scale/rotation tests exclude numerical degeneracy
and do not claim invariance under arbitrary anisotropic reparameterizations.

The module deliberately does NOT create model states, normalize an Ouro state,
change a cache, read a gold answer, run a recurrent step, or invoke an excursion.
Its result is a direction in declared metric coordinates, not an injectable latent.
No new method name is registered and S0/E2/E3 execution is unchanged.

Remaining integration gates: a validated metric/retraction adapter, causal capture
of a bounded per-request native anchor bank, reference-scale and readout-ambiguity
estimators, and shadow-mode logging before any intervention. The caller currently
supplies the already-observed scale and Boolean ambiguity; their causal provenance
cannot be established by a pure geometry function. A low-motion diagnostic is not
permitted to bypass the ambiguity/persistence gate.
