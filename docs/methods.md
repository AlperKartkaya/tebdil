# Excursion-and-return: implemented operator

## Visit, compute, return

Let F_t be one genuine recurrent transition, including normalization, and V_t(H) = F_t(H) − H. Receiving A and donor B share model weights, but have separate hidden states.

With a token-scope mask M, define D = (H_B − H_A) ⊙ M. The implementation bounds the visit offset:

$$\alpha_{\mathrm{eff}}=\alpha\min\left(1,\frac{\rho_{\max}\|H_A\|_F}{\|\alpha D\|_F+\varepsilon}\right),\qquad H_{\mathrm{visit}}=H_A+\alpha_{\mathrm{eff}}D.$$

The bound applies when `max_relative_norm > 0`; zero disables clipping. Compute V_A, V_B and V_visit through actual transitions. For the full-scope, unprojected return:

$$H_A'=H_A+(1-\beta)V_A+\beta V_{\mathrm{visit}}.$$

A remains the anchor; B advances normally. This is **not** continuing from H_visit or replacing A with B. The visited vector is `F(visit) − visit`, not `F(visit) − A`.

## Scoping, projection and gates

The general applied correction is `β × gate_scale × P((V_effective − V_A) ⊙ M)`, added to the normal A update. Here P is the configured coordinate restriction. Coordinate and `subspace_fraction` variants retain the largest-magnitude correction coordinates per token; this is **not a learned low-rank subspace** and does not project the initial donor direction. Ties can retain more than the requested fraction.

For `E1_gated`, `gate_scale = 1 − 0.5 × sigmoid(native_exit_gate)` when available. This operator gate is separate from the controller's margin/entropy action-eligibility rules. Unstable/nonfinite returns can fall back to the ordinary A transition.

The identity cases α=0, β=0, or identical A/B (without a random-direction override) short-circuit to normal transitions. Recorded zero curvature in that path is an implementation convention, not an evaluated geometric derivative. See [operator](../src/tebdil/core/excursions.py) and [identity tests](../tests/test_excursion_invariants.py).

## Nonlinear residual

$$C=V_{\mathrm{visit}}-[(1-\alpha_{\mathrm{eff}})V_A+\alpha_{\mathrm{eff}}V_B].$$

For full-scope, ungated, unprojected transfer in exact arithmetic, excursion differs from the corresponding linear update mixture by βC, using the effective clipped alpha. Affine transitions give C = 0 only if the actual visit lies on the full endpoint segment; dtype casting can move it off that segment. The gate C diagnostic records pre-/post-cast identity errors separately. Under masks/projections/gates, use the applied correction above; the full norm of C is not the applied correction norm. The recorded `curvature_frob` measures the full response before those restrictions. It does not identify an attractor basin.

## Controls distinguish the mechanism

| Method | What actually changes |
| --- | --- |
| S0 / S1 | Ordinary recurrence / extra recurrent depth |
| P0 / P1 | Confidence selection / answer-informed oracle selection across independent outputs |
| M0 | Mix hidden states, then continue from that mixture |
| M1 | Mix ordinary update vectors; never evaluate a visited state |
| C0 | Compute at the visit and continue from it; do not return |
| R0 | Evaluate F(A + ε), with pre-cast input-noise norm β × clipped visit-offset norm; **not output-matched** |
| E0 | Visit along a seeded random direction, then return |
| E1 / E2 | Fixed / heuristic adaptive excursion-and-return |
| E1_no_curv | Compute the visit but remove its nonlinear residual before return |
| E1_rand_curv | Replace C by random noise matched to the full residual norm **before** return masking, projection and casting |

These are local ablations, not proof that any term is universally necessary. Historical control telemetry sometimes contains placeholder zeros; the observatory does not present those as measured curvature. [Controls source](../src/tebdil/core/controls.py).

## Execution and visualization boundaries

Forked states clone branch context; current Ouro execution disables KV caching rather than sharing cache state. Identity and donor-isolation checks include real-checkpoint integration. PCA depicts only recorded last-token branch samples in independently fitted axes. The observatory withholds reconstructed visit geometry and later-token projections without matching data. See [methodology](methodology.md), [limitations](limitations.md), and [reproduction](reproduction.md).

## Fixed-prefix output controls

Gate C adds an analysis-only `build_output_matched_random_state`: it draws seeded
noise in the configured scope with norm equal to the **actual reference-minus-base
output displacement**, then refreshes readout history and native gate. FP32 target
norm matching precedes casting; realized BF16 errors are explicitly reported. A
zero displacement returns an isolated clone. No transition occurs in this helper,
but computing its reference is not free. It is not a new scheduler method.

M1 now refreshes its native gate at the mixed output rather than retaining the
ordinary A gate. Last-step hidden states and all historical recordings are unchanged;
future native-gate readouts use the corrected history.

See the [frozen diagnostic protocol](operator_controls_protocol.md) and
[verification record](../reports/operator_controls_verification.md).

## Normalization diagnostics are separate from production recurrence

The native final RMSNorm remains part of every F. Gate D only constructs diagnostic
radial/tangential visits and per-token norm-preserved visits/returns. Matching can
use Euclidean coordinates or coordinates divided by the learned boundary weight;
these metrics are not interchangeable or semantic geometry. Preservation precedes
explicit dtype casting, and realized errors/displacements are measured. No new
production normalizer, method flag or trained policy is added. See the
[frozen protocol](normalization_protocol.md) and [observations](../reports/normalization_verification.md).
