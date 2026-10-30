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

For full-scope unprojected transfer, excursion differs from the corresponding linear update mixture by βC. Under masks/projections/gates, use the applied correction above; the full norm of C is not the applied correction norm. The recorded `curvature_frob` measures the full response before those restrictions. It does not identify an attractor basin.

## Controls distinguish the mechanism

| Method | What actually changes |
| --- | --- |
| S0 / S1 | Ordinary recurrence / extra recurrent depth |
| P0 / P1 | Confidence selection / answer-informed oracle selection across independent outputs |
| M0 | Mix hidden states, then continue from that mixture |
| M1 | Mix ordinary update vectors; never evaluate a visited state |
| C0 | Compute at the visit and continue from it; do not return |
| R0 | Apply norm-matched random noise, not a visited update |
| E0 | Visit along a seeded random direction, then return |
| E1 / E2 | Fixed / heuristic adaptive excursion-and-return |
| E1_no_curv | Compute the visit but remove its nonlinear residual before return |
| E1_rand_curv | Replace the nonlinear residual with norm-matched random residual |

These are local ablations, not proof that any term is universally necessary. Historical control telemetry sometimes contains placeholder zeros; the observatory does not present those as measured curvature. [Controls source](../src/tebdil/core/controls.py).

## Execution and visualization boundaries

Forked states clone branch context; current Ouro execution disables KV caching rather than sharing cache state. Identity and donor-isolation checks include real-checkpoint integration. PCA depicts only recorded last-token branch samples in independently fitted axes. The observatory withholds reconstructed visit geometry and later-token projections without matching data. See [methodology](methodology.md), [limitations](limitations.md), and [reproduction](reproduction.md).
