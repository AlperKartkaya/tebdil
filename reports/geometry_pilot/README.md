# Geometry development pilot — 2026-11-08

**No demonstrated gain. The detector never activated.** Geometry produced exactly
S0's token sequences on all four tasks. This does not evaluate waypoint quality.

## Results

Pinned Ouro-1.4B-Thinking, FP32/SDPA, NVIDIA L4, seed 42, 1,024 output tokens per
run. Four preselected development tasks × three methods; all 12 runs finished.
Budget-exhausted answers count as unsuccessful, not correct partial reasoning.

| Method | Correct | Budget exhausted | Generated tokens | Recurrent calls | Applied excursions |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 3/4 | 1/4 | 2,600 | 10,416 | 0 |
| E2 | 3/4 | 1/4 | 2,633 | 11,368 | 50 |
| geometry | 3/4 | 1/4 | 2,600 | 10,416 | 0 |

All three methods answered hotel/empty rooms, harmonic-mean speed and bat/ball/glove
correctly. All exhausted the budget on four-door Monty Hall. Both E2 and geometry
had zero rescues and zero regressions relative to S0 in this subset. This does not
invalidate gains or regressions on other historical E2 tasks.

Geometry also made **2,600 extra vocabulary projections**. Equal recurrent calls
are not equal total compute. Measured sequential wall times were 317.3910 seconds
for S0, 352.9735 for E2 and 325.8308 for geometry. Run order was not balanced and
includes warm-up effects; these are recorded costs, not a throughput benchmark.
No accuracy or speed advantage is established.

## What failed to activate

There were 32 warm-up observations and 2,568 classified as progress. No cycling,
flattening, eligible selection or attempted visit occurred. The smallest net/path
ratio was 0.42875 (warning threshold <0.25); the smallest relative movement was
0.50075 (flattening threshold <0.10). On Monty Hall alone the minima were 0.44170
and 0.54332. Ambiguity and persistence could not activate a missing motion warning.

This is evidence that the chosen warning signature did not occur in this pilot,
not that the model never struggled. The forced real-checkpoint plumbing test passed
commit, rollback, prefix-cache preservation and two-transition endpoint accounting;
that cannot substitute for naturally triggered interventions or answer-quality gains.

**Next scientific question:** can a causal, label-free diagnostic identify useful
intervention opportunities without confusing ordinary convergence with difficulty?
Do not manufacture positive results by moving thresholds onto these four cases.
A larger model sweep is premature while the detector remains inactive here.

## Implementation and evidence

Code: `f9ef5d5d5a1b4f86090fe0596509028203c773a6`. Model revision:
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Detector/selection profile was fixed before
execution. The endpoint uses alpha=1, beta=0.65, pre-norm transport and the existing
E2 return mechanics; no E3 acceptance test, candidate search or learned controller.

`summary.json` contains all outcomes and geometry traces. `checkpoints/` retains
every generation with its input/configuration contract and payload hash. All 52
execution-source hashes matched the local checkout after transfer. `transfer.json`
records the verified SHA-256 of every downloaded artifact. `preflight.log` records
the real-weight commit/rollback test; `run.log` records the complete pilot.

Reproduce with the existing Colab runner:

```sh
python scripts/run_colab_l4.py --suite deceptive \
  --task-ids trap_hotel_empty_rooms trap_monty_hall_4doors \
             trap_bat_ball_glove trap_speed_harmonic_mean \
  --methods S0 E2 geometry --max-new-tokens 1024 \
  --dtype float32 --attn sdpa --seed 42 \
  --checkpoint-dir <persistent-directory> --output <new-summary.json>
```

This is a small development pilot, not held-out validation, a significance result,
or evidence of model-independent gains. Geometry remains opt-in.
