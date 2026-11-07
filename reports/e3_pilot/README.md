# E3 post-return acceptance pilot

Frozen before GPU execution, 2026-11-08. E3 reuses E2's causal donor selection
and pre-norm excursion operator. Before committing an excursion, compare the
returned receiver against its already-computed normal next state. Accept only
finite logits with at least +0.10 top-two logit margin and no entropy increase.
Otherwise keep the exact normal hidden state and per-layer K/V. Rejected probes
still count toward physical recurrent work; they are not accepted excursions.
No labels, task answers or validators enter this decision. This is a confidence
proxy, not a correctness oracle or a trained controller. Historical E3_* learned
variants are distinct and unchanged.

First experiment: pinned Ouro-1.4B-Thinking, FP32/SDPA, one L4 worker, seed 42,
1024-token cap, S0/E2/E3 on `trap_hotel_empty_rooms` (historical rescue) and
`trap_monty_hall_4doors` (historical regression). These are outcome-selected
**development diagnostics**, not held-out accuracy or an unbiased benchmark.
No prompt/threshold/budget changes after observing outcomes. Retain every outcome.
This fast first pass uses a shorter cap than the historical 3000-token runs; do
not compare their accuracy or costs as matched experiments.
The small scope prioritizes turnaround; expand only after checking these records.

The deceptive suite now uses the canonical checkpointed runner, preserving the
historical alpha/beta 0.65, relative perturbation 0.14, one-token window and E2
policy settings. All three methods get identical settings. E3's rejected returns
do not start the accepted-intervention cooldown, so repeated proposals can cost
more; compare total recurrent calls as well as tokens and time.

```sh
python scripts/run_colab_l4.py --suite deceptive \
  --task-ids trap_hotel_empty_rooms trap_monty_hall_4doors \
  --methods S0 E2 E3 --max-new-tokens 1024 --seed 42 \
  --dtype float32 --attn sdpa --output <new-summary.json> \
  --checkpoint-dir <persistent-directory>
```

Implementation scope: policy predicate, excursion rollback, scheduler method
routing/rejection telemetry, canonical suite integration, mini-cache regression
tests. No new dependency or learned-controller training. S0/E2 algorithm settings
are unchanged. Unit tests verify both accepted/rejected cache selection and
rejected E3 token/cache equality with S0.
