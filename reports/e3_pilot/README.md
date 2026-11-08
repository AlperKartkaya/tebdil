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

## Completed GPU results

Executed commit: `0b937cb62b5259a2b348ed48f2dc62a3b071580c`.
All six fresh generations completed their execution contract (EOS or the fixed
1024-token budget). None reused a checkpoint. [Raw summary](summary.json), six
per-generation [checkpoints](checkpoints/), and remote/local SHA-256
[transfer verification](transfer.json) are retained. Each checkpoint binds prompt
IDs, effective parameters, pinned model revision, environment and source hashes.

| Task | Method | Outcome | Tokens | Recurrent calls | Accepted / rejected returns | Seconds |
|---|---|---|---:|---:|---:|---:|
| Hotel | S0 | Correct: 0 | 736 | 2,948 | 0 / 0 | 90.6519 |
| Hotel | E2 | Correct: 0 | 769 | 3,331 | 19 / 0 | 103.1071 |
| Hotel | E3 | Correct: 0 | 686 | 3,353 | 6 / 17 | 105.0320 |
| Monty Hall | S0 | Budget exhausted | 1,024 | 4,100 | 0 / 0 | 123.8699 |
| Monty Hall | E2 | Budget exhausted | 1,024 | 4,461 | 21 / 0 | 138.2475 |
| Monty Hall | E3 | Budget exhausted | 1,024 | 4,960 | 12 / 22 | 155.0912 |

### Interpretation

- All methods completed one correct answer out of two attempts. E3 demonstrated
  neither an accuracy gain nor a compute saving in this pilot.
- Hotel E3 used fewer tokens than S0 and E2, but **more recurrent calls and time**.
  A shorter answer trajectory did not make the speculative computation cheaper.
- Across both tasks: S0 used 7,048 calls, E2 7,792, E3 8,313. Rejection preserves
  the normal state/cache, but does not refund donor/visit work already performed.
- E3 accepted 18 excursions and rejected 39 candidate returns. Fewer accepted
  interventions alone is not evidence of a better policy.
- The historical hotel rescue did not reproduce as a baseline failure: FP32 S0
  itself answered correctly here. Precision, provenance and runtime differences
  prevent attributing that change to any one cause.
- Monty Hall remained unfinished for every method. This shorter-budget pilot
  does not establish how E3 performs at the historical 3000-token budget.

Keep E3 experimental; it is not promoted to the default. The next targeted
optimization is a bounded probe/cooldown policy that limits the cost of repeated
rejected proposals, evaluated against this unchanged pilot. Any larger-budget
study must be a separately frozen campaign, not a retroactive extension or
replacement of these outcomes. No held-out questions or controller training were
used, and single-run timings are not latency-distribution estimates.
