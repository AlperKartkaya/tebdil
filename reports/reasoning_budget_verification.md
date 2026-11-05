# Reasoning baseline budget diagnostic

Status: **forcibly interrupted by explicit user request; no completed pair report**.
[Frozen protocol](../docs/reasoning_budget_protocol.md).

**User computation hold:** the previously allowed process was also killed. No execution exception remains.
No new model/mini-model execution, retry, extension, controller training or experiment
is authorized until explicit user resumption. After this point, verification is
limited to existing-artifact, static frontend and recorded-browser checks; no new
checkpoint or tokenizer/model integration test is started. See the
[end-to-end walkthrough](../docs/computation_walkthrough.md) and [AGENTS.md](../AGENTS.md).

## Question and scope

Both original True-mode reasoning attempts exhausted 2048 output tokens without
EOS or a closed reasoning section. The next narrow question is whether an unchanged
ordinary baseline completes within a fixed 512-token extension. This does not test
TEBDİL utility, prompt optimality, a semantic-stall detector or general difficulty.
Both original cases are retained, with terminal native token/state parity before
resuming and separately charged verification, new inference and reused work.

The original observations and twelve-context sensitivity probe remain immutable.
No harder instance, reserved-set task, coached prompt or parameter sweep is added.
Continued truncation would leave the adequacy question unresolved, not authorize
an outcome-driven sequence of increasing budgets.

## Reference check

The [official XERJ guide](https://xerj.org/llms.txt) was checked. Installed XERJ
`v1.0.0-rc.100` supports `ask`; Hub discovery reported **no coverage** for Ouro
or the Transformers generation contract, so unrelated projects were not substituted.
The existing pinned Apache-2.0 Ouro source at
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` was indexed locally under the new
`budgetouro` prefix: **10/10 files, both detected Python sources, 3802 records,
no junk**, with the data map inspected. This selected source corpus is not an
index of the entire current TEBDİL repository.

Lexical retrieval of `max_new_tokens` returns upstream `README.md:126`'s 512-token
usage example. That is not evidence that 512 or 2048 tokens suffice on our tasks.
Direct source inspection of the vendored forward/readout and existing native
generation confirms the full-prefix, cache-free continuation contract used here.
No upstream code was copied, no dependency changed and no external feedback/PR
was submitted. The session-owned loopback XERJ node was stopped before inference.

## Acceptance

Before inference: mini-native uninterrupted/resumed token equality, physical
ledger/prefix checks, terminal state parity, EOS-without-invented-boundary,
invalid-baseline refusal, nonfinite/callback cleanup and existing-output protection.
Full local tests and static checks must pass, then source/protocol/tests are committed
before observations. Raw results and public verification are added after execution;
a frontend pass alone is not evidence of model computation or completion.

Pre-observation validation: **334 full tests passed** in 60.66 seconds, including
real-checkpoint integration and ten new mini-native continuation checks. Scoped
Ruff, JS syntax, TypeScript, task-freeze identity, export `--check` and diff checks
passed. No current model continuation was inferred from those test results.

Protocol/runner/tests were frozen and pushed at `2309ea0` before observations.
[Freeze CI/Pages 41456896938](https://github.com/AlperKartkaya/tebdil/actions/runs/41456896938)
passed. The actual run used a new output directory and append-only journal.
It was then stopped by SIGKILL at the user's explicit instruction (exit 137);
PID 58098 was verified absent afterward. No resume, retry or further model call was started.


## Retained partial observations, not a completed experiment

| Case | New recorded proposals | Returned native continuation? | Observation |
|---|---:|---|---|
| CNF | 512 | Yes | 2560 total output tokens, reasoning open, `budget_exhausted`; no eligible final answer |
| Key/door planning | 450 | **No** | Interrupted during generation; no finalized continuation record or task score |

Both original terminal-token/state parity checks passed before continuation.
The raw journal has 968 complete JSON events, including 962 new greedy proposals;
its last event is a token, not `finish`. No `report.json` was produced. Planning's
journal proposals are actual observed native-forward argmax IDs, **not** a returned
`generate` completion. Do not invent a stop reason, score, decoded suffix or final
answer for that interrupted call.

The [journal](reasoning_budget_interrupted_v1/journal.jsonl) is preserved byte-for-byte,
SHA-256 `a603e5625d19dbc14da801ebd597c22c71c5f21e9a774db3f4a281de73a929e4`.
The separate [operator stop metadata](reasoning_budget_interrupted_v1/interruption.json)
was written **after** termination, not fabricated as a successful runner event.
The recorded proposals and parity events account for **at least 3856** new recurrent
calls. In-flight, unrecorded work may also have executed; total actual cost is unknown.
The old baseline's 16,384 calls are still prior work, not newly executed here.
CNF's completed segment recorded 9217.274 seconds including terminal parity and
instrumented generation; no finalized planning or full-pair runtime is available.
Post-run checkpoint verification and the runner's terminal identity/report phase
were not reached. Current source/journal integrity audits do not retroactively
supply that missing terminal verification.

No full-answer pair comparison, task-suitability conclusion, semantic stall,
rescue or TEBDİL efficacy result follows. The incomplete result is retained, not
retried or presented as completion. The unpublished two-case completion viewer
was removed; Pages instead identifies the interruption and links its partial records.

## Verification during the hold

Only existing-artifact/static/recorded-browser checks are authorized. The full
334-test result above predates the hold and is **not** a fresh full-suite claim for
the release. Model/mini-model execution and tokenizer integration tests are not rerun.
Post-stop artifact-only check: `tests/test_site_release.py -m "not real_model"`
passed **45 tests, 2 deselected in 8.97 seconds**. This does not verify model
execution. The latest interruption notice/archive has not yet been verified on
the public deployment.
