# Reasoning baseline budget diagnostic

Status: protocol/runner/tests verified before inference; no new completion observation yet.
[Frozen protocol](../docs/reasoning_budget_protocol.md).

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
