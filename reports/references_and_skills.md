# MVP references and skills

Reviewed 2026-10-30. References inform implementation patterns, not TEBDIL's experimental results. No upstream application code was copied and no runtime dependency was added.

## Reference coding

Three relevant repositories were cloned into the locally ignored `.research/refs/` directory. XERJ searches preceded the implementation handoff; direct inspection supplied exact source locations.

| Reference | Pinned revision and license | Pattern used in the refactor |
| --- | --- | --- |
| [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/tree/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d) | `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, Apache-2.0 | Preserve the genuine recurrent boundary: shared decoder layers, position/cache context, and normalization after each loop. This is a source reference, not a fresh model execution. |
| [Aim](https://github.com/aimhubio/aim/tree/6e098e38065364c76b2bb7c028f266e53b647642) | `6e098e38065364c76b2bb7c028f266e53b647642`, Apache-2.0 | Bind a displayed measurement to its run, metric name, context and step. Derive related panels from the same selected record rather than independently chosen defaults. |
| [Observable Plot](https://github.com/observablehq/plot/tree/535723d5e433727720d9b673c31622821bb03210) | `535723d5e433727720d9b673c31622821bb03210`, ISC | Synchronize selected point, visual emphasis and text readout. Keep an explicit selection control rather than requiring hover or exact point targeting. |

Exact source references:

- [Ouro `modeling_ouro.py:633–656`](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d/modeling_ouro.py#L633): the recurrent loop applies `self.norm` after the shared layer stack. A latent intervention must not omit this boundary or exchange unrelated position/cache state.
- [Aim `aim/sdk/run.py:395–417`](https://github.com/aimhubio/aim/blob/6e098e38065364c76b2bb7c028f266e53b647642/aim/sdk/run.py#L395): values belong to a named, contextualized sequence with an explicit step. TEBDIL reuses the principle without installing Aim.
- [Plot `docs/interactions/pointer.md:129–156`](https://github.com/observablehq/plot/blob/535723d5e433727720d9b673c31622821bb03210/docs/interactions/pointer.md#L129): point targeting has dead spots/coincident-point limitations; linked marks and the readout should share the focused datum. TEBDIL does not add Plot as a dependency.

The Ouro clone contains a Git LFS pointer, **not checkpoint weights**. The machine lacked `git-lfs`; source checkout used a disabled LFS filter without downloading weights. Its revision matches the project's pinned checkpoint source.

## XERJ evidence

Official documentation: [installation and agent guide](https://xerj.org/llms.txt). The official installer was inspected before execution; SHA-256 verification remained enabled. The task-local binary is `.research/bin/xerj`, version `v1.0.0-rc.100`. The node binds to loopback at `http://127.0.0.1:9237`, with storage outside the indexed project.

Reference indexing uses `--no-graph --no-semantic --workers 2`. Retrieval is lexical, not neural. Commands and outputs remain in ignored `.research/logs/`; no automatic feedback, external report, or PR was submitted.

```sh
xerj autoindex .research/refs/ouro --prefix mvpouro --url http://127.0.0.1:9237 --no-graph --no-semantic --workers 2
xerj autoindex .research/refs/aim --prefix mvpaim --url http://127.0.0.1:9237 --no-graph --no-semantic --workers 2
xerj autoindex .research/refs/plot --prefix mvpplot --url http://127.0.0.1:9237 --no-graph --no-semantic --workers 2
xerj autoindex map --prefix mvpouro --url http://127.0.0.1:9237 --json
xerj search --prefix mvpouro --url http://127.0.0.1:9237 'total_ut_steps hidden_states position_embeddings'
xerj search --prefix mvpaim --url http://127.0.0.1:9237 'def track sequence name context step'
xerj search --prefix mvpplot --url http://127.0.0.1:9237 'pointer transform focused input event'
```

Coverage verified from data maps:

- Project `tebdilmvp`: 124 files, 4,358 records; all 52 detected code files indexed, no junk. `.xerjignore` explicitly includes experiment results and artifacts otherwise excluded by Git. This is the preparation snapshot, not a claim that subsequent edits were reindexed.
- Ouro `mvpouro`: 10 selected files, 3,802 records; both Python source files indexed, no junk. The LFS pointer is merely text; indexing it does not verify model weights.
- Aim `mvpaim`: 1,980 unique files indexed, 11,173 records; all 1,628 detected code files indexed. Seven duplicate aliases and three junk files are explicitly recorded: empty `aim/cli/__init__.py`, `CHANGELOG.md` misclassified as YAML, and a binary `.eot` font. The changelog requires direct inspection if release history is needed. The completed-with-junk exit code was `3`, not an unqualified all-files success.
- Plot `mvpplot`: 1,586 unique files indexed, 550,900 records; all 617 detected code files indexed, 39 duplicate aliases, no whole files junked. Two individual records were junked, producing exit `3` (`completed-with-junk`). Final data map and search results were inspected; supported source coverage is complete for the selected corpus.

Hidden files/directories are excluded by XERJ. `.agents/`, `.github/` and tool/runtime configuration require direct inspection. Selected source coverage is not whole-filesystem coverage; binary tensors are not searchable hidden-state evidence.

## Five installed skills

The source trees and instructions were inspected before copying them into `.agents/skills/`. Each installed directory retains its upstream license and an `UPSTREAM.json` with repository, full commit, path and retrieval date. No remote installer was executed. Existing skills were not overwritten. Two installed instruction files have trailing whitespace normalized; their `UPSTREAM.json` records this non-semantic local change.

| Skill | Source and version | Scope and applied guidance |
| --- | --- | --- |
| **frontend-design** | [Anthropic](https://github.com/anthropics/skills/tree/ca45fc496a3436251f191354091d557da31c15b5/skills/frontend-design), commit `ca45fc496a3436251f191354091d557da31c15b5`; Apache-2.0 | Frontend design. Put the defining mechanism first, keep supporting content quiet, use restrained design tokens and plain user-facing names. |
| **accessibility** | [Addy Osmani](https://github.com/addyosmani/web-quality-skills/tree/afa8da942115f2961fdbfa80807ea0b232ff6c00/skills/accessibility), version 2.0, commit `afa8da942115f2961fdbfa80807ea0b232ff6c00`; MIT | Frontend accessibility. Native controls, associated labels, visible focus, skip navigation, reduced motion and keyboard checks. Automated checks do not establish WCAG conformance. |
| **webapp-testing** | [Anthropic](https://github.com/anthropics/skills/tree/ca45fc496a3436251f191354091d557da31c15b5/skills/webapp-testing), commit `ca45fc496a3436251f191354091d557da31c15b5`; Apache-2.0 | Frontend verification. Inspect rendered DOM before actions, test real selections, capture screenshots and console failures. |
| **scientific-visualization** | [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scientific-visualization), version 1.4, commit `5f2bc0341354fd5590aef76a533ada1e1e5b4e62`; MIT | Research presentation. Preserve raw evidence; distinguish missing data from zero; label projections/illustrations; use consistent comparison scales and redundant color/text encoding. |
| **statistical-analysis** | [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/statistical-analysis), version 2.0, same commit; MIT | Statistical review. State paired sampling units, effect estimates, uncertainty and exploratory status. Do not equate a non-significant result with evidence of no effect. |

The first three are specifically frontend-focused; no React skill was added to a non-React application. Vercel's `web-design-guidelines` was inspected but not installed because its cloned repository supplied no license file. The explicitly MIT-licensed accessibility skill replaced it.

### Applied decisions and checks

- `DESIGN.md` applies frontend-design through a mechanism-first, continuous five-step journey, restrained tokens and progressive disclosure. Its accessibility contract specifies native controls, labels, visible focus, skip navigation, reduced motion, text alternatives and non-color-only outcomes.
- `scripts/build_landing_data.py` applies the tracing and scientific-integrity guidance: pairing validation, per-run source hashes, explicit recorded mode, missing/invalid measurement rejection, recomputed paired summaries and an exploratory-study label. These are export safeguards, not proof of a scientific effect.
- The webapp-testing guidance supplied the browser acceptance procedure: inspect rendered selectors, exercise real selections, inspect screenshots and capture failures. Completion of browser checks belongs to the final verification report, not the skill installation record.
- The installed scientific-visualization palette script was executed on the five `DESIGN.md` foreground colors against both proposed backgrounds. All ten normal-text contrast checks pass 4.5:1. Minimum contrast is **7.633:1** (`#b8a2ef` on `#131d2b`); muted text is **8.237:1** on that surface. Exact reports are `.research/logs/contrast-base.json` and `contrast-panel.json`.
- Six of ten color pairs trigger the palette script's grayscale-separation heuristic. Keep trajectory names and line styles; hue alone is insufficient. The numeric contrast check does not establish rendered-page accessibility, screenshot quality or WCAG conformance.

### Script and compatibility review

- `frontend-design` and `accessibility` contain instructions/references, not executable scripts.
- `webapp-testing` contains a server wrapper and three Playwright examples. All four Python files were read. The wrapper executes caller-supplied shell commands and terminates only its own child processes; it is not used with untrusted commands. Its `--help` command was verified.
- All nine scientific Python scripts/assets were parsed and inspected for imports, I/O, dynamic loading and process/network calls. Palette dynamic imports target the bundled local palette file. Figure helpers write explicitly requested outputs and refuse overwrite by default. Statistical helpers perform local numerical analysis. No install-time execution occurs.
- Scientific guidance is applied without installing its optional plotting/statistics environments or changing TEBDIL's lockfile. The palette CLI is standard-library-only; its `--help` command was verified. Optional scientific examples require separately documented dependencies and are not claimed as executed.
- Skill discovery is available to later agent turns. The current task explicitly read the installed instructions, so usage does not depend on restarting an agent.

K-Dense attribution: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). The [official arXiv record](https://arxiv.org/abs/2609.00065) was checked: current version v2, revised 2026-09-02. The paper reports documentation-corpus measurements, not demonstrated task-level reliability.

## Deployment references

The Pages verify/package/deploy jobs follow [GitHub's custom workflow requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). Checkout/setup-uv pins follow [Astral's official GitHub integration guide](https://docs.astral.sh/uv/guides/integration/github/). Deployment credentials are limited to packaging/deployment jobs; PR verification has read-only contents access. The workflow was inspected locally, not executed on GitHub during this refactor.

## Gate C application (2026-10-31)

Recovered the loopback XERJ node without altering another project's server.
`tebdilcontract` searches located `build_visit_state` and R0; the index explicitly
warned it predates gate B, so current files and all new code were read directly.
`mvpouro` and `mvpaim` searches reconfirmed the recurrent normalization boundary
and run/context/step identity pattern. No new dependency or upstream code copy.
Statistical-analysis guidance fixed sampling units/design before observations and
prevented inferential claims from twelve dependent conditions. Frontend/accessibility
guidance preserved the original page hierarchy, expandable evidence and keyboard
controls; webapp-testing guidance supplies fresh browser verification.

Gate D reused XERJ's pinned Ouro norm source and inspected the current vendored
cast order directly. Statistical-analysis guidance again fixes the three independent
prefixes versus twelve repeated conditions and prevents efficacy claims from
logit sensitivity. Frontend/accessibility guidance keeps the existing design and
puts detailed normalization evidence in its existing disclosure. No new skill,
dependency, upstream code copy or native normalization change.

## Completed-answer gate 1

Reused the project's already-verified `native_generate` rather than adding another
inference engine; existing `completed_final_integer_v1` remains unchanged. The
statistical-analysis skill informed partition freezing, denominator retention and
the explicit separation of smoke validation from efficacy/statistical support.
Ponytail guidance kept this to a record validator/scorer and native baseline script,
without a new framework/dependency. Webapp-testing guidance applies to the existing
keyboard/disclosure/download/responsive browser matrix; public verification is
reported separately from local tests. No new skill installation or upstream code
copy was needed; prior reference revisions/licenses above remain unchanged.

Gate 2 reuses the existing recurrent scheduler and tested operators directly.
It deliberately does not reuse the short-answer runner's retained per-prefix
full-vocabulary logit dictionary for long completions. No existing research code
was rewritten. Scientific guidance keeps input-distance random excursions distinct
from output-norm-matched controls, and per-prefix call matching distinct from
whole-answer compute matching. New mini-model tests carry no checkpoint claims.

Gate-2 observation: five real serial completions are preserved without post-hoc
parser repair or favorable filtering. The boxed-answer miss is reported separately
from reasoning correctness. Webapp-testing/release checks bind public downloads to
raw report hashes, all-arm journals and every reconstructed prefix. No new skills,
references, dependencies or copied upstream implementations were added in this gate.

Measurement v2 applies the existing statistical-analysis guidance: separate
completion, formatting and correctness; label known-output recalibration as
post-hoc rather than model improvement. Ponytail guidance kept the repair versioned
and additive because v1 sources/scores are immutable provenance inputs. Synthetic
negative cases precede implementation; webapp-testing verifies the separate raw
audit in the original disclosure. No new dependency, skill install or upstream
code copy was needed.

The baseline difficulty pilot reuses native generation and the original ledger
validator, adding independent forward observations instead of another inference
engine. Statistical guidance separates task headroom from format/truncation errors,
retains all twelve preselected attempts and avoids test-set peeking. Mini-model
checks precede real observations; no fixture is presented as a new Ouro completion.
No new dependency, cloud resource, skill installation or upstream code copy.

Baseline observation follow-through: all twelve attempts are published, including
the two unsupported-format outputs. Statistical guidance keeps their post-hoc
matching gold values separate from frozen scores; no arithmetic error, rescue
opportunity, uncertainty interval or intervention effect is inferred. Webapp-testing
adds the actual twelve-record download and count/cost reconciliation to the
existing browser matrix. No redesign or new dependency was needed.

Reasoning-stall study correction (2026-11-01): direct code inspection found that
`compute_stagnation_ratio` is an intra-prefix depth-update norm, while the current
adaptive trigger uses entropy/margin, not CoT progress. Statistical guidance keeps
recovery distinct from efficiency and forbids outcome-based case/trigger selection.
The [draft](../docs/reasoning_stall_study.md) cites primary papers inspected directly:
[Turpin et al.](https://arxiv.org/abs/2305.04388) for CoT faithfulness limitations and
[Chen et al.](https://arxiv.org/abs/2412.21187) for process/outcome compute efficiency.
Neither paper supplies Ouro or TEBDİL evidence. No source code was copied, dependency
installed, new generation experiment launched or monitoring functionality claimed
in this correction.

Prompt-started reasoning gate: XERJ retrieval over the pinned Ouro tokenizer sources
was followed by direct inspection of the cached chat template. True appends an
opening tag to the input; it is not a model-internal guarantee of useful reasoning.
Ponytail guidance keeps the fix additive and reuses the native decoder, ledger
validator and v2 integer grammar. Regression fixtures precede the new implementation.
No historical hash-bound source was changed, new dependency installed or upstream
code copied. The old False-mode artifacts are not rescored as True-mode runs.
