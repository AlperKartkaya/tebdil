# Reasoning dataset sources

Updated 2026-11-07. Status describes this repository, not benchmark-wide evaluation.
**Used (adapted)** means local authored/modified questions, not the official split or an official benchmark score.
Sources: `src/tebdil/tasks/{reasoning,literature_reasoning,deceptive_reasoning}.py` and frozen manifests in `configs/`.
**Planned** is a candidate, gated on license, pinned revision, item IDs, deduplication and a frozen evaluator.
**Will not use** means outside the current study, not a quality judgment. Unknown licensing blocks new redistribution; public access is not permission.

## Current sources and priority candidates

| Source | Status | License / provenance boundary | Content (description; size; format) |
|---|---|---|---|
| [TEBDIL synthetic tasks](../src/tebdil/tasks/) | Used | Apache-2.0 repository; third-party adaptations retain upstream obligations | Arithmetic, logic, graphs, CNF and key planning; manifest-specific counts; Python generators / JSON. |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Used (inspiration); planned official evaluation | MIT, verified dataset card | Grade-school multistep arithmetic; 7,473 train + 1,319 test; JSONL / Parquet. Local `gsm8k_multistep` tasks are not official GSM8K items. |
| [GSM-Symbolic / GSM-NoOp](https://huggingface.co/datasets/apple/GSM-Symbolic) | Used (adapted) | Verify exact configuration's terms before redistribution | Parameterized arithmetic and irrelevant-information variants; release/configuration-specific size; structured question/answer records. |
| [Audited GSM-NoOp](https://huggingface.co/datasets/Experimental-Orange/gsm-noop-audited) | Used (selected IDs 60, 10, 89) | Unverified; audit upstream and derivative terms | Audited distractor arithmetic; three local selected items; dataset records. |
| [ReasoningTrap](https://github.com/ReasoningTrap/ReasoningTrap) | Used (adapted and selected) | Verify each dataset's terms; code license alone is insufficient | Modified AIME, MATH500 and PuzzleTrivial questions; local source labels identify selected items; structured datasets. |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Used (selected) | Card does not declare a license; verify original MATH rights before further redistribution | 500 competition-math test questions; JSON with solution, answer, level, subject and original ID. |
| [MATH](https://github.com/hendrycks/math) | Used (inspiration) | Check original dataset terms, not only repository code | 12,500 competition problems; JSON with worked solutions; local CRT questions are adaptations. |
| [FastChat MT-Bench](https://github.com/lm-sys/FastChat) | Used (selected via XERJ) | Apache-2.0 repository; preserve source attribution and local pin | 80 multi-turn prompts; JSONL; local provenance names question-file lines 24, 25, 33 and 35. XERJ is acquisition provenance, not the original dataset. |
| [SimpleBench](https://simple-bench.com/) | Used (selected and inspiration) | Unverified redistribution terms | Commonsense traps; public subset and held-out benchmark differ; local Q1/Q8 and authored variants. |
| [Cognitive Reflection Test](https://doi.org/10.1257/089533005775196732) | Used (adapted) | Article copyright; no blanket dataset license established | Original three-item reflection test and local variants; prose / local JSON. |
| [Knights and Knaves](https://en.wikipedia.org/wiki/Knights_and_Knaves) | Used (inspiration) | No blanket license for Smullyan books; local authored puzzles must be distinguished | Truth-teller/liar constraint puzzles; local small set; text and verified answers. |
| [CLRS](https://github.com/google-deepmind/clrs) | Used (inspiration) | Verify code/data terms before importing; no CLRS dataset import claimed | Algorithmic reasoning tasks; generated instances; local shortest-path puzzles are authored examples. |
| [ZebraLogic](https://huggingface.co/datasets/WildEval/ZebraLogic) | Used (inspiration); planned | Verify pinned dataset card | Constraint-satisfaction logic grids; local authored assignment puzzles; structured records. |
| [BBEH](https://github.com/google-deepmind/bbeh) | Used (adapted); planned | Verify data-specific terms before importing more | 23 hard reasoning tasks; JSON task files; local object-counting adaptation is not benchmark-wide evaluation. |
| [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) | Planned | Verify task-specific BIG-bench licenses | 23 challenging task subsets; up to 250 examples/task; JSON input/target. |
| [ProofWriter](https://allenai.org/data/proofwriter) | Planned | Verify release terms | Rule-based entailment with proof depths and unknown labels; configuration-dependent size; JSONL. |
| [FOLIO](https://huggingface.co/datasets/yale-nlp/FOLIO) | Planned | Verify pinned card | First-order logic entailment; roughly 1.4K examples; structured premises/conclusions. |
| [ProntoQA](https://github.com/asaparov/prontoqa) | Planned | Verify generator and data licenses separately | Generated deductive reasoning with controllable depth; size configurable; JSON. |
| [CRUXEval](https://github.com/facebookresearch/cruxeval) | Planned | Verify data-specific terms | 800 Python input/output reasoning functions; JSONL; requires sandboxed execution if executing code. |
| [MuSR](https://huggingface.co/datasets/TAUR-Lab/MuSR) | Planned | Verify pinned card | Narrative multistep reasoning; 756 problems across three domains; structured choices and answers. |
| [HotpotQA](https://hotpotqa.github.io/) | Will not use in current study | Verify data terms and Wikipedia attribution | About 113K Wikipedia multi-hop QA pairs; JSON supporting facts; retrieval confounds latent-intervention analysis. |
| [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) | Will not use in current study | Apache-2.0 repository; verify included source-data terms | Wikipedia/Wikidata multi-hop QA, not an 8K training set; JSON contexts and evidence paths; exact count depends on release. |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | Will not use in current study | Verify data-specific terms | Composed 2–4-hop reading comprehension, not science multiple choice; answerable/full variants; JSONL. |
| [StrategyQA](https://allenai.org/data/strategyqa) | Will not use in current study | Verify release terms | 2,780 implicit-reasoning yes/no questions; decomposition/evidence records; factual-recall confound. |
| [QASC](https://allenai.org/data/qasc) | Will not use in current study | Verify release terms | 9,980 eight-choice science questions plus fact corpus; JSON; retrieval confound. |
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | Will not use in current study | Verify pinned card | 7,787 science exam questions split Easy/Challenge; structured choices; knowledge-heavy. |
| [DROP](https://allenai.org/data/drop) | Planned | Verify release terms | Roughly 96K paragraph-based numerical/discrete QA items; JSON; requires span/set-aware scoring, not the current integer-only parser. |
| [ARC-CoT](https://huggingface.co/datasets/Locutusque/arc-cot) | Will not use | Unverified derivative rights | Synthetic rationales for ARC questions; release-dependent size; structured text. Avoid rationale-training/evaluation leakage. |

## Additional candidates retained for scope decisions

| Source | Status | License / provenance boundary | Content (description; size; format) |
|---|---|---|---|
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Will not use in current study | Verify dataset card independently of code | About 12K ten-choice knowledge/reasoning questions; Parquet. |
| [MMLU-Redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux) | Will not use in current study | Verify source and correction rights | Corrected MMLU subsets; release-specific counts; structured choices. |
| [LogiQA / LogiQA 2.0](https://github.com/csitfun/LogiQA2.0) | Planned | Verify data redistribution terms | Exam-derived logical reading comprehension; version-specific counts; JSON/text. |
| [ReClor](https://github.com/yuweihao/reclor) | Planned | Verify exam-derived data terms | 6,138 logical reading-comprehension questions; JSON. |
| [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | Will not use pending rights review | Exam-derived rights unresolved | Analytical reasoning games; grouped passages/questions; JSON. |
| [LiveBench](https://github.com/LiveBench/LiveBench) | Will not use in frozen study | Per-source terms; verify snapshot | Updating mixed-domain evaluation; counts change; structured records. |
| [LiveCodeBench](https://livecodebench.github.io/) | Will not use in current study | Per-problem contest rights | Time-split programming problems; rolling size; code/test records; requires execution sandbox. |
| [OlympiadBench](https://github.com/OpenBMB/OlympiadBench) | Will not use in current study | Verify source-problem rights | 8,476 bilingual math/physics problems, including images; JSON and assets. |
| [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | Will not use in current study | Verify gated dataset terms | 448 graduate science questions; Diamond subset 198; structured multiple choice; knowledge-heavy. |
| [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | Planned | Verify card and source rights | 800 theorem-grounded problems; mixed answer types; JSON and some images. |
| [AGIEval](https://github.com/ruixiangcui/AGIEval) | Will not use in current study | Source exam rights vary | Human exam tasks; subset-dependent counts; JSONL. |
| [CLUTRR](https://github.com/facebookresearch/clutrr) | Planned | Verify release terms | Synthetic family-relation reasoning; generated split sizes; CSV. |
| [LongBench](https://github.com/THUDM/LongBench) | Will not use in current study | Verify release-specific data terms | Long-context tasks; multiple release formats/counts; memory/context confounds current intervention study. |

## Admission rules

Freeze source revision, item IDs, hashes, license evidence, prompt, parser and split before inference. Split by underlying problem/template, not seed. Do not tune on reserved tests. Report invalid format, truncation, incorrect answers and correct answers separately. Keep adapted-item results separate from official benchmark results. Many license cells remain explicit verification gates: they are not assertions of permission.
