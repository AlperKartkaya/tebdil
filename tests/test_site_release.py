"""Release contract for the static TEBDIL research observatory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
INDEX = SITE / "index.html"
MANIFEST = SITE / "static" / "tebdil_data.js"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ORIGINAL_INLINE_CSS_SHA256 = "38da19b42714bfb06f0431c6d0fcb438e1be17b615fdbd3112b4450a46f69d87"
ORIGINAL_IDS = {
    "top", "tocRail", "tocToggle", "tocLinks", "main-content", "demo",
    "stageStepBadge", "stageStepLabel", "playLoopBtn", "playLoopIcon", "playLoopText",
    "caseSelect", "methodSelect", "liveProblemHeader", "liveGoldBadge", "livePromptBody",
    "liveFormulaPill", "excursionStageSvg", "stageStepNarrative", "pcaVarLabel",
    "pcaPlotSvg", "telemetryStepTag", "telemetryGrid", "terminalCallCost",
    "terminalMainText", "terminalMethodBadges", "abstract", "architecture", "gIndigo",
    "gSky", "theory", "patterns", "results", "phase78TestTable", "paretoChartSvg",
    "familyBreakdownTable", "phase5DonorTable", "phase6SelectiveTable",
    "phase3ValidationTable", "scaling", "gridMetricSelect", "alphaBetaHeatmap",
    "dimAblationTable", "case-study", "examples", "explorerSplitFilter",
    "explorerFamilyFilter", "explorerMethodFilter", "explorerCountBadge",
    "traceExplorerTable", "traceExplorerBody", "roadmap", "bibtex", "copyBibBtn",
    "bibtexCode",
}
ORIGINAL_SECTIONS = [
    "top", "demo", "abstract", "architecture", "theory", "patterns", "results",
    "scaling", "case-study", "examples", "roadmap", "bibtex",
]

SPEC = importlib.util.spec_from_file_location("build_landing_data", ROOT / "scripts/build_landing_data.py")
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
check = BUILDER.check
ledger_matches = BUILDER.ledger_matches
validate_rows = BUILDER.validate_rows


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.local_assets: list[str] = []
        self.runtime_assets: list[str] = []
        self.links: list[str] = []
        self.sections: list[str] = []
        self.attributes: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        element_id = values.get("id")
        if element_id:
            self.ids.append(element_id)
            self.attributes[element_id] = values
        if tag == "section" and element_id:
            self.sections.append(element_id)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])

        targets: list[str] = []
        if tag in {"script", "img", "source"} and values.get("src"):
            targets.append(values["src"])
        rel = set(values.get("rel", "").split())
        if tag == "link" and values.get("href") and rel & {
            "stylesheet",
            "icon",
            "preload",
            "modulepreload",
        }:
            targets.append(values["href"])
        for target in targets:
            if urlsplit(target).scheme or target.startswith("//"):
                self.runtime_assets.append(target)
            elif not target.startswith(("data:", "#")):
                self.local_assets.append(target)


def parse_site() -> SiteParser:
    parser = SiteParser()
    parser.feed(INDEX.read_text(encoding="utf-8"))
    return parser


def load_manifest() -> dict:
    text = MANIFEST.read_text(encoding="utf-8").strip()
    prefix = "window.TEBDIL_DATA = "
    assert text.startswith(prefix), "manifest must use the documented global assignment"
    return json.loads(text[len(prefix) :].removesuffix(";"))


def trace_path(run: dict) -> Path:
    url = run["trace_url"]
    parsed = urlsplit(url)
    assert not parsed.scheme and not parsed.netloc and not url.startswith("/")
    path = (SITE / parsed.path).resolve()
    assert path.is_relative_to(SITE.resolve()), f"trace escapes site root: {url}"
    return path


def test_original_page_structure_is_preserved() -> None:
    parser = parse_site()
    assert len(parser.ids) == len(set(parser.ids)), "HTML IDs must be unique"
    assert ORIGINAL_IDS <= set(parser.ids), "the pre-refactor page surfaces must remain available"
    assert parser.sections == ORIGINAL_SECTIONS


def test_original_inline_design_is_preserved_exactly() -> None:
    html = INDEX.read_text(encoding="utf-8")
    inline_styles = re.findall(r"<style>(.*?)</style>", html, flags=re.DOTALL)
    assert len(inline_styles) == 1
    assert hashlib.sha256(inline_styles[0].encode()).hexdigest() == ORIGINAL_INLINE_CSS_SHA256
    assert 'class="hero-banner"' in html
    assert 'class="pub-links"' in html
    assert "TEBDİL: Self-Directed &amp; Cooperative Latent Excursions" in html


def test_execution_contract_is_separate_and_traceable() -> None:
    report_path = ROOT / "reports/ouro_execution_contract.json"
    report = json.loads(report_path.read_text())
    public = SITE / "static/evidence/ouro_execution_contract.json"
    assert public.read_bytes() == report_path.read_bytes()
    assert report["model_id"] == "ByteDance/Ouro-1.4B-Thinking"
    assert report["all_passed"] is True
    assert report["cache_mode"] == "disabled"
    assert len(report["validations"]) == 6
    for case in report["validations"]:
        assert case["all_passed"] and len(case["steps"]) == 4
        assert all(step["passed"] for step in case["steps"])
    assert len(report["cross_precision_diagnostic"]) == 3
    for source, digest in report["source_sha256"].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest)
    html = INDEX.read_text()
    assert 'id="ouroExecutionContract"' in html
    assert "not improved reasoning" in html
    assert any(e["source"] == "reports/ouro_execution_contract.json" for e in load_manifest()["evidence"])


def test_pilot_calibration_prefixes_and_counts_are_traceable() -> None:
    from tebdil.core.state import hash_tensor
    import torch

    path = ROOT / "reports/ouro_baseline_calibration.json"
    report = json.loads(path.read_text())
    assert (SITE / "static/evidence" / path.name).read_bytes() == path.read_bytes()
    assert report["independent_problems"] == 3 and len(report["records"]) == 9
    for row in report["records"]:
        assert row["trajectory_semantics"] == "full_prefix_refork_per_generated_token"
        assert len(row["token_ledger"]) == len(row["generated_token_ids"])
        assert row["recurrent_block_calls"] + row["scoring_recurrent_calls"] == row["executed_recurrent_calls_including_scoring"]
        for k, entry in enumerate(row["token_ledger"]):
            prefix = row["prompt_token_ids"] + row["generated_token_ids"][:k]
            assert entry["prefix_length"] == len(prefix)
            assert entry["prefix_token_ids_sha256"] == hash_tensor(torch.tensor([prefix]), length=64)
            assert entry["branch_seed"] == report["seed"] + 131 * k
    for budget in (2, 64):
        raw = [r for r in report["records"] if r["prompt_format"] == "raw" and r["max_output_tokens"] == budget]
        assert len(raw) == 3 and sum(r["strict_correct"] for r in raw) == 1
    chat = [r for r in report["records"] if r["prompt_format"] == "chat"]
    assert len(chat) == 3 and all(r["stop_reason"] == "max_output_tokens" for r in chat)



def test_native_completion_preserves_actual_prefixes_and_journal() -> None:
    from tebdil.core.state import hash_tensor
    import torch

    path = ROOT / "reports/ouro_baseline_completion.json"
    report = json.loads(path.read_text())
    initial_path = ROOT / "reports/ouro_baseline_calibration.json"
    initial = json.loads(initial_path.read_text())
    reference = {r["problem_id"]: r for r in initial["records"] if r["prompt_format"] == "chat"}
    journal = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert journal[0]["event"] == "start"
    for key, value in journal[0].items():
        if key != "event":
            assert report[key] == value
    assert report["reference_calibration_sha256"] == hashlib.sha256(initial_path.read_bytes()).hexdigest()
    assert report["checkpoint_sha256"] == initial["checkpoint_sha256"]
    assert report["model_id"] == initial["model_id"] and report["model_revision"] == initial["model_revision"]
    assert report["cache_mode"] == "disabled" and report["max_output_tokens"] == 2048
    assert report["independent_problems"] == len(report["records"]) == 3
    assert {r["problem_id"] for r in report["records"]} == set(reference)
    assert [event["record"] for event in journal[1:] if event["event"] == "completion"] == report["records"]
    for source, digest in report["source_sha256"].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest)
    for row in report["records"]:
        old = reference[row["problem_id"]]
        assert row["prompt_token_ids"] == old["prompt_token_ids"]
        assert row["gold_answer"] == old["gold_answer"] and row["split"] == "pilot"
        assert len(row["token_ledger"]) == len(row["generated_token_ids"])
        assert row["recurrent_block_calls"] == 4 * len(row["generated_token_ids"])
        assert row["strict_correct"] == (row["final_integer"] is not None and int(row["final_integer"]) == int(row["gold_answer"]))
        n = min(len(old["generated_token_ids"]), len(row["generated_token_ids"]))
        assert row["reference_prefix_tokens_compared"] == n
        divergence = next((k for k in range(n) if old["generated_token_ids"][k] != row["generated_token_ids"][k]), None)
        assert row["first_divergent_token_vs_adapter"] == divergence
        for k, entry in enumerate(row["token_ledger"]):
            prefix = row["prompt_token_ids"] + row["generated_token_ids"][:k]
            assert entry["processed_input_ids"] == prefix
            assert entry["prefix_length"] == entry["processed_sequence_length"] == len(prefix)
            assert entry["prefix_token_ids_sha256"] == hash_tensor(torch.tensor([prefix]), length=64)
            assert entry["calls_before"] == 4 * k and entry["recurrent_block_calls"] == 4
    assert (SITE / "static/evidence" / path.name).read_bytes() == path.read_bytes()
    assert any(e["source"] == "reports/ouro_baseline_completion.json" for e in load_manifest()["evidence"])



def test_completion_rescoring_is_separate_and_explicitly_posthoc() -> None:
    from tebdil.tasks.completion import extract_completed_integer

    path = ROOT / "reports/ouro_completion_scoring.json"
    report = json.loads(path.read_text())
    raw_path = ROOT / report["input_source"]
    raw = json.loads(raw_path.read_text())
    assert report["posthoc"] is True
    assert report["answer_extraction"] == "completed_final_integer_v1"
    assert report["input_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert report["independent_problems"] == len(report["records"]) == len(raw["records"]) == 3
    for source, digest in report["source_sha256"].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest)
    for row, native in zip(report["records"], raw["records"], strict=True):
        assert row["problem_id"] == native["problem_id"] and row["split"] == "pilot"
        completed = native["generated_token_ids"][-1] == report["eos_token_id"]
        assert row["completed"] == completed
        answer = extract_completed_integer(
            native["decoded_text_with_special_tokens"], report["eos_token"], completed=completed,
        )
        assert row["final_integer"] == answer
        assert row["format_valid"] == (answer is not None)
        assert row["correct"] == (answer is not None and int(answer) == int(native["gold_answer"]))
        assert row["original_strict_correct"] == native["strict_correct"]
    assert (SITE / "static/evidence" / path.name).read_bytes() == path.read_bytes()
    assert any(e["source"] == "reports/ouro_completion_scoring.json" for e in load_manifest()["evidence"])


def test_primary_controls_and_states_are_present() -> None:
    parser = parse_site()
    required = {
        "runSelect",
        "caseSelect",
        "methodSelect",
        "tokenSelect",
        "stepSelect",
        "loadStatus",
        "retryLoad",
        "terminalMainText",
        "terminalMethodBadges",
        "selectedTraceJson",
        "downloadTrace",
    }
    assert required <= set(parser.ids)
    assert parser.attributes["loadStatus"].get("role") == "status"


def test_javascript_does_not_target_missing_elements() -> None:
    parser = parse_site()
    source = (SITE / "static/app.js").read_text(encoding="utf-8")
    references = set(
        re.findall(r'(?:el|select|button|getElementById)\(\s*["\']([^"\']+)["\']', source)
    )
    assert references <= set(parser.ids), f"missing JS targets: {sorted(references - set(parser.ids))}"


def test_mechanism_does_not_describe_controls_as_excursion_returns() -> None:
    script = r"""
const fs = require("fs");
const source = fs.readFileSync("site/static/app.js", "utf8");
const start = source.indexOf("  function mechanism(row, tel=telemetry())");
const end = source.indexOf("\n  function renderStageSvg", start);
if (start < 0 || end < 0) throw new Error("mechanism must remain independently testable");
const state = {step: 2};
const num = (value, digits=3) => typeof value === "number" ? value.toFixed(digits) : "Unavailable";
const decision = () => null;
const token = () => null;
const telemetry = () => null;
eval(source.slice(start, end).trim() + `
const noise = mechanism({method:"R0"}, null);
if (noise.formula !== "A′ = F(A + ε)" || !/input/i.test(noise.note)) throw new Error("R0 must label actual input noise");
for (const method of ["M0", "M1", "C0", "R0", "P0", "P1", "S0"]) {
  const model = mechanism({method}, null);
  if (model.kind === "excursion" || /computed update returns|return update/i.test(model.note)) {
    throw new Error(method + " must not claim an excursion return");
  }
}
for (const method of ["E1", "E1_no_curv", "E1_rand_curv"]) {
  const model = mechanism({method}, {effective_alpha: 0.5, beta: 0.6, visit_state_hash: "visit", receiving_state_hash: "a"});
  if (model.kind !== "excursion" || !/return/i.test(model.note + " " + model.labels.join(" "))) {
    throw new Error(method + " must explain the return to the receiving trajectory");
  }
}
const noOp = mechanism({method: "E1"}, {effective_alpha: 0, beta: 0.6, visit_state_hash: "a", receiving_state_hash: "a"});
if (noOp.kind !== "normal" || !/identity|no-op/i.test(noOp.note + " " + noOp.labels.join(" "))) throw new Error("identity intervention must remain a no-op");
`);
"""
    completed = subprocess.run(
        ["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_site_is_subpath_safe_and_has_no_cdn_runtime_dependency() -> None:
    parser = parse_site()
    assert parser.runtime_assets == []
    for url in parser.local_assets:
        assert not url.startswith("/"), f"root-absolute asset breaks GitHub Pages: {url}"
        path = (SITE / urlsplit(url).path).resolve()
        assert path.is_relative_to(SITE.resolve())
        assert path.is_file(), f"missing local asset: {url}"


def test_initial_static_payload_is_bounded_and_traces_stay_lazy() -> None:
    parser = parse_site()
    blocking = {INDEX, *(SITE / urlsplit(url).path for url in parser.local_assets)}
    total = sum(path.stat().st_size for path in blocking)
    assert total <= 4 * 1024 * 1024, f"initial payload is {total / 1024:.1f} KiB"
    assert MANIFEST.stat().st_size <= 100 * 1024
    assert not any(path.is_relative_to(SITE / "static/traces") for path in blocking)


def test_recorded_mode_is_not_presented_as_live_simulation() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert re.search(r"recorded experiment", html, re.IGNORECASE)
    assert not re.search(r"live simulator", html, re.IGNORECASE)


def test_repository_source_links_point_to_real_paths() -> None:
    marker = "https://github.com/AlperKartkaya/tebdil/"
    for url in parse_site().links:
        if not url.startswith(marker):
            continue
        relative = url[len(marker) :]
        for prefix in ("blob/main/", "tree/main/"):
            if relative.startswith(prefix):
                assert (ROOT / relative[len(prefix) :]).exists(), f"broken repository link: {url}"


def test_manifest_and_lazy_trace_files_are_self_consistent() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] == 2
    runs = manifest["runs"]
    assert runs

    for run_id, run in runs.items():
        assert run["run_id"] == run_id
        assert run["label"] and run["split"]
        assert isinstance(run["methods"], dict) and run["methods"]
        assert SHA256.fullmatch(run["trace_sha256"])
        assert SHA256.fullmatch(run["source_trace_sha256"])
        assert isinstance(run["ledger_mismatches"], int)

        path = trace_path(run)
        assert path.is_file()
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == run["trace_sha256"]
        records = json.loads(raw)
        assert isinstance(records, list) and records
        assert len(records) == run["record_count"]
        assert sorted({row["problem_id"] for row in records}) == sorted(run["problem_ids"])
        assert len(run["problem_ids"]) == run["num_problems"]
        assert sorted({row["seed"] for row in records}) == sorted(run["seeds"])
        assert {row["method"] for row in records} == set(run["methods"])


def test_manifest_metrics_and_paired_keys_match_exported_records() -> None:
    for run in load_manifest()["runs"].values():
        records = json.loads(trace_path(run).read_text(encoding="utf-8"))
        methods = run["methods"]
        paired_sets: list[set[tuple[str, int]]] = []
        for method, summary in methods.items():
            rows = [row for row in records if row["method"] == method]
            keys = [(row["problem_id"], row["seed"]) for row in rows]
            assert len(keys) == len(set(keys)), f"duplicate paired key in {run['run_id']}/{method}"
            paired_sets.append(set(keys))
            assert summary["num_instances"] == len(rows)
            assert math.isclose(
                summary["accuracy"], sum(bool(row["is_correct"]) for row in rows) / len(rows)
            )
            assert math.isclose(
                summary["mean_recurrent_calls"],
                sum(row["recurrent_block_calls"] for row in rows) / len(rows),
            )
        assert all(keys == paired_sets[0] for keys in paired_sets[1:])


def test_ledger_mismatch_count_is_explicit_and_correct() -> None:
    for run in load_manifest()["runs"].values():
        records = json.loads(trace_path(run).read_text(encoding="utf-8"))
        mismatches = sum(not ledger_matches(row) for row in records)
        assert run["ledger_mismatches"] == mismatches


def test_evidence_links_are_local_and_hash_verified() -> None:
    evidence = load_manifest()["evidence"]
    assert isinstance(evidence, list) and evidence
    for item in evidence:
        assert item["label"] and SHA256.fullmatch(item["sha256"])
        url = item["url"]
        assert not url.startswith("/") and not urlsplit(url).scheme
        path = (SITE / urlsplit(url).path).resolve()
        assert path.is_relative_to(SITE.resolve()) and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def recording(method: str = "S0", problem_id: str = "p1") -> dict:
    return {
        "problem_id": problem_id,
        "seed": 42,
        "method": method,
        "is_correct": True,
        "recurrent_block_calls": 4,
        "token_ledger": [{"recurrent_block_calls": 4}],
    }


def test_builder_rejects_duplicate_recording_keys() -> None:
    row = recording()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows([row, row.copy()])


def test_builder_rejects_unpaired_methods() -> None:
    with pytest.raises(ValueError, match="Unpaired"):
        validate_rows([recording(), recording("E1", problem_id="p2")])


def test_builder_rejects_missing_required_measurements() -> None:
    row = recording()
    del row["recurrent_block_calls"]
    with pytest.raises((KeyError, ValueError)):
        validate_rows([row])


def test_parallel_reference_budget_is_not_verified_executed_cost() -> None:
    row = recording("P1")
    row.update(
        {
            "derived_parallel_selection": True,
            "token_ledger": [{"recurrent_block_calls": 1}],
            "reference_parallel_budget_ledger": [{"recurrent_block_calls": 4}],
        }
    )
    assert not ledger_matches(row), "parallel selection cost stays unverified until all source compute is charged"


def test_committed_export_check_needs_no_ignored_raw_results(tmp_path: Path) -> None:
    shutil.copytree(SITE, tmp_path / "site")
    manifest = check(tmp_path)
    assert manifest["schema_version"] == 2


def test_committed_export_check_rejects_tampered_trace(tmp_path: Path) -> None:
    shutil.copytree(SITE, tmp_path / "site")
    manifest = check(tmp_path)
    run = next(iter(manifest["runs"].values()))
    path = tmp_path / "site" / run["trace_url"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum|match"):
        check(tmp_path)


def test_original_page_does_not_restore_stale_scientific_claims() -> None:
    html = INDEX.read_text(encoding="utf-8")
    script = (SITE / "static/app.js").read_text(encoding="utf-8")
    assert "controllerComparisonSummary" in html
    assert "controller.methods[method]" in script
    assert "it does not normalize or clip the visited update" in html
    assert "\\gamma_{\\max}" not in html
    assert "11.08 recurrent calls" not in html
    assert "confidence-weighted vote" not in html
    assert "No visit point is synthesized" in script
    assert 'branch!=="VISIT"' in script
    assert 'snapshots().find((item) => item.step === state.step)' in script


def test_legacy_heuristic_decisions_map_to_exact_recorded_step() -> None:
    script = r"""
const fs = require('fs'), assert = require('node:assert/strict');
const source = fs.readFileSync('site/static/app.js', 'utf8');
const start = source.indexOf('  function decision()');
const end = source.indexOf('\n  function ledgerMatches', start);
assert(start >= 0 && end > start);
let samples = [];
const state = {step: 2}, snapshots = () => samples;
eval(source.slice(start, end) + `
for (const [action, expected] of [['visit','EXCURSION'], ['no_visit','CONTINUE']]) {
  samples = [{step: 3, policy_decision: {action}}];
  state.step = 2; assert.equal(decision().action_type, expected);
  state.step = 1; assert.equal(decision(), null);
}
samples = [{step: 2, policy_decision: {action_type:'EXIT', state_features:{step:2}}}];
state.step = 2; assert.equal(decision().action_type, 'EXIT');
`);
"""
    completed = subprocess.run(
        ["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_method_comparison_preserves_context_including_missing_snapshots() -> None:
    script = r"""
const fs=require('fs'), assert=require('node:assert/strict');
const source=fs.readFileSync('site/static/app.js','utf8');
const state={run:'same-run',problem:'same-problem',seed:'42',method:'S1',token:3,step:5};
let entries=[], samples=[], options={}, renders=0, stopped=0;
const controls={methodSelect:{},tokenSelect:{},stepSelect:{}};
const select=id=>controls[id], ledger=()=>entries, snapshots=()=>samples;
const trace=()=>({step_snapshots:[{step:1},{step:2},{step:3},{step:4}]});
const setOptions=(target,items,value)=>{const id=Object.keys(controls).find(id=>controls[id]===target);options[id]=items;target.value=String(value);};
const stopAutoLoop=()=>stopped++, renderAll=()=>renders++;
const start=source.indexOf('  function changeMethod('), end=source.indexOf('\n  function renderAll()',start);
assert(start>=0&&end>start,'comparison controls need one shared changeMethod handler');
eval(source.slice(start,end)+`
changeMethod('E1');
assert.deepEqual(state,{run:'same-run',problem:'same-problem',seed:'42',method:'E1',token:3,step:5});
assert.equal(renders,1);assert.equal(stopped,1);
assert(options.tokenSelect.some(([v,label])=>v===3&&label.includes('not recorded')));
assert(options.stepSelect.some(([v,label])=>v===5&&label.includes('not recorded')));
assert.deepEqual(availableSteps(),[],'no fabricated default steps');
entries=[{decoded_token:'4'},{decoded_token:'8'}];state.token=1;samples=[{step:1},{step:2}];
changeMethod('S0');assert.equal(state.token,1);assert.equal(state.step,5);
assert(options.stepSelect.some(([v,label])=>v===5&&label.includes('not recorded')));
state.step=2;changeMethod('M1');assert.equal(state.step,2);assert.equal(state.token,1);
`);
assert(source.includes('changeMethod(select("methodSelect").value)'));
assert(source.includes('changeMethod(item.getAttribute("data-method")'));
"""
    completed = subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr



def test_demo_copy_does_not_fix_recording_dependent_parameters() -> None:
    html = INDEX.read_text(encoding="utf-8")
    script = (SITE / "static/app.js").read_text(encoding="utf-8")
    for stale in ("midpoint", "2-branch", "2-Branch Vote", "Best-of-2", r"n{=}36", r"\alpha{=}0.5"):
        assert stale not in html
    assert "Learned action score:" not in script
    assert "Recorded action score:" in script
    assert "not a probability" in script


def test_selected_step_narrative_does_not_invent_visits_for_controls() -> None:
    script = r"""
const fs=require('fs'),assert=require('node:assert/strict'),source=fs.readFileSync('site/static/app.js','utf8');
const start=source.indexOf('  function renderStepNarrative('),end=source.indexOf('\n  function renderPca(',start);
const target={},el=()=>target,state={step:2,token:0};
const num=(v,d=3)=>typeof v==='number'?v.toFixed(d):'Unavailable',esc=v=>String(v??'Unavailable');
let tel={step:2,effective_alpha:.65,beta:.65,update_visit_frob:72.143,returned_shift_vs_normal_frob:42.232};
const telemetry=()=>tel,decision=()=>({action_type:'CONTINUE',reason:'fixed_schedule',predicted_advantage:0});
const token=()=>({recurrent_block_calls:11,donor_calls:6,visit_calls:1}),availableSteps=()=>[1,2,3,4];
eval(source.slice(start,end)+`
for(const kind of ['mix','switch','noise','normal']){
 renderStepNarrative({method:'M1'},{kind,note:'Control; no anchored excursion'},[]);
 assert(!target.innerHTML.includes('Compute V(visit)'));
 assert(!target.innerHTML.includes('keep A as anchor'));
 assert(target.innerHTML.includes('Recorded action score:'));
}
renderStepNarrative({method:'E1'},{kind:'excursion',note:'Visited response'},[]);
assert(target.innerHTML.includes('42.232'));assert(target.innerHTML.includes('ID not captured'));
assert(target.innerHTML.includes('keep A as anchor'));
tel.skipped_unstable=true;
renderStepNarrative({method:'E1'},{kind:'normal',note:'Intervention skipped'},[]);
assert(!target.innerHTML.includes('Compute V(visit)'));assert(target.innerHTML.includes('Intervention skipped'));
`);
"""
    completed = subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_fixed_prefix_operator_evidence_is_separate_and_cost_reconciled() -> None:
    from tebdil.core.state import hash_tensor
    import torch

    path = ROOT / 'reports/ouro_operator_controls.json'
    raw = path.read_bytes()
    report = json.loads(raw)
    public = SITE / 'static/evidence' / path.name
    assert public.read_bytes() == raw
    assert report['all_checks_passed']
    assert report['independent_prefixes'] == 3
    assert report['cache_mode'] == 'disabled' and report['readout_mode'] == 'last_step'
    assert len(report['conditions']) == 12
    assert len(report['preparation']) == 6
    for source, digest in {**report['source_sha256'], **report['input_sha256']}.items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest), source
    journal = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert journal[0]['event'] == 'start'
    assert journal[0]['source_sha256'] == report['source_sha256']
    observed = [{k: v for k, v in r.items() if k != 'event'} for r in journal if r['event'] == 'condition']
    assert observed == report['conditions']
    physical = sum(r['physical_calls'] for r in journal if r['event'] == 'dtype_calls')
    accounted = sum(r['recurrent_calls'] for r in report['preparation'])
    pilots = json.loads((ROOT / 'reports/ouro_baseline_calibration.json').read_text())['records']
    ids = {r['problem_id']: r['prompt_token_ids'] for r in pilots if r['prompt_format'] == 'chat'}
    for condition in report['conditions']:
        assert all(condition['checks'].values())
        assert condition['prompt_token_ids'] == ids[condition['problem_id']]
        assert condition['prefix_sha256'] == hash_tensor(torch.tensor([condition['prompt_token_ids']]), 64)
        assert condition['first_answer_token_id'] == condition['answer_token_ids'][0]
        measurements = condition['measurements']
        assert len(measurements) == 14
        assert len({r['method'] for r in measurements}) == 14
        assert all(r['continuation_recurrent_calls'] == 1 for r in measurements)
        assert sum('target_output_norm' in r for r in measurements) == 6
        for measurement in measurements:
            if 'target_output_norm' in measurement:
                error = abs(measurement['realized_output_norm'] - measurement['target_output_norm']) / max(measurement['target_output_norm'], 1e-12)
                assert math.isclose(measurement['relative_norm_error'], error, abs_tol=1e-12)
        cost = condition['construction_and_reference_recurrent_calls'] + sum(r['continuation_recurrent_calls'] for r in measurements)
        assert cost == condition['total_recurrent_calls']
        accounted += cost
    assert physical == accounted == report['actual_recurrent_calls'] == 306
    assert any(e['source'] == 'reports/ouro_operator_controls.json' for e in load_manifest()['evidence'])


def test_historical_scope_contrast_is_not_presented_as_normalization_causality() -> None:
    html = INDEX.read_text()
    assert 'Sandwich RMSNorm Contraction vs. Token Scope' not in html
    assert 'not a causal test of normalization' in html


def test_normalization_evidence_binds_native_boundaries_and_ablations() -> None:
    import torch
    from tebdil.core.state import hash_tensor

    path = ROOT / 'reports/ouro_normalization.json'
    raw = path.read_bytes()
    report = json.loads(raw)
    assert (SITE / 'static/evidence' / path.name).read_bytes() == raw
    assert report['all_checks_passed'] and report['independent_prefixes'] == 3
    assert report['cache_mode'] == 'disabled' and report['readout_mode'] == 'last_step'
    assert len(report['conditions']) == 12 and len(report['preparation']) == 6
    for source, digest in {**report['source_sha256'], **report['input_sha256']}.items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest), source
    journal = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert journal[0]['event'] == 'start'
    assert journal[0]['source_sha256'] == report['source_sha256']
    assert [{k: v for k, v in r.items() if k != 'event'} for r in journal if r['event'] == 'condition'] == report['conditions']
    assert [b for r in journal if r['event'] == 'boundaries' for b in r['records']] == report['boundary_records']
    assert len(report['boundary_records']) == report['actual_recurrent_calls'] == 246
    assert all(b['native_formula_max_abs_error'] == 0 for b in report['boundary_records'])
    pilots = json.loads((ROOT / 'reports/ouro_baseline_calibration.json').read_text())['records']
    ids = {r['problem_id']: r['prompt_token_ids'] for r in pilots if r['prompt_format'] == 'chat'}
    accounted = sum(p['recurrent_calls'] for p in report['preparation'])
    for condition in report['conditions']:
        assert all(condition['checks'].values())
        assert condition['prompt_token_ids'] == ids[condition['problem_id']]
        assert condition['prefix_sha256'] == hash_tensor(torch.tensor([condition['prompt_token_ids']]), 64)
        assert len(condition['measurements']) == 9 and len(condition['visits']) == 5
        assert len({m['method'] for m in condition['measurements']}) == 9
        assert condition['total_recurrent_calls'] == condition['construction_and_reference_recurrent_calls'] + sum(m['continuation_recurrent_calls'] for m in condition['measurements'])
        boundaries = [b for b in report['boundary_records'] if b['problem_id'] == condition['problem_id'] and b['dtype'] == condition['dtype'] and b['radius'] == condition['max_relative_norm']]
        assert len(boundaries) == condition['total_recurrent_calls']
        for boundary in boundaries:
            assert all(len(batch) == len(condition['prompt_token_ids']) for batch in boundary['input_rms_by_token'])
        accounted += condition['total_recurrent_calls']
    assert accounted == 246
    assert any(e['source'] == 'reports/ouro_normalization.json' for e in load_manifest()['evidence'])
    assert '246 measured recurrent boundaries' in INDEX.read_text()


def test_completed_answer_smoke_is_frozen_replayable_and_separate_from_efficacy() -> None:
    import torch
    from tebdil.core.state import hash_tensor
    from tebdil.evaluation.completion import score_completion, summarize_completions, validate_task_manifest
    from tebdil.tasks.completion import extract_completed_integer

    path = ROOT / 'reports/ouro_completed_answer_smoke.json'
    report = json.loads(path.read_text())
    assert (SITE / 'static/evidence' / path.name).read_bytes() == path.read_bytes()
    assert report['all_checks_passed'] and all(report['checks'].values())
    assert report['protocol'] == 'completed_answer_v1' and report['cache_mode'] == 'disabled'
    assert report['model_revision'] == '3aaa2224253a92ca45cf2e3d427c360e1ef9c93d'
    assert report['recurrent_depth'] == 4 and report['max_output_tokens'] == 2048
    for source, digest in report['source_sha256'].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest)
    manifest = json.loads((ROOT / 'configs/completion_v1_tasks.json').read_text())
    validate_task_manifest(manifest)
    dev = {r['problem_id']: r for r in manifest['partitions']['development']}
    assert [r['problem_id'] for r in report['records']] == manifest['smoke_problem_ids']
    for record in report['records']:
        assert record['split'] == 'development' and record['method'] == 'S0'
        assert record['prompt'] == dev[record['problem_id']]['prompt']
        assert record['gold_answer'] == dev[record['problem_id']]['gold_answer']
        tokens = record['generated_token_ids']
        assert record['completed'] == (tokens[-1] == report['eos_token_id'])
        assert record['completed'], 'Gate 1 still needs a genuinely completed smoke'
        assert report['eos_token_id'] not in tokens[:-1]
        answer = extract_completed_integer(record['decoded_text_with_special_tokens'], report['eos_token'], completed=True)
        assert record['extracted_answer'] == answer
        assert record['format_valid'] == (answer is not None)
        assert record['format_valid'], 'Gate 1 needs a recognized completed answer'
        assert {k: record[k] for k in ('status', 'is_correct')} == score_completion(record, record['gold_answer'])
        assert record['recurrent_block_calls'] == len(tokens) * 4
        assert len(record['token_ledger']) == len(tokens)
        for k, entry in enumerate(record['token_ledger']):
            prefix = record['prompt_token_ids'] + tokens[:k]
            assert entry['prefix_length'] == entry['processed_sequence_length'] == len(prefix)
            assert entry['prefix_token_ids_sha256'] == hash_tensor(torch.tensor([prefix]), 64)
            assert entry['calls_before'] == k * 4 and entry['recurrent_block_calls'] == 4
            assert 'processed_input_ids' not in entry
        assert record['recurrent_token_positions'] == sum(e['prefix_length'] * 4 for e in record['token_ledger'])
    assert report['summary'] == summarize_completions(report['records'])
    assert report['actual_recurrent_calls'] == report['summary']['recurrent_block_calls']
    events = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'start' and events[-1]['event'] == 'finish'
    assert events[0]['source_sha256'] == report['source_sha256']
    assert [e['record'] for e in events if e['event'] == 'completion'] == report['records']
    assert events[-1]['summary'] == report['summary']
    assert any(e['source'] == 'reports/ouro_completed_answer_smoke.json' for e in load_manifest()['evidence'])


def test_completed_answer_evidence_stays_in_existing_disclosure() -> None:
    html = INDEX.read_text()
    assert 'Frozen completed-answer protocol' in html
    assert 'not an intervention-accuracy result' in html
    assert 'static/evidence/ouro_completed_answer_smoke.json' in html
    assert 'docs/completed_answer_protocol.md' in html
    disclosure = html.split('id="ouroExecutionContract"', 1)[1].split('</details>', 1)[0]
    assert 'docs/completed_answer_comparison_protocol.md' in disclosure
    assert 'reports/completed_answer_comparison_verification.md' in disclosure
    assert 'not an efficacy result' in disclosure
    assert 'format failure, not demonstrated wrong arithmetic' in disclosure
    assert 'static/evidence/ouro_completed_answer_comparison.json' in disclosure


def test_fixed_completion_evidence_retains_every_arm_and_reconciles_real_costs() -> None:
    import torch
    from tebdil.core.state import hash_tensor
    from tebdil.evaluation.comparison import METHOD_CALLS
    from tebdil.evaluation.completion import score_completion
    from tebdil.tasks.completion import extract_completed_integer

    path = ROOT / 'reports/ouro_completed_answer_comparison.json'
    report = json.loads(path.read_text())
    reference = json.loads((ROOT / report['baseline_reference']).read_text())
    assert report['all_checks_passed'] and report['cache_mode'] == 'disabled'
    assert report['protocol'] == 'completed_answer_comparison_smoke_v1'
    assert report['checkpoint_sha256'] == reference['checkpoint_sha256']
    assert report['model_revision'] == reference['model_revision']
    assert report['device'] == reference['device'] and report['torch_version'] == reference['torch_version']
    assert report['baseline_record'] == reference['records'][0]
    assert report['methods'] == ['B_only', 'M0', 'M1', 'E0', 'E1']
    assert [r['method'] for r in report['records']] == report['methods']
    assert report['parity']['generated_token_ids'] == report['baseline_record']['generated_token_ids'][:8]
    assert report['parity']['recurrent_block_calls'] == 32
    for source, digest in report['source_sha256'].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest), source
    events = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'start' and events[-1]['event'] == 'finish'
    assert events[0]['source_sha256'] == report['source_sha256']
    assert not any(e['event'] == 'failure' for e in events)
    assert [e['method'] for e in events if e['event'] == 'method_start'] == report['methods']
    assert [e['record'] for e in events if e['event'] == 'method_finish'] == report['records']
    for record in report['records']:
        assert record['problem_id'] == 'completion_dev_000' and record['split'] == 'development'
        assert record['prompt_token_ids'] == report['baseline_record']['prompt_token_ids']
        assert record['gold_answer'] == report['baseline_record']['gold_answer']
        tokens, method = record['generated_token_ids'], record['method']
        assert 0 < len(tokens) <= 2048
        assert record['completed'] == (tokens[-1] == reference['eos_token_id'])
        assert reference['eos_token_id'] not in tokens[:-1]
        assert record['stop_reason'] == ('eos_token' if record['completed'] else 'max_output_tokens')
        assert record['completed'] or len(tokens) == 2048
        answer = extract_completed_integer(record['decoded_text_with_special_tokens'], reference['eos_token'], completed=record['completed'])
        assert record['extracted_answer'] == answer and record['format_valid'] == (answer is not None)
        assert {k: record[k] for k in ('status', 'is_correct')} == score_completion(record, record['gold_answer'])
        calls = METHOD_CALLS[method]
        assert record['recurrent_block_calls'] == calls * len(tokens)
        assert len(record['token_ledger']) == len(tokens)
        for k, entry in enumerate(record['token_ledger']):
            prefix = record['prompt_token_ids'] + tokens[:k]
            assert entry['token_index'] == k and entry['token_id'] == tokens[k]
            assert entry['prefix_length'] == len(prefix)
            assert entry['prefix_token_ids_sha256'] == hash_tensor(torch.tensor([prefix]), 64)
            assert entry['branch_seed'] == 20261101 + 131 * k
            assert entry['calls_before'] == k * calls and entry['recurrent_block_calls'] == calls
            assert entry['selected_branch'] == ('B' if method == 'B_only' else 'A')
            assert SHA256.fullmatch(entry['selected_hidden_hash'])
            telemetry = entry['intervention']
            if method == 'B_only':
                assert telemetry is None
            else:
                assert telemetry['post_finite'] and not telemetry['skipped_unstable']
        assert record['recurrent_token_positions'] == sum(e['prefix_length'] * calls for e in record['token_ledger'])
        token_events = [{k: v for k, v in e.items() if k not in ('event', 'method')} for e in events if e['event'] == 'token' and e['method'] == method]
        assert token_events == record['token_ledger']
    assert report['actual_new_recurrent_calls'] == 32 + sum(r['recurrent_block_calls'] for r in report['records'])
    assert events[-1]['actual_new_recurrent_calls'] == report['actual_new_recurrent_calls']
    assert report['reused_baseline_recurrent_calls'] == report['baseline_record']['recurrent_block_calls']
    assert (SITE / 'static/evidence' / path.name).read_bytes() == path.read_bytes()
    assert any(e['source'] == 'reports/ouro_completed_answer_comparison.json' for e in load_manifest()['evidence'])


def test_measurement_v2_audit_is_separate_posthoc_and_hash_bound() -> None:
    from tebdil.tasks.completion_v2 import evaluate_completion_v2

    path = ROOT / 'reports/ouro_completion_v2_audit.json'
    report = json.loads(path.read_text())
    assert report['all_checks_passed'] and report['posthoc']
    assert report['new_model_executions'] == 0 and report['independent_problems'] == 1
    assert report['protocol'] == 'completed_answer_measurement_v2' and len(report['records']) == 6
    assert (SITE / 'static/evidence' / path.name).read_bytes() == path.read_bytes()
    for source, digest in {**report['source_sha256'], **report['input_sha256']}.items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest), source
    for row in report['records']:
        source = json.loads((ROOT / row['source']).read_text())
        raw = source['records'][row['record_index']]
        assert row['problem_id'] == raw['problem_id'] == 'completion_dev_000'
        assert row['method'] == raw['method']
        assert row['original_v1'] == {k: raw[k] for k in row['original_v1']}
        assert hashlib.sha256(raw['decoded_text_with_special_tokens'].encode()).hexdigest() == row['decoded_text_sha256']
        assert row['v2_diagnostic'] == evaluate_completion_v2(raw, '<|im_end|>', raw['gold_answer'])
    assert any(e['source'] == path.relative_to(ROOT).as_posix() for e in load_manifest()['evidence'])
    disclosure = INDEX.read_text().split('id="ouroExecutionContract"', 1)[1].split('</details>', 1)[0]
    assert 'Measurement v2 — offline format audit' in disclosure
    assert 'original v1 scores and model outputs remain unchanged' in disclosure
    assert 'static/evidence/ouro_completion_v2_audit.json' in disclosure
    assert 'docs/baseline_difficulty_protocol.md' in disclosure
    assert 'reports/baseline_difficulty_verification.md' in disclosure
    assert 'Baseline difficulty pilot — headroom unresolved' in disclosure
    assert '10 recognized correct, 0 recognized incorrect, 2 invalid format, 0 budget exhausted' in disclosure
    assert 'static/evidence/ouro_baseline_difficulty.json' in disclosure


def test_baseline_difficulty_retains_all_development_attempts_and_actual_native_work() -> None:
    import torch
    from tebdil.core.state import hash_tensor
    from tebdil.evaluation.completion import score_completion, summarize_completions
    from tebdil.tasks.completion_v2 import evaluate_completion_v2

    path = ROOT / 'reports/ouro_baseline_difficulty.json'
    report = json.loads(path.read_text())
    tasks = json.loads((ROOT / 'configs/completion_v1_tasks.json').read_text())['partitions']['development']
    reference = json.loads((ROOT / report['reused_reference']).read_text())
    assert report['all_checks_passed'] and report['protocol'] == 'development_baseline_difficulty_v1'
    assert report['answer_extraction'] == 'completed_final_integer_v2' and report['cache_mode'] == 'disabled'
    assert report['model_revision'] == reference['model_revision'] and report['checkpoint_sha256'] == reference['checkpoint_sha256']
    assert report['problem_ids'] == [t['problem_id'] for t in tasks]
    assert report['new_problem_ids'] == report['problem_ids'][1:]
    assert report['reused_problem_ids'] == report['problem_ids'][:1]
    assert [r['problem_id'] for r in report['records']] == report['problem_ids']
    assert (SITE / 'static/evidence' / path.name).read_bytes() == path.read_bytes()
    for source, digest in report['source_sha256'].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(digest), source
    events = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'start' and events[-1]['event'] == 'finish'
    assert events[0]['source_sha256'] == report['source_sha256']
    assert not any(e['event'] == 'failure' for e in events)
    finished = [e['record'] for e in events if e['event'] in ('reuse', 'completion')]
    assert finished == report['records']
    assert [e['problem_id'] for e in events if e['event'] == 'attempt_start'] == report['new_problem_ids']
    for k, (row, task) in enumerate(zip(report['records'], tasks, strict=True)):
        assert row['split'] == 'development' and row['method'] == 'S0'
        assert row['prompt'] == task['prompt'] and row['gold_answer'] == task['gold_answer']
        assert row['template_id'] == task['template_id']
        fields = evaluate_completion_v2(row, report['eos_token'], row['gold_answer'])
        assert {key: row[key] for key in fields} == fields
        old = row['v1_evaluation']
        assert old['answer_extraction'] == 'completed_final_integer_v1'
        assert {key: old[key] for key in ('status', 'is_correct')} == score_completion({**row, **old}, row['gold_answer'])
        tokens = row['generated_token_ids']
        assert 0 < len(tokens) <= report['max_output_tokens']
        assert row['completed'] == (tokens[-1] == report['eos_token_id'])
        assert report['eos_token_id'] not in tokens[:-1]
        assert row['completed'] or len(tokens) == report['max_output_tokens']
        assert row['stop_reason'] == ('eos_token' if row['completed'] else 'max_output_tokens')
        assert row['recurrent_block_calls'] == 4 * len(tokens)
        assert len(row['token_ledger']) == len(tokens)
        for i, entry in enumerate(row['token_ledger']):
            prefix = row['prompt_token_ids'] + tokens[:i]
            assert entry['prefix_length'] == entry['processed_sequence_length'] == len(prefix)
            assert entry['prefix_token_ids_sha256'] == hash_tensor(torch.tensor([prefix]), 64)
            assert entry['calls_before'] == i * 4 and entry['recurrent_block_calls'] == 4
        assert row['recurrent_token_positions'] == sum(e['prefix_length'] * 4 for e in row['token_ledger'])
        observed = [e for e in events if e['event'] == 'forward' and e['problem_id'] == row['problem_id']]
        if k == 0:
            assert row['origin'] == 'reused_gate_1' and not observed
            assert all(row[key] == value for key, value in reference['records'][0].items() if key not in fields)
        else:
            assert row['origin'] == 'new_native_generation' and row['independent_forward_checks_passed']
            assert [e['greedy_readout_token_id'] for e in observed] == tokens
            for observation, entry in zip(observed, row['token_ledger'], strict=True):
                assert all(observation[key] == entry[key] for key in ('prefix_length', 'prefix_token_ids_sha256', 'calls_before', 'recurrent_block_calls'))
    assert report['summary'] == summarize_completions(report['records'])
    for template, summary in report['by_template'].items():
        assert summary == summarize_completions([r for r in report['records'] if r['template_id'] == template])
    assert report['actual_new_recurrent_calls'] == sum(r['recurrent_block_calls'] for r in report['records'][1:])
    assert report['reused_recurrent_calls'] == reference['actual_recurrent_calls']
    assert report['summary']['recurrent_block_calls'] == report['actual_new_recurrent_calls'] + report['reused_recurrent_calls']
    assert events[-1]['summary'] == report['summary']
    assert events[-1]['actual_new_recurrent_calls'] == report['actual_new_recurrent_calls']
    assert any(e['source'] == path.relative_to(ROOT).as_posix() for e in load_manifest()['evidence'])


def test_reasoning_stall_focus_is_explicitly_proposed_not_existing_monitoring() -> None:
    note = (ROOT / 'docs/reasoning_stall_study.md').read_text()
    assert 'Draft study design; not a frozen protocol or an executed experiment' in note
    assert 'CoT repetition is a candidate signal, not proof of a reasoning stall' in note
    assert 'same observed prefix' in note
    assert 'ground truth, verifier feedback, future tokens or eventual outcome' in note
    assert 'correct-to-incorrect' in note
    disclosure = INDEX.read_text().split('id="ouroExecutionContract"', 1)[1].split('</details>', 1)[0]
    assert 'does not test reasoning stalls' in disclosure
    assert 'Proposed reasoning-recovery study — not yet run' in disclosure
    assert 'docs/reasoning_stall_study.md' in disclosure


def test_prompt_started_reasoning_real_smoke_preserves_actual_input_output_boundary() -> None:
    import torch
    from tebdil.core.state import hash_tensor
    from tebdil.evaluation.reasoning import PROMPT_SUFFIX, evaluate_reasoning_integer, split_prompt_started_reasoning

    path = ROOT / 'reports/ouro_prompt_started_reasoning.json'
    report = json.loads(path.read_text())
    assert report['protocol'] == 'prompt_started_reasoning_v1' and report['enable_thinking'] is True
    assert report['all_checks_passed'] and report['problem_ids'] == ['completion_dev_000']
    assert report['cache_mode'] == 'disabled' and report['recurrent_depth'] == 4
    assert report['readout_mode'] == 'last_step' and len(report['records']) == 1
    row = report['records'][0]
    assert row['split'] == 'development' and row['method'] == 'S0'
    assert row['enable_thinking'] and row['reasoning_open_in_prompt'] and row['independent_forward_checks_passed']
    assert row['prompt_text_with_special_tokens'].endswith(PROMPT_SUFFIX)
    assert row['reasoning_text'] + '</think>' + row['final_text'] + report['eos_token'] == row['decoded_text_with_special_tokens']
    assert not row['decoded_text_with_special_tokens'].startswith('<think>')
    assert row['prompt'] in row['prompt_text_with_special_tokens']
    reference = json.loads((ROOT / 'reports/ouro_completed_answer_smoke.json').read_text())
    assert row['prompt'] == reference['records'][0]['prompt'] and row['gold_answer'] == reference['records'][0]['gold_answer']
    assert report['checkpoint_sha256'] == reference['checkpoint_sha256']
    assert report['model_revision'] == reference['model_revision']
    for source, expected in report['source_sha256'].items():
        assert (ROOT / source).exists() and SHA256.fullmatch(expected), source
    parts = split_prompt_started_reasoning(row['decoded_text_with_special_tokens'], report['eos_token'], completed=row['completed'])
    assert all(row[key] == value for key, value in parts.items())
    scores = evaluate_reasoning_integer(row, row['gold_answer'])
    assert all(row[key] == value for key, value in scores.items())
    events = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'start' and events[-1]['event'] == 'finish'
    assert events[0]['source_sha256'] == report['source_sha256']
    assert not any(e['event'] == 'failure' for e in events)
    assert events[-1]['record'] == row
    observed = [e for e in events if e['event'] == 'forward']
    tokens = row['generated_token_ids']
    assert [e['greedy_readout_token_id'] for e in observed] == tokens
    assert 0 < len(tokens) <= report['max_output_tokens']
    assert row['completed'] == (tokens[-1] == report['eos_token_id'])
    assert report['eos_token_id'] not in tokens[:-1]
    assert row['completed'] or len(tokens) == report['max_output_tokens']
    assert len(row['token_ledger']) == len(tokens)
    for k, (entry, observation) in enumerate(zip(row['token_ledger'], observed, strict=True)):
        prefix = row['prompt_token_ids'] + tokens[:k]
        assert entry['prefix_length'] == entry['processed_sequence_length'] == len(prefix)
        assert entry['prefix_token_ids_sha256'] == hash_tensor(torch.tensor([prefix]), 64)
        assert entry['calls_before'] == 4*k and entry['recurrent_block_calls'] == 4
        assert all(observation[key] == entry[key] for key in ('prefix_length', 'prefix_token_ids_sha256', 'calls_before', 'recurrent_block_calls'))
    assert row['recurrent_token_positions'] == sum(4*e['prefix_length'] for e in row['token_ledger'])
    assert report['actual_new_recurrent_calls'] == events[-1]['actual_new_recurrent_calls'] == row['recurrent_block_calls'] == 4*len(tokens)
    assert (SITE / 'static/evidence' / path.name).read_bytes() == path.read_bytes()
    assert any(e['source'] == path.relative_to(ROOT).as_posix() for e in load_manifest()['evidence'])


def test_reasoning_tasks_are_unexecuted_development_prompts_not_new_model_results():
    path=SITE/'static/evidence/reasoning_v1_development.json'
    data=json.loads(path.read_text())
    assert data['status']=='unexecuted_task_prompts' and len(data['tasks'])==6
    assert all(t['problem_id'].startswith('reasoning_v1_development_') for t in data['tasks'])
    assert all(set(t)=={'problem_id','family','structural_level','prompt'} for t in data['tasks'])
    manifest=load_manifest()
    assert manifest['files']['static/evidence/reasoning_v1_development.json']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    html=INDEX.read_text()
    assert 'New reasoning tasks — frozen specification' in html
    assert 'Development prompts (no model results)' in html
    assert hashlib.sha256((ROOT/'configs/reasoning_v1_tasks.json').read_bytes()).hexdigest()=='016c4c8343fa3656825b63f3df6dc283d06f19ffaaba17714c8c163c10590e17'


def test_passive_reasoning_artifacts_reconcile_real_states_prefixes_costs_and_scores():
    import math
    import torch
    from tebdil.core.state import hash_tensor
    from tebdil.evaluation.monitoring import SNAPSHOT_TOKEN_INDICES
    from tebdil.tasks.reasoning import evaluate_reasoning_task
    folder=ROOT/'reports/passive_reasoning_v1'
    report=json.loads((folder/'report.json').read_text())
    assert report['protocol']=='passive_reasoning_v1' and report['all_checks_passed']
    assert report['enable_thinking'] is True and report['recurrent_depth']==4
    assert report['checkpoint_sha256']==json.loads((ROOT/'reports/ouro_prompt_started_reasoning.json').read_text())['checkpoint_sha256']
    assert all((ROOT/p).exists() and SHA256.fullmatch(h) for p,h in report['source_sha256'].items())
    tasks=json.loads((ROOT/'configs/reasoning_v1_tasks.json').read_text())['partitions']['development'][:2]
    assert report['selected_problem_ids']==[t['problem_id'] for t in tasks]
    assert len(report['records'])==2
    run=[json.loads(line) for line in (folder/'run.jsonl').read_text().splitlines()]
    assert [e['event'] for e in run]==['start','case_start','case_finish','case_start','case_finish','finish']
    assert run[-1]['report_sha256']==hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()
    for p in folder.iterdir():
        assert p.read_bytes()==(SITE/'static/evidence/passive_reasoning_v1'/p.name).read_bytes()
    calls=0
    for task,summary in zip(tasks,report['records'],strict=True):
        path=folder/summary['raw_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['raw_sha256']
        case=json.loads(path.read_text())
        assert case['all_checks_passed'] and case['source_sha256']==report['source_sha256']
        row,observations=case['record'],case['observations']
        assert row['problem_id']==task['problem_id'] and row['prompt']==task['prompt']
        assert row['split']=='development' and row['method']=='S0'
        assert all(row[k]==v for k,v in evaluate_reasoning_task(task,row).items())
        n=len(row['generated_token_ids'])
        assert n==summary['generated_tokens']==len(observations)==len(row['token_ledger'])
        assert row['recurrent_block_calls']==summary['recurrent_block_calls']==4*n
        assert row['status']==summary['status'] and row['reasoning_status']==summary['reasoning_status']
        assert n<=2048 and (row['completed'] or n==2048)
        journal=folder/summary['journal_file']
        assert hashlib.sha256(journal.read_bytes()).hexdigest()==summary['journal_sha256']
        events=[json.loads(line) for line in journal.read_text().splitlines()]
        assert [e['event'] for e in events]==['case_start']+['observation']*n+['case_finish']
        assert events[-1]['record']==row
        flags=[]
        for k,(entry,ledger) in enumerate(zip(observations,row['token_ledger'],strict=True)):
            prefix=row['prompt_token_ids']+row['generated_token_ids'][:k]
            assert entry['token_index']==entry['generated_prefix_token_count']==k
            assert entry['greedy_readout_token_id']==row['generated_token_ids'][k]
            for key in ('prefix_length','prefix_token_ids_sha256','calls_before','recurrent_block_calls'):
                assert entry[key]==ledger[key]
            assert entry['prefix_length']==len(prefix) and entry['calls_before']==4*k
            assert entry['prefix_token_ids_sha256']==hash_tensor(torch.tensor([prefix]),64)
            assert {key:value for key,value in events[k+1].items() if key!='event'}==entry
            window=row['generated_token_ids'][max(0,k-128):k]
            matches=sum(window[i:i+8]==window[-8:] for i in range(len(window)-8)) if len(window)>=8 else 0
            assert entry['prior_exact_8gram_matches']==matches
            flag=entry['phase']=='reasoning' and k>=128 and matches>=2
            assert entry['candidate_repeat_flag']==flag
            if flag:
                flags.append(k)
            depths=entry['depth_states']
            assert len(depths)==4
            for t,state in enumerate(depths):
                assert state['depth']==t and state['position']==len(prefix)-1 and state['prefix_length']==len(prefix)
                assert state['positions_sha256']==hash_tensor(torch.arange(len(prefix))[None],64)
                assert state['dtype']=='torch.bfloat16'
                assert all(math.isfinite(state[key]) and state[key]>=0 for key in ('input_l2','raw_l2','normalized_l2','postnorm_update_l2'))
                if t:
                    assert state['input_sha256']==depths[t-1]['normalized_sha256']
                    assert state['input_l2']==depths[t-1]['normalized_l2']
                assert ('input_vector' in state)==(k in SNAPSHOT_TOKEN_INDICES)
                if k in SNAPSHOT_TOKEN_INDICES:
                    vectors={key:torch.tensor(state[key+'_vector'],dtype=torch.bfloat16) for key in ('input','raw','normalized')}
                    assert all(hash_tensor(v,64)==state[key+'_sha256'] for key,v in vectors.items())
                    assert float(vectors['input'].float().norm())==pytest.approx(state['input_l2'],rel=2e-6,abs=2e-6)
                    assert float(vectors['raw'].float().norm())==pytest.approx(state['raw_l2'],rel=2e-6,abs=2e-6)
                    assert float(vectors['normalized'].float().norm())==pytest.approx(state['normalized_l2'],rel=2e-6,abs=2e-6)
                    delta=float((vectors['normalized'].float()-vectors['input'].float()).norm())
                    assert delta==pytest.approx(state['postnorm_update_l2'],rel=2e-6,abs=2e-6)
        assert len(flags)==summary['candidate_repeat_flags']
        assert (flags[0] if flags else None)==summary['first_candidate_flag_token_index']
        assert row['recurrent_token_positions']==summary['recurrent_token_positions']==sum(4*len(row['prompt_token_ids']+row['generated_token_ids'][:k]) for k in range(n))
        calls+=row['recurrent_block_calls']
    assert report['actual_new_recurrent_calls']==calls


@pytest.mark.real_model
def test_passive_reasoning_recorded_phase_and_raw_text_match_pinned_tokenizer():
    from transformers import AutoTokenizer
    from tebdil.evaluation.monitoring import repeat_features
    from tebdil.evaluation.reasoning import PROMPT_SUFFIX, split_prompt_started_reasoning
    from tebdil.models.ouro_adapter import PINNED_OURO_THINKING_REV
    tok=AutoTokenizer.from_pretrained('ByteDance/Ouro-1.4B-Thinking',revision=PINNED_OURO_THINKING_REV,local_files_only=True)
    folder=ROOT/'reports/passive_reasoning_v1'
    report=json.loads((folder/'report.json').read_text())
    for summary in report['records']:
        case=json.loads((folder/summary['raw_file']).read_text())
        row=case['record']
        assert tok.decode(row['prompt_token_ids'],skip_special_tokens=False)==row['prompt_text_with_special_tokens']
        assert row['prompt_text_with_special_tokens'].endswith(PROMPT_SUFFIX)
        text=tok.decode(row['generated_token_ids'],skip_special_tokens=False)
        assert text==row['decoded_text_with_special_tokens']
        parts=split_prompt_started_reasoning(text,tok.eos_token,completed=row['completed'])
        assert all(row[k]==v for k,v in parts.items())
        assert row['completed']==(row['generated_token_ids'][-1]==tok.eos_token_id)
        assert tok.eos_token_id not in row['generated_token_ids'][:-1]
        for k,entry in enumerate(case['observations']):
            assert all(entry[key]==value for key,value in repeat_features(tok,row['generated_token_ids'][:k]).items())


def test_passive_process_review_does_not_turn_thoughts_into_final_answers():
    from tebdil.tasks.reasoning import evaluate_reasoning_task, solve_planning
    folder=ROOT/'reports/passive_reasoning_v1'
    report=json.loads((folder/'report.json').read_text())
    review=(ROOT/'reports/passive_process_review.md').read_text()
    assert [r['first_candidate_flag_token_index'] for r in report['records']]==[176,207]
    for summary in report['records']:
        case=json.loads((folder/summary['raw_file']).read_text())
        assert case['observations'][summary['first_candidate_flag_token_index']]['prefix_token_ids_sha256'] in review
        assert case['record']['status']=='budget_exhausted' and case['record']['extracted_answer'] is None
    task=json.loads((ROOT/'configs/reasoning_v1_tasks.json').read_text())['partitions']['development'][1]
    raw=json.loads((folder/report['records'][1]['raw_file']).read_text())['record']
    assert '0->3->0->4->5' in raw['reasoning_text']
    assert not any(u==3 and v==0 for u,v,_ in task['instance']['edges'])
    assert solve_planning(task['instance'])==[0,3,1,0,4,5]
    synthetic=dict(completed=True,reasoning_status='closed',final_text='{"path":[0,3,0,4,5]}')
    assert evaluate_reasoning_task(task,synthetic)['status']=='incorrect'
    assert raw['status']=='budget_exhausted'  # A diagnostic on text never rescores the actual unfinished attempt.


def test_reasoning_cot_probe_matches_original_prefixes_physical_work_and_journal():
    folder=ROOT/'reports/reasoning_cot_probe_v1'
    report=json.loads((folder/'report.json').read_text())
    passive=ROOT/'reports/passive_reasoning_v1'
    original=json.loads((passive/'report.json').read_text())
    assert report['protocol']=='reasoning_cot_probe_v1' and report['all_checks_passed']
    assert report['enable_thinking'] and report['recurrent_depth']==4 and report['cache_mode']=='disabled'
    assert report['base_revision'].startswith('09b7f1f')
    assert report['checkpoint_sha256']==original['checkpoint_sha256']
    assert all((ROOT/p).exists() and SHA256.fullmatch(h) for p,h in report['source_sha256'].items())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in report['input_sha256'].items())
    for p in folder.iterdir():
        assert p.read_bytes()==(SITE/'static/evidence/reasoning_cot_probe_v1'/p.name).read_bytes()
    journal=[json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()]
    assert [e['event'] for e in journal]==['start']+['probe']*12+['finish']
    assert journal[-1]['report_sha256']==hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()
    assert [{k:v for k,v in e.items() if k!='event'} for e in journal[1:-1]]==report['probes']
    assert report['windows']==[]  # No invented CoT continuation when every E1 proposal stays unchanged.
    assert report['probe_indices']==[0,127,255,511,1023,2047]
    assert len(report['probes'])==12
    for summary in original['records']:
        case=json.loads((passive/summary['raw_file']).read_text())
        probes=[p for p in report['probes'] if p['problem_id']==summary['problem_id']]
        assert [p['token_index'] for p in probes]==report['probe_indices']
        for p in probes:
            k=p['token_index']
            old=case['observations'][k]
            assert p['prefix_length']==old['prefix_length']
            assert p['prefix_sha256']==old['prefix_token_ids_sha256']
            assert p['positions_sha256']==old['depth_states'][0]['positions_sha256']
            assert p['branch_seed']==20261102+131*k and p['inputs_unchanged']
            assert p['recurrent_calls']==sum(s['recurrent_calls'] for s in p['stages'])==16
            assert [s['recurrent_calls'] for s in p['stages']]==[1,1,1,1,3,2,2,1,1,1,1,1]
            assert p['recurrent_token_positions']==16*p['prefix_length']
            assert [m['method'] for m in p['measurements']]==['S0','M0','M1','E1','R_output']
            baseline=p['measurements'][0]
            assert baseline['next_token_id']==case['record']['generated_token_ids'][k]
            assert baseline['final_last_position_hash']==old['depth_states'][-1]['normalized_sha256']
            assert baseline['logit_l2_vs_S0']==baseline['shift_vs_ordinary']==0
            for m in p['measurements']:
                assert m['next_token_id']==baseline['next_token_id'] and not m['changed_vs_S0']
                assert all(math.isfinite(m[key]) and m[key]>=0 for key in ('logit_l2_vs_S0','shift_vs_ordinary','kl_method_to_S0'))
                assert SHA256.fullmatch(m['post_intervention_hash']) and SHA256.fullmatch(m['final_last_position_hash'])
                if m['method']!='S0':
                    assert m['logit_l2_vs_S0']>0 and m['shift_vs_ordinary']>0
            t=p['intervention']
            assert not t['skipped_unstable'] and t['post_finite'] and t['returned_shift_vs_normal_frob']>0
            assert t['visit_state_hash']!=t['receiving_state_hash']
            assert p['random_relative_norm_error']==pytest.approx(abs(p['random_realized_norm']-p['random_target_norm'])/p['random_target_norm'])
    assert report['actual_new_recurrent_calls']==sum(p['recurrent_calls'] for p in report['probes'])==192
    assert report['recurrent_token_positions']==sum(p['recurrent_token_positions'] for p in report['probes'])==178944
    assert report['reused_baseline_calls']==16384


@pytest.mark.real_model
def test_reasoning_cot_probe_text_is_decoded_from_actual_recorded_ids():
    from transformers import AutoTokenizer
    from tebdil.models.ouro_adapter import PINNED_OURO_THINKING_REV
    tok=AutoTokenizer.from_pretrained('ByteDance/Ouro-1.4B-Thinking',revision=PINNED_OURO_THINKING_REV,local_files_only=True)
    report=json.loads((ROOT/'reports/reasoning_cot_probe_v1/report.json').read_text())
    for p in report['probes']:
        raw=json.loads((ROOT/'reports/passive_reasoning_v1'/(p['problem_id']+'.json')).read_text())['record']
        tokens=raw['generated_token_ids'][:p['token_index']]
        assert p['reasoning_prefix']==tok.decode(tokens,skip_special_tokens=False)
        for m in p['measurements']:
            assert m['next_token_text']==tok.decode([m['next_token_id']],skip_special_tokens=False)
            assert m['text_with_proposal']==tok.decode(tokens+[m['next_token_id']],skip_special_tokens=False)


def test_reasoning_viewer_does_not_conflate_tokens_with_verified_recovery():
    html=INDEX.read_text()
    script=(SITE/'static/app.js').read_text()
    assert 'Reasoning CoT: does relocation change the next token?' in html
    assert 'not live inference or demonstrated rescue' in html
    assert 'Historical arithmetic/graph replays remain below' in html
    assert '#historicalReplay .stage-grid' in script
    assert 'No results substituted' in script
    assert 'no eligible completed final answer' in script

def test_resumed_computation_policy_and_ci_exclude_real_checkpoints():
    policy=(ROOT/'AGENTS.md').read_text()
    workflow=(ROOT/'.github/workflows/pages.yml').read_text()
    walkthrough=(ROOT/'docs/computation_walkthrough.md').read_text()
    assert 'Computation hold lifted by explicit user instruction' in policy
    assert 'UniversalTransformerCache' in policy
    assert 'scripts/run_colab_l4.py' in policy
    assert 'uv run pytest -q -m "not real_model"' in workflow
    assert 'uv run pytest -q -m "real_model"' not in workflow
    assert 'native_generate' in walkthrough and 'KV cache' in walkthrough
    assert 'researchHoldNotice' in INDEX.read_text()


def test_interrupted_budget_journal_is_preserved_without_a_fabricated_completion():
    import struct
    folder=ROOT/'reports/reasoning_budget_interrupted_v1'
    meta=json.loads((folder/'interruption.json').read_text())
    data=(folder/'journal.jsonl').read_bytes()
    events=[json.loads(s) for s in data.splitlines()]
    assert hashlib.sha256(data).hexdigest()==meta['raw_journal_sha256']
    assert meta['status']=='interrupted_by_user' and meta['signal']=='SIGKILL' and meta['exit_code']==137
    assert not meta['final_report_present'] and not meta['terminal_finish_event'] and not meta['total_actual_calls_known']
    assert len(events)==meta['journal_events']==968 and events[-1]['event']=='token'
    assert not any(e['event'] in ('finish','failure') for e in events)
    assert all((ROOT/p).exists() and SHA256.fullmatch(h) for p,h in events[0]['source_sha256'].items())
    tokens=[e for e in events if e['event']=='token']
    assert len(tokens)==meta['observed_new_greedy_proposals']==962
    assert meta['recorded_new_recurrent_calls_lower_bound']==sum(e['recurrent_block_calls'] for e in tokens)+sum(e['recurrent_calls'] for e in events if e['event']=='parity')==3856
    for index,summary in enumerate(meta['per_case']):
        id=summary['problem_id']
        original=json.loads((ROOT/'reports/passive_reasoning_v1'/(id+'.json')).read_text())
        old=original['record']
        rows=[e for e in tokens if e['problem_id']==id]
        assert len(rows)==summary['observed_new_greedy_proposals']==(512 if index==0 else 450)
        prefix=old['prompt_token_ids']+old['generated_token_ids']
        for k,e in enumerate(rows):
            assert e['token_index']==2048+k and e['prefix_length']==len(prefix)
            # Same typed int64 hash, reconstructed from recorded IDs without loading a model.
            typed=f'(1, {len(prefix)}):torch.int64:'.encode()+struct.pack('<'+'q'*len(prefix),*prefix)
            assert e['prefix_token_ids_sha256']==hashlib.sha256(typed).hexdigest()
            assert e['calls_before']==4*k and e['recurrent_block_calls']==4
            prefix.append(e['token_id'])
        finishes=[e['record'] for e in events if e['event']=='case_finish' and e['record']['problem_id']==id]
        assert bool(finishes)==summary['native_generate_returned']==(index==0)
        if finishes:
            assert finishes[0]['generated_token_ids']==prefix[len(old['prompt_token_ids']):]
            assert finishes[0]['status']=='budget_exhausted' and not finishes[0]['completed']
    for p in folder.iterdir():
        assert p.read_bytes()==(SITE/'static/evidence/reasoning_budget_interrupted_v1'/p.name).read_bytes()
