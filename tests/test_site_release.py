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
