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
VIEWS = {"problem", "mechanism", "decision", "experiment", "evidence"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")

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
        self.views: list[str] = []
        self.attributes: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        element_id = values.get("id")
        if element_id:
            self.ids.append(element_id)
            self.attributes[element_id] = values
        if "data-view" in values:
            self.views.append(values["data-view"])
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


def test_journey_has_five_accessible_linked_panels() -> None:
    parser = parse_site()
    assert len(parser.ids) == len(set(parser.ids)), "HTML IDs must be unique"
    assert set(parser.views) == VIEWS

    for view in VIEWS:
        panel_id = f"view-{view}"
        assert panel_id in parser.attributes
        nav = [attrs for attrs in parser.attributes.values() if attrs.get("data-view") == view]
        assert len(nav) == 1
        assert nav[0].get("aria-controls") == panel_id
        assert parser.attributes[panel_id].get("aria-labelledby") == nav[0].get("id")

    selected = [
        attrs for attrs in parser.attributes.values() if attrs.get("aria-selected") == "true"
    ]
    assert len(selected) == 1


def test_original_light_academic_identity_and_shared_diagram_colors() -> None:
    html = INDEX.read_text(encoding="utf-8")
    css = (SITE / "static/style.css").read_text(encoding="utf-8")
    script = (SITE / "static/app.js").read_text(encoding="utf-8")
    assert 'class="hero-banner"' in html
    assert 'class="pub-links"' in html
    assert "TEBDİL: Self-Directed Latent Excursions" in html
    assert "color-scheme:light" in css
    for token, color in {
        "bg": "#ffffff", "text": "#0f172a", "muted": "#475569",
        "a": "#4338ca", "b": "#92400e", "visit": "#7c3aed",
    }.items():
        assert f"--{token}:{color}" in css
    for token in ("a", "b", "visit", "muted"):
        assert f"var(--{token})" in script, "SVGs must follow the light interface palette"


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
        "baselineOutput",
        "methodOutput",
        "rawTrace",
        "downloadTrace",
    }
    assert required <= set(parser.ids)
    assert parser.attributes["loadStatus"].get("role") == "status"


def test_javascript_does_not_target_missing_elements() -> None:
    parser = parse_site()
    source = (SITE / "static/app.js").read_text(encoding="utf-8")
    references = set(
        re.findall(r'(?:\$\(|getElementById\()\s*["\']([^"\']+)["\']', source)
    )
    assert references <= set(parser.ids), f"missing JS targets: {sorted(references - set(parser.ids))}"


def test_mechanism_copy_does_not_describe_controls_as_excursion_returns() -> None:
    script = r"""
const fs = require("fs");
const source = fs.readFileSync("site/static/app.js", "utf8");
const start = source.indexOf("  function mechanismCopy(method)");
const end = source.indexOf("\n  function renderRecording()", start);
if (start < 0 || end < 0) throw new Error("mechanismCopy must remain independently testable");
eval(source.slice(start, end).trim());
for (const method of ["M0", "M1", "C0", "R0"]) {
  const copy = mechanismCopy(method);
  if (copy.residualMeasured) throw new Error(`${method} must not report a measured excursion residual`);
  if (copy.stages.some((stage) => /returns? to A/i.test(stage))) {
    throw new Error(`${method} must not claim an excursion return`);
  }
}
for (const method of ["E1", "E1_no_curv", "E1_rand_curv"]) {
  const copy = mechanismCopy(method);
  if (!copy.residualMeasured) throw new Error(`${method} must preserve visited-state measurement`);
  if (!copy.stages.some((stage) => /returns? to A/i.test(stage))) {
    throw new Error(`${method} must explain the return to the receiving trajectory`);
  }
}
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


def test_initial_static_payload_stays_small() -> None:
    parser = parse_site()
    blocking = {INDEX, *(SITE / urlsplit(url).path for url in parser.local_assets)}
    total = sum(path.stat().st_size for path in blocking)
    assert total <= 150 * 1024, f"initial payload is {total / 1024:.1f} KiB"


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
