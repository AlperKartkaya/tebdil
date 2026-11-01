"""Post-hoc format audit of six existing development outputs; no model execution."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from tebdil.evaluation.completion import score_completion
from tebdil.tasks.completion import extract_completed_integer
from tebdil.tasks.completion_v2 import evaluate_completion_v2

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("reports/ouro_completed_answer_smoke.json", "reports/ouro_completed_answer_comparison.json")
SOURCES = (
    "scripts/verify_completion_v2.py", "src/tebdil/tasks/completion_v2.py",
    "src/tebdil/tasks/completion.py", "src/tebdil/evaluation/completion.py",
    "tests/test_completion_v2.py", "docs/completed_answer_v2_protocol.md",
)


def identities(paths):
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve prior evidence")
    inputs, sources = identities(INPUTS), identities(SOURCES)
    eos = json.loads((ROOT / INPUTS[0]).read_text())["eos_token"]
    rows = []
    for source in INPUTS:
        report = json.loads((ROOT / source).read_text())
        if not report["all_checks_passed"]:
            raise ValueError("Verified source report required")
        for index, record in enumerate(report["records"]):
            if record["problem_id"] != "completion_dev_000" or record["split"] != "development":
                raise ValueError("Only already-observed development smoke outputs allowed")
            original = {k: record[k] for k in ("answer_extraction", "format_valid", "extracted_answer", "status", "is_correct")}
            answer = extract_completed_integer(record["decoded_text_with_special_tokens"], eos, completed=record["completed"])
            if (original["answer_extraction"] != "completed_final_integer_v1"
                    or answer != original["extracted_answer"] or original["format_valid"] != (answer is not None)
                    or score_completion(record, record["gold_answer"]) != {k: original[k] for k in ("status", "is_correct")}):
                raise ValueError("Original v1 scores cannot be reproduced")
            rows.append({
                "source": source, "record_index": index, "problem_id": record["problem_id"],
                "method": record["method"], "split": record["split"],
                "decoded_text_sha256": hashlib.sha256(record["decoded_text_with_special_tokens"].encode()).hexdigest(),
                "original_v1": original, "v2_diagnostic": evaluate_completion_v2(record, eos, record["gold_answer"]),
            })
    if inputs != identities(INPUTS) or sources != identities(SOURCES):
        raise ValueError("Source/input identities changed during audit")
    result = {
        "schema_version": 1, "protocol": "completed_answer_measurement_v2",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources, "input_sha256": inputs, "posthoc": True,
        "new_model_executions": 0, "independent_problems": 1, "records": rows,
        "all_checks_passed": True,
        "limitations": [
            "Known development outputs; parser amendment motivated by an observed format failure",
            "Original model outputs and v1 scores remain unchanged; v2 fields are a separate diagnostic",
            "Recognition is not new reasoning success, rescue, held-out accuracy or universal format coverage",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Audited {len(rows)} existing outputs; no new model executions; original artifacts unchanged")


if __name__ == "__main__":
    main()
