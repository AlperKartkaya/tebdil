"""Post-hoc pilot format diagnostic. Never rewrites raw completion outcomes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from tebdil.models.ouro_adapter import PINNED_OURO_THINKING_REV
from tebdil.tasks.completion import extract_completed_integer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier evidence")
    source = ROOT / "reports/ouro_baseline_completion.json"
    raw = json.loads(source.read_text())
    if raw["model_revision"] != PINNED_OURO_THINKING_REV:
        raise ValueError("Unsupported tokenizer revision")
    tokenizer = AutoTokenizer.from_pretrained(raw["model_id"], revision=raw["model_revision"], local_files_only=True)
    records = []
    for row in raw["records"]:
        tokens = row["generated_token_ids"]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        if text != row["decoded_text_with_special_tokens"]:
            raise ValueError("Recorded text does not match token IDs")
        completed = bool(tokens) and tokens[-1] == tokenizer.eos_token_id
        if completed != (row["stop_reason"] == "eos_token"):
            raise ValueError("Recorded stop reason does not match token IDs")
        answer = extract_completed_integer(text, tokenizer.eos_token, completed=completed)
        records.append({
            "problem_id": row["problem_id"], "split": row["split"],
            "completed": completed, "final_integer": answer,
            "format_valid": answer is not None,
            "correct": answer is not None and int(answer) == int(row["gold_answer"]),
            "original_strict_correct": row["strict_correct"],
        })
    sources = ["scripts/score_ouro_completion.py", "src/tebdil/tasks/completion.py", "uv.lock"]
    report = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Post-hoc pilot final-answer format diagnostic, not a held-out benchmark",
        "posthoc": True, "answer_extraction": "completed_final_integer_v1",
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
        "input_source": source.relative_to(ROOT).as_posix(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "independent_problems": len(records), "records": records,
        "limitations": [
            "Parser developed after inspecting arithmetic and logic pilot outputs; not preregistered",
            "The same pilot problems, not new independent observations or evidence of generalization",
            "Original strict scores and native outputs remain unchanged; no intervention was evaluated",
            "Future held-out comparisons must freeze this parser and share one prompt, completion and compute protocol",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
