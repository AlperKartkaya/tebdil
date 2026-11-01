"""Versioned format repair; the frozen v1 parser and recorded scores stay intact."""

import re

from tebdil.evaluation.completion import score_completion
from tebdil.tasks.completion import extract_completed_integer

INTEGER = r"[+-]?[0-9]+"
BOX = rf"\\boxed\{{\s*{INTEGER}\s*\}}"
ATOM = rf"(?:{BOX}|{INTEGER})"
DELIMITERS = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))


def _balanced_math(text):
    closing = dict(DELIMITERS)
    pending = []
    for token in re.findall(r"\\[()\[\]]|(?<!\\)\$\$|(?<!\\)\$", text):
        if pending and token == pending[-1]:
            pending.pop()
        elif token in closing and not pending:
            pending.append(closing[token])
        else:
            return False
    return not pending


def extract_completed_integer_v2(text: str, eos_text: str, *, completed: bool) -> str | None:
    """Recognize one explicit final integer, never unfinished or ambiguous text."""
    if not completed:
        return None
    text = text.strip()
    if eos_text and eos_text in text:
        if text.count(eos_text) != 1 or not text.endswith(eos_text):
            return None
        text = text[:-len(eos_text)].strip()
    if "<think>" in text or "</think>" in text:
        if (not text.startswith("<think>") or text.count("<think>") != 1
                or text.count("</think>") != 1):
            return None
        text = text.split("</think>", 1)[1].strip()
    text = text.replace("**", "").strip()
    if not _balanced_math(text):
        return None
    for opening, closing in DELIMITERS:
        text = re.sub(re.escape(opening) + rf"\s*({ATOM})\s*" + re.escape(closing), r"\1", text)
    boxes = re.findall(BOX, text)
    if len(boxes) > 1 or text.count(r"\boxed") != len(boxes):
        return None
    text = re.sub(rf"\\boxed\{{\s*({INTEGER})\s*\}}", lambda m: r"\boxed{" + m[1] + "}", text)
    labels = list(re.finditer(r"(?:^|\n)[ \t]*(?:Final\s+)?(?:Answer|Result)\s*:\s*", text, re.IGNORECASE))
    if len(labels) > 1:
        return None
    if labels:
        candidate = text[labels[0].end():].strip()
        if not re.fullmatch(rf"{ATOM}\s*\.?", candidate):
            return None
        text = candidate
    if re.fullmatch(rf"{INTEGER}\s*\.", text):
        text = text[:-1].strip()
    answer = extract_completed_integer(text, "", completed=True)
    return answer if answer is not None and re.fullmatch(INTEGER, answer) else None


def evaluate_completion_v2(record, eos_text, gold_answer):
    """Return separate offline fields; never edit a historical record in place."""
    answer = extract_completed_integer_v2(
        record["decoded_text_with_special_tokens"], eos_text, completed=record["completed"],
    )
    return {
        "answer_extraction": "completed_final_integer_v2", "extracted_answer": answer,
        "format_valid": answer is not None,
        **score_completion({"completed": record["completed"], "extracted_answer": answer}, gold_answer),
    }
