"""Conservative integer extraction for completed chat responses, never thought text."""

import re


def extract_completed_integer(text: str, eos_text: str, *, completed: bool) -> str | None:
    """Accept a whole integer, a single terminal box, or a terminal Answer/Result label."""
    if not completed:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    text = text.strip()
    if eos_text and text.endswith(eos_text):
        text = text[:-len(eos_text)].strip()
    text = text.replace("**", "").strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return text
    boxes = re.findall(r"\\boxed\{([+-]?\d+)\}", text)
    if boxes:
        terminal = re.search(r"\\boxed\{([+-]?\d+)\}[\s.$]*$", text)
        return terminal.group(1) if len(boxes) == 1 and terminal else None
    label = re.search(r"(?:^|\n)(?:Final\s+)?(?:Answer|Result)\s*:\s*([+-]?\d+)\s*\.?$", text, re.IGNORECASE)
    return label.group(1) if label else None
