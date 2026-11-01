"""Additive True-mode recording; historical completion contracts stay unchanged."""

from tebdil.evaluation.completion import score_completion, validate_completion
from tebdil.tasks.completion_v2 import extract_completed_integer_v2

PROMPT_SUFFIX = '<|im_start|>assistant\n<think>\n'


def _prompt_text(tokenizer, prompt_ids):
    text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    if not text.endswith(PROMPT_SUFFIX):
        raise ValueError('Expected a prompt-started Ouro reasoning section')
    return text


def encode_reasoning_prompt(tokenizer, prompt):
    """Encode only the question; the opening reasoning tag belongs to the input."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Nonempty question required')
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=True, return_tensors='pt',
    )
    _prompt_text(tokenizer, ids.tolist()[0])
    return ids


def split_prompt_started_reasoning(text, eos_text, *, completed):
    """Split only actually generated text; never invent a generated opening tag."""
    if not eos_text or type(completed) is not bool:
        raise ValueError('EOS text and a Boolean completion flag required')
    result = {'reasoning_status': 'malformed', 'reasoning_text': None, 'final_text': None}
    if text.count(eos_text) != int(completed) or (completed and not text.endswith(eos_text)):
        return result
    body = text[:-len(eos_text)] if completed else text
    if '<think>' in body or body.count('</think>') > 1:
        return result
    if '</think>' not in body:
        return {**result, 'reasoning_status': 'missing_close' if completed else 'open', 'reasoning_text': body}
    thought, final = body.split('</think>', 1)
    return {'reasoning_status': 'closed', 'reasoning_text': thought, 'final_text': final}


def validate_reasoning_completion(tokenizer, prompt_ids, tokens, ledger, calls, *, budget, depth):
    """Reuse physical-ledger validation, but derive boundaries from True-mode input."""
    prompt_text = _prompt_text(tokenizer, prompt_ids)
    record = validate_completion(tokenizer, prompt_ids, tokens, ledger, calls, budget=budget, depth=depth)
    for key in ('answer_extraction', 'format_valid', 'extracted_answer'):
        del record[key]
    return {
        **record, 'enable_thinking': True, 'reasoning_open_in_prompt': True,
        'prompt_text_with_special_tokens': prompt_text,
        **split_prompt_started_reasoning(record['decoded_text_with_special_tokens'], tokenizer.eos_token,
                                       completed=record['completed']),
    }


def evaluate_reasoning_integer(record, gold_answer):
    """Offline integer smoke only; future task families require their own verifier."""
    answer = None
    if record['completed'] and record['reasoning_status'] == 'closed':
        answer = extract_completed_integer_v2(record['final_text'], '', completed=True)
    return {
        'answer_extraction': 'prompt_started_reasoning_integer_v1',
        'extracted_answer': answer, 'format_valid': answer is not None,
        **score_completion({'completed': record['completed'], 'extracted_answer': answer}, gold_answer),
    }
