"""Passive same-prefix depth diagnostics; exact repeats are not semantic stalls."""
from contextlib import contextmanager

import torch

from tebdil.core.state import hash_tensor

SNAPSHOT_TOKEN_INDICES = (0, 127, 255, 511, 1023, 2047)


def repeat_features(tokenizer, generated_prefix):
    """No task, answer, verifier feedback, next token or future context is accepted."""
    text = tokenizer.decode(generated_prefix, skip_special_tokens=False)
    phase = ('malformed' if '<think>' in text or text.count('</think>') > 1 else
             'final' if '</think>' in text else 'reasoning')
    window = generated_prefix[-128:]
    matches = (sum(window[i:i+8] == window[-8:] for i in range(len(window)-8))
               if len(window) >= 8 else 0)
    return dict(phase=phase, prior_exact_8gram_matches=matches,
                candidate_repeat_flag=phase == 'reasoning' and len(generated_prefix) >= 128 and matches >= 2)


@contextmanager
def passive_observer(adapter, tokenizer, on_entry):
    """Yield a native on_forward callback, recording real last-position tensors."""
    if adapter.total_steps != 4 or adapter.readout_mode != 'last_step':
        raise ValueError('Four-depth last-step contract required')
    generated, depths, pending = [], [], []

    def capture_input(module, args, kwargs):
        depth = kwargs['current_ut']
        hidden = args[0]
        positions = kwargs['position_ids']
        if (depth != len(depths) or pending or hidden.ndim != 3 or hidden.shape[0] != 1
                or not torch.equal(positions, torch.arange(hidden.shape[1], device=positions.device)[None])):
            raise ValueError('Unaligned native depth/position context')
        vector = hidden[0,-1].detach().cpu().clone()
        if depths and hash_tensor(vector,64) != depths[-1]['normalized_sha256']:
            raise ValueError('Recurrent input does not match previous normalized state')
        pending.append((vector, hidden.shape[1], hash_tensor(positions,64)))

    def capture_output(module, args, output):
        if len(pending) != 1:
            raise ValueError('Missing recurrent input boundary')
        before, length, positions_hash = pending.pop()
        raw, after = args[0][0,-1].detach().cpu().clone(), output[0,-1].detach().cpu().clone()
        if any(not torch.isfinite(v).all() for v in (before,raw,after)):
            raise ValueError('Nonfinite observed state')
        inp, pre, out = before.float(), raw.float(), after.float()
        input_norm, output_norm = float(inp.norm()), float(out.norm())
        delta = float((out-inp).norm())
        row = dict(depth=len(depths), position=length-1, prefix_length=length,
                   positions_sha256=positions_hash, dtype=str(before.dtype),
                   input_sha256=hash_tensor(before,64), raw_sha256=hash_tensor(raw,64),
                   normalized_sha256=hash_tensor(after,64),
                   input_l2=input_norm, raw_l2=float(pre.norm()), normalized_l2=output_norm,
                   postnorm_update_l2=delta,
                   relative_postnorm_update_l2=delta/input_norm if input_norm else None,
                   cosine_input_normalized=float(torch.dot(inp,out)/(input_norm*output_norm)) if input_norm*output_norm else None)
        if len(generated) in SNAPSHOT_TOKEN_INDICES:
            row.update(input_vector=inp.tolist(), raw_vector=pre.tolist(), normalized_vector=out.tolist())
        depths.append(row)

    def observe(entry):
        if len(depths) != 4 or pending or entry['token_index'] != len(generated):
            raise ValueError('Passive depth/forward ledger mismatch')
        if any(d['prefix_length'] != entry['prefix_length'] for d in depths):
            raise ValueError('Depth states belong to a different prefix')
        on_entry({**entry, 'generated_prefix_token_count':len(generated),
                  **repeat_features(tokenizer,generated), 'depth_states':list(depths)})
        generated.append(entry['greedy_readout_token_id'])
        depths.clear()

    first = adapter.model.model.layers[0].register_forward_pre_hook(capture_input, with_kwargs=True)
    norm = adapter.model.model.norm.register_forward_hook(capture_output)
    try:
        yield observe
        if pending or depths:
            raise ValueError('Unfinished passive observation boundary')
    finally:
        norm.remove()
        first.remove()
