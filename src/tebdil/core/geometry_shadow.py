"""Causal, generation-only geometric observer. Never executes an intervention."""
from collections import deque
from dataclasses import asdict
import math
import torch
from torch import Tensor
from tebdil.core.geometry import (
    GeometryAnchor, GeometryGate, GeometryProfile, diagnose_motion, select_waypoint,
)


class GeometryShadow:
    # Frozen mechanics profile, not tuned on answers; decode-only bank avoids prefill hooks.
    boundary = 3
    bank_capacity = 64
    reference_capacity = 32
    warmup = 8
    ambiguity_quantile = .75

    def __init__(self):
        self.profile = GeometryProfile()
        self.gate = GeometryGate()
        self.anchors = deque(maxlen=self.bank_capacity)
        self.movements = deque(maxlen=self.reference_capacity)
        self.entropies = deque(maxlen=self.reference_capacity)
        self.records = []
        self.extra_readouts = 0

    def observe(self, path: Tensor, logits: Tensor, token_position: int) -> None:
        """Three post-normalization boundaries of ONE token, before boundary four."""
        if path.ndim != 2 or path.shape[0] != 3:
            raise ValueError('Shadow requires three comparable post-boundary vectors')
        if logits.ndim != 1 or logits.numel() < 2:
            raise ValueError('Shadow requires one vocabulary readout')
        if type(token_position) is not int or token_position <= self.gate.last_position:
            raise ValueError('Shadow observations must be strictly causal')
        path = path.detach().float()
        norms = path.norm(dim=-1, keepdim=True)
        valid = bool(torch.isfinite(path).all() and torch.isfinite(norms).all()
                     and (norms > self.profile.epsilon).all() and torch.isfinite(logits).all())
        ready = len(self.movements) >= self.warmup
        scale = float(torch.tensor(list(self.movements)).median()) if ready else 0.
        threshold = float(torch.quantile(torch.tensor(list(self.entropies)), self.ambiguity_quantile)) if ready else None
        entropy = None
        if valid:
            logp = logits.detach().float().log_softmax(-1)
            entropy = float(-(logp.exp() * logp).sum() / math.log(logits.numel()))
            valid = math.isfinite(entropy)
            if not valid:
                entropy = None
        motion = diagnose_motion(path, scale if valid else 0., self.profile)
        ambiguous = bool(valid and ready and entropy > self.profile.epsilon and entropy >= threshold)
        eligible = self.gate.observe(token_position, motion, ambiguous=ambiguous)
        waypoint = select_waypoint(path, self.anchors, request_id='generation', token_position=token_position,
                                   boundary=self.boundary, reference_scale=scale, profile=self.profile) if eligible else None
        self.records.append({
            'token_position': token_position, 'boundary': self.boundary,
            'reference_count': len(self.movements), 'reference_scale': scale,
            'normalized_entropy': entropy, 'ambiguity_threshold': threshold,
            'ambiguous': ambiguous, 'motion': asdict(motion), 'eligible': eligible,
            'closed': self.gate.closed, 'bank_positions': [a.token_position for a in self.anchors],
            'selected': None if waypoint is None else {
                'anchor_position': waypoint.anchor_position, 'radius': waypoint.radius,
                'cosine': waypoint.cosine, 'novelty': waypoint.novelty, 'score': waypoint.score,
            },
            'attempted': False,
        })
        # Update references only AFTER the decision; current/future states cannot select themselves.
        if valid and not self.gate.closed:
            q = path / norms
            self.movements.append(float((q[-1] - q[-2]).norm()))
            self.entropies.append(entropy)
            self.anchors.append(GeometryAnchor('generation', token_position, self.boundary, path[-1].clone()))

    def report(self) -> dict:
        return {'mode': 'shadow_only', 'metric': 'adapter_declared', 'boundary': self.boundary,
                'bank_source': 'prior_decode_positions_only', 'bank_capacity': self.bank_capacity,
                'reference_capacity': self.reference_capacity, 'warmup': self.warmup,
                'ambiguity_quantile': self.ambiguity_quantile, 'profile': asdict(self.profile),
                'persistence': self.gate.persistence, 'cooldown_tokens': self.gate.cooldown_tokens,
                'extra_readout_projections': self.extra_readouts, 'records': self.records}
