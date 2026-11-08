"""Training-free geometry primitives; not yet connected to inference or caches.

Vectors must already be in one adapter-declared comparable metric space. These
functions diagnose motion and select directions, not correctness or semantic value.
"""
from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GeometryProfile:
    """Provisional dimensionless profile, fixed before any efficacy experiment."""
    efficiency_threshold: float = 0.25
    flattening_threshold: float = 0.10
    min_cosine: float = 0.50
    max_cosine: float = 0.95
    min_score: float = 0.05
    radius_multiplier: float = 0.50
    max_radius: float = 0.10
    epsilon: float = 1e-8

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError('Geometry profile must be finite')
        if not (0 < self.efficiency_threshold < 1 and 0 < self.flattening_threshold < 1
                and -1 <= self.min_cosine < self.max_cosine < 1
                and 0 < self.min_score <= 1 and self.radius_multiplier > 0
                and 0 < self.max_radius < 1 and 0 < self.epsilon < 1):
            raise ValueError('Invalid geometry profile')


@dataclass(frozen=True)
class MotionDiagnostic:
    reason: str
    path_length: float = 0.0
    efficiency: float = 0.0
    relative_movement: float = 0.0

    @property
    def warning(self) -> bool:
        return self.reason in ('cycling', 'flattening')


@dataclass(frozen=True)
class GeometryAnchor:
    request_id: str
    token_position: int
    boundary: int
    vector: Tensor


@dataclass(frozen=True)
class GeometricWaypoint:
    anchor_position: int
    direction: Tensor  # Unit tangent in metric space, NOT an injectable hidden state.
    radius: float
    cosine: float
    novelty: float
    score: float


def _unit_path(path: Tensor, profile: GeometryProfile) -> Tensor | None:
    if path.ndim != 2 or path.shape[1] < 2 or not path.is_floating_point():
        raise ValueError('Expected floating metric vectors [observations, hidden_dim >= 2]')
    if path.shape[0] < 3:
        return None
    values = path.detach().to(dtype=torch.float64 if path.dtype == torch.float64 else torch.float32)
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if not torch.isfinite(values).all() or not torch.isfinite(norms).all() or (norms <= profile.epsilon).any():
        return None
    return values / norms


def diagnose_motion(path: Tensor, reference_scale: float, profile: GeometryProfile = GeometryProfile()) -> MotionDiagnostic:
    """Within-token motion only; reference scale must precede this observation."""
    q = _unit_path(path, profile)
    if q is None or not math.isfinite(reference_scale) or reference_scale <= profile.epsilon:
        return MotionDiagnostic('insufficient_or_invalid_geometry')
    movements = q[1:] - q[:-1]
    lengths = torch.linalg.vector_norm(movements, dim=1)
    length = float(lengths.sum())
    efficiency = float(torch.linalg.vector_norm(q[-1] - q[0])) / max(length, profile.epsilon)
    relative = float(lengths[-1]) / reference_scale
    if length > profile.epsilon and efficiency < profile.efficiency_threshold:
        reason = 'cycling'
    elif relative < profile.flattening_threshold:
        reason = 'flattening'
    else:
        reason = 'progress'
    return MotionDiagnostic(reason, length, efficiency, relative)


def select_waypoint(path: Tensor, anchors: Sequence[GeometryAnchor], *, request_id: str,
                    token_position: int, boundary: int, reference_scale: float,
                    profile: GeometryProfile = GeometryProfile()) -> GeometricWaypoint | None:
    """Rank existing causal anchors. No model calls, RNG, state injection or fallback."""
    q = _unit_path(path, profile)
    if (not isinstance(request_id, str) or not request_id or type(token_position) is not int
            or type(boundary) is not int or token_position < 0 or boundary < 0):
        raise ValueError('Current request, position and boundary required')
    positions = set()
    for anchor in anchors:
        if (type(anchor.token_position) is not int or type(anchor.boundary) is not int
                or anchor.request_id != request_id or anchor.boundary != boundary
                or not 0 <= anchor.token_position < token_position
                or anchor.token_position in positions):
            raise ValueError('Bank must contain unique causal anchors from this request/boundary')
        if anchor.vector.ndim != 1 or anchor.vector.shape[0] != path.shape[-1] or not anchor.vector.is_floating_point():
            raise ValueError('Anchor and path metric dimensions must match')
        positions.add(anchor.token_position)
    if q is None or not math.isfinite(reference_scale) or reference_scale <= profile.epsilon:
        return None
    current = q[-1]
    movements = q[1:] - q[:-1]
    tangents = movements - (movements @ current).unsqueeze(1) * current
    lengths = torch.linalg.vector_norm(tangents, dim=1)
    tangents = tangents[lengths > profile.epsilon]
    lengths = lengths[lengths > profile.epsilon]
    if not len(tangents):
        return None
    directions = tangents / lengths.unsqueeze(1)
    best = None
    # Stable provenance tie-break, independent of input enumeration order.
    for anchor in sorted(anchors, key=lambda a: a.token_position):
        vector = anchor.vector.detach().to(device=q.device, dtype=q.dtype)
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= profile.epsilon:
            continue
        unit = vector / norm
        cosine = float(torch.dot(current, unit).clamp(-1, 1))
        if not profile.min_cosine <= cosine <= profile.max_cosine:
            continue
        tangent = unit - cosine * current
        size = torch.linalg.vector_norm(tangent)
        if size <= profile.epsilon:
            continue
        direction = tangent / size
        redundancy = float((directions @ direction).square().max().clamp(0, 1))
        novelty = 1 - redundancy
        score = (1 + cosine) * 0.5 * novelty
        if score < profile.min_score or (best is not None and score <= best.score):
            continue
        best = GeometricWaypoint(anchor.token_position, direction,
                                 min(profile.max_radius, profile.radius_multiplier * reference_scale),
                                 cosine, novelty, score)
    return best


@dataclass
class GeometryGate:
    """One gate per request/boundary; ambiguity must be computed causally by caller."""
    persistence: int = 3
    cooldown_tokens: int = 18
    consecutive: int = 0
    last_position: int = -1
    blocked_through: int = -1
    closed: bool = False

    def __post_init__(self):
        if type(self.persistence) is not int or self.persistence < 1 or type(self.cooldown_tokens) is not int or self.cooldown_tokens < 0:
            raise ValueError('Positive persistence and nonnegative integer cooldown required')

    def observe(self, token_position: int, motion: MotionDiagnostic, *, ambiguous: bool) -> bool:
        if type(token_position) is not int or token_position <= self.last_position or type(ambiguous) is not bool:
            raise ValueError('Observations must be strictly causal with Boolean ambiguity')
        if token_position != self.last_position + 1:
            self.consecutive = 0
        self.last_position = token_position
        if self.closed or token_position <= self.blocked_through or not ambiguous or not motion.warning:
            self.consecutive = 0
            return False
        self.consecutive = min(self.persistence, self.consecutive + 1)
        return self.consecutive == self.persistence

    def attempted(self) -> None:
        """Call for either committed or rolled-back excursions, never on selection alone."""
        if self.closed or self.consecutive < self.persistence:
            raise ValueError('An attempt requires a persistent eligible observation')
        self.blocked_through = self.last_position + self.cooldown_tokens
        self.consecutive = 0

    def close(self) -> None:
        self.closed = True
        self.consecutive = 0
