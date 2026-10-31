"""Per-token coordinate diagnostics, not semantic geometry or production operators."""

import torch
from torch import Tensor


def _checked_pair(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    if a.shape != b.shape or a.ndim != 3 or a.device != b.device:
        raise ValueError('Expected matching [batch, tokens, hidden] shape and device')
    a, b = a.detach().float(), b.detach().float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Coordinate diagnostics require finite tensors')
    return a, b


def radial_decomposition(anchor: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    """Split delta into Euclidean radial/orthogonal components, independently per token."""
    anchor, delta = _checked_pair(anchor, delta)
    squared_norm = anchor.square().sum(-1, keepdim=True)
    if not torch.isfinite(squared_norm).all() or (squared_norm <= 0).any():
        raise ValueError('Radial decomposition requires nonzero finite anchor norms')
    radial = (delta * anchor).sum(-1, keepdim=True) / squared_norm * anchor
    tangent = delta - radial
    if not torch.isfinite(radial).all() or not torch.isfinite(tangent).all():
        raise ValueError('Radial decomposition produced nonfinite components')
    return radial, tangent


def match_token_norm(candidate: Tensor, reference: Tensor, weight: Tensor | None = None) -> Tensor:
    """Return FP32 candidate directions rescaled to reference's per-token norms.

    Optional coordinate divisor is the native learned RMSNorm weight (not a new
    normalization layer). Callers cast explicitly and measure realized norm error.
    """
    candidate, reference = _checked_pair(candidate, reference)
    if weight is not None:
        weight = weight.detach().to(device=candidate.device, dtype=torch.float32)
        if weight.shape != (candidate.shape[-1],) or not torch.isfinite(weight).all() or (weight.abs() <= 1e-8).any():
            raise ValueError('Metric weight must be a finite nonzero hidden-size vector')
        candidate_metric, reference_metric = candidate / weight, reference / weight
    else:
        candidate_metric, reference_metric = candidate, reference
    norm = candidate_metric.norm(dim=-1, keepdim=True)
    target = reference_metric.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norm).all() or not torch.isfinite(target).all():
        raise ValueError('Metric norms must be finite')
    if ((norm == 0) & (target != 0)).any():
        raise ValueError('Cannot preserve direction of a zero candidate with nonzero target')
    result = candidate * (target / norm.clamp_min(torch.finfo(torch.float32).tiny))
    if not torch.isfinite(result).all():
        raise ValueError('Norm matching produced nonfinite coordinates')
    return result
