"""Deterministic pure-PyTorch ToyRecurrentAdapter for fast unit tests and CI."""

from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import Tensor, nn

from tebdil.core.state import (
    BranchState,
    TransitionContext,
    hash_tensor,
)
from tebdil.models.interface import validate_state_compatibility


class ToyRecurrentBlock(nn.Module):
    """Shared-weight recurrent transformer-like block with causal self-attention and SwiGLU MLP."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4, step_conditioned: bool = True, max_steps: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.step_conditioned = step_conditioned
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=0.0, batch_first=True, bias=False
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.step_embed = nn.Embedding(max_steps, hidden_size) if step_conditioned else None
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, hidden: Tensor, causal_mask: Tensor, step: int) -> Tensor:
        x = hidden
        if self.step_embed is not None:
            step_idx = min(step, self.step_embed.num_embeddings - 1)
            step_vec = self.step_embed.weight[step_idx].view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            x = x + 0.05 * step_vec
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = hidden + attn_out
        mlp_in = self.norm2(x)
        mlp_out = self.down_proj( torch.nn.functional.silu(self.gate_proj(mlp_in)) * self.up_proj(mlp_in) )
        x = x + mlp_out
        return self.final_norm(x)


class ToyLoopedModel(nn.Module):
    """Self-contained looped language model with shared recurrent core."""

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        total_steps: int = 4,
        max_seq_len: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.total_steps = total_steps
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [ToyRecurrentBlock(hidden_size=hidden_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._init_weights(gen)
        self.eval()

    def _init_weights(self, gen: torch.Generator) -> None:
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.08, generator=gen)

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward_step(self, hidden: Tensor, causal_mask: Tensor, step: int) -> Tensor:
        h = hidden
        for layer in self.layers:
            h = layer(h, causal_mask, step=step)
        return self.norm(h)

    def forward(self, input_ids: Tensor, num_steps: Optional[int] = None) -> Tensor:
        steps = num_steps if num_steps is not None else self.total_steps
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        h = self.embed_tokens(input_ids) + self.pos_embed(pos_ids)
        causal_mask = self.build_causal_mask(seq_len, h.device, h.dtype)
        for t in range(steps):
            h = self.forward_step(h, causal_mask, step=t)
        return self.lm_head(h)


class ToyRecurrentAdapter:
    """RecurrentAdapter wrapping ToyLoopedModel with context and call tracking."""

    def __init__(self, model: Optional[ToyLoopedModel] = None, seed: int = 42):
        self.model = (model if model is not None else ToyLoopedModel(seed=seed)).eval()
        self._recurrent_block_calls = 0
        self._context_cache: Dict[str, tuple[Tensor, Tensor]] = {}

    @property
    def total_steps(self) -> int:
        return self.model.total_steps

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    @property
    def recurrent_block_calls(self) -> int:
        return self._recurrent_block_calls

    def reset_call_counter(self) -> None:
        self._recurrent_block_calls = 0

    @torch.no_grad()
    def initialize(self, input_ids: Tensor) -> BranchState:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        h0 = self.model.embed_tokens(input_ids) + self.model.pos_embed(pos_ids)
        causal_mask = self.model.build_causal_mask(seq_len, h0.device, h0.dtype)

        tok_hash = hash_tensor(input_ids)
        mask_hash = hash_tensor(causal_mask)
        pos_hash = hash_tensor(pos_ids)
        self._context_cache[f"{tok_hash}:{mask_hash}:{pos_hash}"] = (causal_mask, pos_ids)

        ctx = TransitionContext(
            step=0,
            token_ids_hash=tok_hash,
            mask_hash=mask_hash,
            positions_hash=pos_hash,
            cache_mode="disabled",
        )
        return BranchState(
            branch_id="root",
            recurrent_step=0,
            hidden=h0.detach().clone(),
            context=ctx,
            gate_history=(),
            hidden_history=(),
        )

    @torch.no_grad()
    def transition(self, state: BranchState) -> BranchState:
        key = f"{state.context.token_ids_hash}:{state.context.mask_hash}:{state.context.positions_hash}"
        if key not in self._context_cache:
            seq_len = state.hidden.shape[1]
            causal_mask = self.model.build_causal_mask(seq_len, state.hidden.device, state.hidden.dtype)
        else:
            causal_mask, _ = self._context_cache[key]

        h_next = self.model.forward_step(state.hidden, causal_mask, step=state.recurrent_step)
        self._recurrent_block_calls += 1
        next_step = state.recurrent_step + 1
        next_ctx = TransitionContext(
            step=next_step,
            token_ids_hash=state.context.token_ids_hash,
            mask_hash=state.context.mask_hash,
            positions_hash=state.context.positions_hash,
            cache_mode=state.context.cache_mode,
        )
        return BranchState(
            branch_id=state.branch_id,
            recurrent_step=next_step,
            hidden=h_next,
            context=next_ctx,
            gate_history=state.gate_history,
            hidden_history=state.hidden_history + (h_next,),
        )

    @torch.no_grad()
    def readout_logits(self, state: BranchState) -> Tensor:
        return self.model.lm_head(state.hidden)

    def validate_compatibility(self, a: BranchState, b: BranchState) -> None:
        validate_state_compatibility(a, b)
