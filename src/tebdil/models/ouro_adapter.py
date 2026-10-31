"""OuroRecurrentAdapter: Instrumented stepwise adapter for OuroForCausalLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional
import torch
from torch import Tensor

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from tebdil.core.state import BranchState, TransitionContext, hash_tensor
from tebdil.models.interface import validate_state_compatibility

# Ensure vendored Ouro package is importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vendor.ouro_1_4b_thinking.configuration_ouro import OuroConfig
from vendor.ouro_1_4b_thinking.modeling_ouro import OuroForCausalLM


PINNED_OURO_THINKING_REV = "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"
PINNED_OURO_BASE_REV = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


@dataclass
class _OuroStepContextData:
    causal_mask_mapping: Dict[str, Optional[Tensor]]
    position_ids: Tensor
    cache_position: Tensor
    position_embeddings: tuple[Tensor, Tensor]


class OuroRecurrentAdapter:
    """Full-prefix, cache-free recurrent execution of the vendored Ouro model.

    Enforces `use_cache=False` during recurrent state transitions so that
    branching and excursions never alias or corrupt KV caches.
    `last_step` forces the latest state; `native_gate` honors the model's
    configured forced step or threshold. Neither readout saves native loop work.
    """

    def __init__(
        self,
        model: OuroForCausalLM,
        readout_mode: str = "last_step",
        model_id: str = "ByteDance/Ouro-1.4B-Thinking",
        model_revision: str = PINNED_OURO_THINKING_REV,
    ):
        if readout_mode not in ("last_step", "native_gate"):
            raise ValueError(f"Unsupported readout_mode: {readout_mode}")
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.readout_mode = readout_mode
        self.model_id = model_id
        self.model_revision = model_revision
        self._recurrent_block_calls = 0
        self._step_context_store: Dict[str, _OuroStepContextData] = {}

    @classmethod
    def from_mini_config(
        cls,
        vocab_size: int = 256,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        total_ut_steps: int = 4,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int = 42,
        readout_mode: str = "last_step",
    ) -> "OuroRecurrentAdapter":
        """Create an OuroRecurrentAdapter backed by a lightweight in-memory OuroForCausalLM."""
        torch.manual_seed(seed)
        config = OuroConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=hidden_size // num_attention_heads,
            total_ut_steps=total_ut_steps,
            early_exit_threshold=1.0,
            use_cache=False,
            _attn_implementation="eager",
        )
        model = OuroForCausalLM(config).to(device=device, dtype=dtype).eval()
        return cls(
            model=model,
            readout_mode=readout_mode,
            model_id="ouro-mini-verification",
            model_revision="local-vendored-cb8c980b",
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "ByteDance/Ouro-1.4B-Thinking",
        revision: str = PINNED_OURO_THINKING_REV,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        readout_mode: str = "last_step",
        total_ut_steps: Optional[int] = None,
        local_files_only: bool = False,
    ) -> "OuroRecurrentAdapter":
        """Load pinned Ouro checkpoint into vendored OuroForCausalLM."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        config = OuroConfig.from_pretrained(
            repo_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        config.use_cache = False
        config._attn_implementation = "eager"
        if total_ut_steps is not None:
            config.total_ut_steps = total_ut_steps
        model = OuroForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            config=config,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        model = model.to(device=device).eval()
        return cls(
            model=model,
            readout_mode=readout_mode,
            model_id=repo_id,
            model_revision=revision,
        )

    @property
    def total_steps(self) -> int:
        return int(getattr(self.model.model, "total_ut_steps", self.model.config.total_ut_steps))

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def recurrent_block_calls(self) -> int:
        return self._recurrent_block_calls

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def reset_call_counter(self) -> None:
        self._recurrent_block_calls = 0

    def estimate_step_flops(self, seq_len: int, batch_size: int = 1) -> int:
        """Estimate FLOPs for a single recurrent step across all shared layers.

        Standard transformer forward FLOP approximation per layer:
        2 * batch * seq_len * (4 * d^2 + 3 * d * d_ff) + 4 * batch * seq_len^2 * d
        """
        d = self.hidden_size
        d_ff = int(self.model.config.intermediate_size)
        L = self.num_layers
        linear_flops = 2 * batch_size * seq_len * (4 * d * d + 3 * d * d_ff)
        attn_flops = 4 * batch_size * (seq_len ** 2) * d
        return L * (linear_flops + attn_flops)

    @torch.no_grad()
    def initialize(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> BranchState:
        """Prepare initial embedding state H_0 and precompute causal masks & RoPE."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, tokens] tensor")
        input_ids = input_ids.to(device=self.device)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids [batch, tokens]")
            attention_mask = attention_mask.to(device=self.device)

        ouro_base = self.model.model
        inputs_embeds = ouro_base.embed_tokens(input_ids)
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        else:
            if (
                position_ids.ndim != 2
                or position_ids.shape[1] != input_ids.shape[1]
                or position_ids.shape[0] not in (1, input_ids.shape[0])
                or position_ids.dtype != torch.long
                or bool((position_ids < 0).any())
            ):
                raise ValueError("position_ids must be nonnegative int64 [1 or batch, tokens]")
            position_ids = position_ids.to(device=self.device).detach().clone()

        mask_kwargs = {
            "config": ouro_base.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping: Dict[str, Optional[Tensor]] = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if ouro_base.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

        position_embeddings = ouro_base.rotary_emb(inputs_embeds, position_ids)

        tok_hash = hash_tensor(input_ids)
        mask_tensor = causal_mask_mapping["full_attention"]
        mask_hash = hash_tensor(mask_tensor) if mask_tensor is not None else hash_tensor(attention_mask)
        pos_hash = hash_tensor(position_ids)
        ctx_key = f"{tok_hash}:{mask_hash}:{pos_hash}"
        self._step_context_store[ctx_key] = _OuroStepContextData(
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

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
            hidden=inputs_embeds.detach().clone(),
            context=ctx,
            gate_history=(),
            hidden_history=(inputs_embeds.detach().clone(),),
        )

    @torch.no_grad()
    def transition(self, state: BranchState) -> BranchState:
        """Execute one shared-stack step, preserving native boundary RMSNorm and gate."""
        if state.context.cache_mode != "disabled":
            raise ValueError("Ouro excursions require cache_mode='disabled'")
        if state.context.step != state.recurrent_step or state.recurrent_step < 0:
            raise ValueError("TransitionContext step must match the nonnegative recurrent step")
        ctx_key = f"{state.context.token_ids_hash}:{state.context.mask_hash}:{state.context.positions_hash}"
        if ctx_key not in self._step_context_store:
            raise KeyError(
                f"Missing precomputed Ouro context for key {ctx_key}. Call initialize() first."
            )
        step_data = self._step_context_store[ctx_key]
        ouro_base = self.model.model
        current_ut = state.recurrent_step

        hidden_states = state.hidden.detach().clone()
        for decoder_layer in ouro_base.layers[: ouro_base.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=step_data.causal_mask_mapping[decoder_layer.attention_type],
                position_ids=step_data.position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=step_data.cache_position,
                position_embeddings=step_data.position_embeddings,
                current_ut=current_ut,
            )

        hidden_states = ouro_base.norm(hidden_states)
        gate_out = ouro_base.early_exit_gate(hidden_states)
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
            hidden=hidden_states,
            context=next_ctx,
            gate_history=state.gate_history + (gate_out,),
            hidden_history=state.hidden_history + (hidden_states,),
        )

    @torch.no_grad()
    def compute_gate(self, hidden_states: Tensor) -> Tensor:
        """Compute Ouro early_exit_gate on an arbitrary hidden state tensor."""
        return self.model.model.early_exit_gate(hidden_states)

    @torch.no_grad()
    def readout_logits(self, state: BranchState) -> Tensor:
        """Project state to vocabulary logits according to configured readout_mode."""
        if self.readout_mode == "last_step" or not state.gate_history:
            return self.model.lm_head(state.hidden)

        # Native gate readout matching OuroForCausalLM.forward (align post-step hidden states with gate_history)
        gate_list = list(state.gate_history)
        hidden_states_list = list(state.hidden_history[-len(gate_list) :])
        forced_step = self.model.early_exit_step
        if forced_step is not None and 0 <= forced_step < len(hidden_states_list):
            return self.model.lm_head(hidden_states_list[forced_step])
        exit_threshold = self.model.early_exit_threshold
        if exit_threshold is None or len(gate_list) != len(hidden_states_list):
            return self.model.lm_head(state.hidden)

        pdf_list = []
        remaining_prob = torch.ones_like(gate_list[0].squeeze(-1))
        for idx, gate_tensor in enumerate(gate_list):
            lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))
            if idx < len(gate_list) - 1:
                p_i = lambda_i * remaining_prob
                remaining_prob = remaining_prob * (1.0 - lambda_i)
            else:
                p_i = remaining_prob
            pdf_list.append(p_i)
        stacked_exit_pdf = torch.stack(pdf_list, dim=2)
        cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
        threshold_mask = cumulative_probs >= exit_threshold
        exit_steps = torch.argmax(threshold_mask.float(), dim=2)
        last_step_idx = stacked_exit_pdf.shape[2] - 1
        if last_step_idx >= 0:
            never_exceeded = ~threshold_mask.any(dim=2)
            exit_steps[never_exceeded] = last_step_idx
        stacked_hidden = torch.stack(hidden_states_list, dim=2)
        gather_index = (
            exit_steps.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, stacked_hidden.size(-1))
        )
        final_hidden_states = torch.gather(stacked_hidden, 2, gather_index).squeeze(2)
        return self.model.lm_head(final_hidden_states)

    def validate_compatibility(self, a: BranchState, b: BranchState) -> None:
        validate_state_compatibility(a, b)
