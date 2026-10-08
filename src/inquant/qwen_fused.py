"""Instance-local Qwen2.5 / Mistral single-token decode over physically packed KV.

Validated against the Transformers 4.53 Qwen2 and Mistral attention interfaces. Prefill and
unsupported attention-mask modes use the original forward. CUDA decoding calls
the optional Triton packed kernel; CPU reference decoding is explicitly opt-in
for correctness tests and is never an accelerated backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from .cache import InQuantCache, PackedAttentionView


def _reference_decode(query: torch.Tensor, view: PackedAttentionView, scale: float) -> torch.Tensor:
    """Test oracle; deliberately materializes all old blocks."""
    def restore(sink, blocks, tail):
        pieces = [] if sink is None else [sink]
        pieces.extend(block.dequantize(dtype=query.dtype) for block in blocks)
        pieces.append(tail)
        return torch.cat(pieces, dim=2)
    key = restore(view.sink_key, view.key_blocks, view.residual_key)
    value = restore(view.sink_value, view.value_blocks, view.residual_value)
    return F.scaled_dot_product_attention(query, key, value, scale=scale, enable_gqa=True)


@dataclass
class _KernelState:
    kernel: Any

    @property
    def nbytes(self) -> int:
        return self.kernel.nbytes


@dataclass
class FusedQwen2Handle:
    """Owns only this model's forward overrides; restore removes them."""
    originals: list[tuple[Any, Any]]
    restored: bool = False

    @property
    def fused_decode_calls(self) -> int:
        return sum(getattr(module, "_inquant_fused_decode_calls", 0) for module, _ in self.originals)

    @property
    def cpu_reference_calls(self) -> int:
        return sum(getattr(module, "_inquant_cpu_reference_calls", 0) for module, _ in self.originals)

    def restore(self) -> None:
        if self.restored:
            return
        for module, original in self.originals:
            module.forward = original
            delattr(module, "_inquant_original_forward")
        self.restored = True

    def __enter__(self) -> "FusedQwen2Handle":
        return self

    def __exit__(self, *_args) -> None:
        self.restore()


def enable_fused_attention(model: Any, *, allow_cpu_reference: bool = False) -> FusedQwen2Handle:
    """Patch this instantiated model only; retain its weights and prefill forward.

    The accelerated path requires CUDA, head_dim=128, batch=1, q_len=1, full
    attention, no padding mask and an InQuantCache. Other query lengths and
    cache types retain the original forward. Arbitrary masks/output_attentions
    also retain the original forward and therefore may materialize KV.
    """
    model_type = getattr(model.config, "model_type", None)
    if model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention as Attention, apply_rotary_pos_emb
    elif model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralAttention as Attention, apply_rotary_pos_emb
        if getattr(model.config, "sliding_window", None) is not None:
            raise ValueError("Mistral sliding-window attention requires a window-aware cache and is not supported")
    else:
        raise ValueError("Expected a dense Qwen2 or Mistral model")
    if model.training:
        raise ValueError("Call model.eval() before enabling inference-only packed decode")
    modules = [layer.self_attn for layer in model.model.layers]
    for module in modules:
        if not isinstance(module, Attention):
            raise TypeError("Expected an unmodified attention instance for this model family")
        if "position_embeddings" not in inspect.signature(module.forward).parameters:
            raise RuntimeError("Unsupported Transformers attention interface; validated versions are 4.53.x")
        if getattr(module, "sliding_window", None) is not None:
            raise ValueError("Sliding-window attention is unsupported by packed decode")
        if hasattr(module, "_inquant_original_forward"):
            raise ValueError("This model already has fused InQuant decode enabled")
        if not allow_cpu_reference and module.head_dim != 128:
            raise ValueError("The CUDA packed decode kernel currently requires head_dim=128")

    def forward(
        attention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        use_packed = (
            isinstance(past_key_value, InQuantCache)
            and hidden_states.shape[1] == 1
            and past_key_value.get_seq_length(attention.layer_idx) > 0
            and attention_mask is None
            and not kwargs.get("output_attentions", False)
        )
        if not use_packed:
            return attention._inquant_original_forward(
                hidden_states=hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_value=past_key_value,
                cache_position=cache_position, **kwargs,
            )
        if attention.training or torch.is_grad_enabled():
            raise RuntimeError("Packed decode is inference-only; use model.eval() and torch.no_grad()")
        if hidden_states.shape[0] != 1:
            raise ValueError("Packed decode supports batch size 1 only")
        if hidden_states.device.type != "cuda" and not allow_cpu_reference:
            raise RuntimeError("Packed decode requires CUDA; CPU reference is only available by explicit opt-in")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        view = past_key_value.append_for_attention(
            key, value, attention.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
        if hidden_states.device.type == "cuda":
            from .triton_attention import PackedDecodeState, SharedDecodeWorkspace
            if past_key_value.config.shared_workspace and past_key_value._shared_decode_workspace is None:
                past_key_value._shared_decode_workspace = SharedDecodeWorkspace()
            state = past_key_value._attention_states.get(attention.layer_idx)
            if state is None:
                state = _KernelState(PackedDecodeState(
                    view.key_blocks, view.value_blocks,
                    workspace_pool=past_key_value._shared_decode_workspace))
                past_key_value._attention_states[attention.layer_idx] = state
            elif len(state.kernel.key_blocks) != len(view.key_blocks):
                state.kernel.extend(view.key_blocks, view.value_blocks)
            result = state.kernel.decode(
                query, sink_key=view.sink_key, sink_value=view.sink_value,
                residual_key=view.residual_key, residual_value=view.residual_value,
                scale=attention.scaling,
            )
            past_key_value.record_peak(attention.layer_idx)
            attention._inquant_fused_decode_calls += 1
        else:
            result = _reference_decode(query, view, attention.scaling)
            attention._inquant_cpu_reference_calls += 1
        output = result.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attention.o_proj(output), None

    originals = []
    for module in modules:
        originals.append((module, module.forward))
        module._inquant_original_forward = module.forward
        module._inquant_fused_decode_calls = 0
        module._inquant_cpu_reference_calls = 0
        module.forward = MethodType(forward, module)
    return FusedQwen2Handle(originals)


def enable_fused_qwen2(model: Any, *, allow_cpu_reference: bool = False) -> FusedQwen2Handle:
    """Backward-compatible Qwen-only entry point."""
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("Expected a dense Qwen2/Qwen2.5 model")
    return enable_fused_attention(model, allow_cpu_reference=allow_cpu_reference)


def enable_fused_mistral(model: Any, *, allow_cpu_reference: bool = False) -> FusedQwen2Handle:
    """Mistral full-attention entry point; preserves native projections and RoPE."""
    if getattr(model.config, "model_type", None) != "mistral":
        raise ValueError("Expected a dense Mistral model")
    return enable_fused_attention(model, allow_cpu_reference=allow_cpu_reference)
