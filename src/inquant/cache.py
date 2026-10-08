"""Append-only Hugging Face reference cache for dense Qwen2.5 and Mistral models.

The persistent cache is compressed, but attention still receives fully materialized
floating-point K/V. This adapter validates quality and storage accounting; it is
not a fused attention backend and makes no latency or peak-memory guarantee.
Only batch-one, append-only greedy inference is supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

try:
    from transformers.cache_utils import Cache as _HFCache
except ImportError:
    class _HFCache:  # Codec/cache unit tests can run without transformers installed.
        pass

from .codec import CodecConfig, PackedTensor, quantize


@dataclass(frozen=True)
class CacheConfig:
    codec: CodecConfig = field(default_factory=CodecConfig)
    block_size: int = 256
    residual_length: int = 128
    sink_tokens: int = 4
    # Controlled runners may skip GPU position-value synchronization only when
    # they generate fresh contiguous positions themselves. Shape checks remain.
    validate_positions: bool = True
    shared_workspace: bool = False
    value_bits: int | None = None
    value_group_size: int = 64
    track_peak_bytes: bool = False

    def __post_init__(self) -> None:
        for name in ("block_size", "residual_length", "sink_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.block_size < 1 or self.residual_length < 0 or self.sink_tokens < 0:
            raise ValueError("block_size must be positive; residual_length and sink_tokens nonnegative")
        if not isinstance(self.validate_positions, bool):
            raise ValueError("validate_positions must be a boolean")
        if not isinstance(self.shared_workspace, bool):
            raise ValueError("shared_workspace must be a boolean")
        if not isinstance(self.track_peak_bytes, bool):
            raise ValueError("track_peak_bytes must be a boolean")
        if self.value_bits is not None and (type(self.value_bits) is not int or self.value_bits not in (2, 4)):
            raise ValueError("value_bits must be None, 2 or 4")
        if type(self.value_group_size) is not int or self.value_group_size not in (32, 64, 128):
            raise ValueError("value_group_size must be 32, 64 or 128")


@dataclass
class _Layer:
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    device: torch.device
    seq_length: int = 0
    sink_key: torch.Tensor | None = None
    sink_value: torch.Tensor | None = None
    residual_key: torch.Tensor | None = None
    residual_value: torch.Tensor | None = None
    key_blocks: list[PackedTensor] = field(default_factory=list)
    value_blocks: list[PackedTensor] = field(default_factory=list)
    packed_nbytes: int = 0


@dataclass(frozen=True)
class PackedAttentionView:
    """Read-only old packed blocks plus exact sink/recent/new tokens for attention."""

    key_blocks: tuple[PackedTensor, ...]
    value_blocks: tuple[PackedTensor, ...]
    sink_key: torch.Tensor | None
    sink_value: torch.Tensor | None
    residual_key: torch.Tensor
    residual_value: torch.Tensor
    seq_length: int


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _quantize_temporal_blocks(
    tensor: torch.Tensor, block_count: int, block_size: int, config: CodecConfig,
) -> list[PackedTensor]:
    """Batch independent temporal blocks without pooling their statistics.

    The codec batch axis represents temporal blocks here, not requests. Every
    exported block receives compact owned storage for both payload and metadata;
    slicing the batched output without cloning would retain the whole prefill
    allocation and make per-block storage accounting count it repeatedly.
    """
    if block_count == 0:
        return []
    if block_count == 1:
        return [quantize(tensor[:, :, :block_size], config)]
    _, heads, _, channels = tensor.shape
    blocked = tensor[0, :, :block_count * block_size].reshape(
        heads, block_count, block_size, channels
    ).permute(1, 0, 2, 3)
    packed = quantize(blocked, config)
    shape = (1, heads, block_size, channels)
    return [
        PackedTensor(
            **{
                name: value[index:index + 1].clone(memory_format=torch.contiguous_format)
                for name, value in packed.tensors.items()
            },
            shape=shape,
            original_dtype=tensor.dtype,
        )
        for index in range(block_count)
    ]


class InQuantCache(_HFCache):
    """Immutable quantized history plus exact sink and a bounded recent tail.

    A layer retains at most ``residual_length + block_size - 1`` recent tokens
    because only complete blocks are sealed. A fresh prefill returns its input
    tensors unchanged. Subsequent calls return dequantized history followed by
    the exact new states, then seal new complete blocks for the next call.
    """

    is_compileable = False

    def __init__(self, config: CacheConfig | None = None) -> None:
        super().__init__()
        self.config = config or CacheConfig()
        self._layers: dict[int, _Layer] = {}
        # Optional fused attention owns descriptor/pointer tables here. Their
        # tensor bytes are additional to packed storage and must be reported.
        self._attention_states: dict[int, Any] = {}
        self._shared_decode_workspace = None
        self._tracked_storage = {}
        self._tracked_auxiliary = {}
        self._tracked_live_bytes = 0
        self._tracked_aux_bytes = 0
        self.peak_nbytes = 0
        self._seen_tokens = 0

    def __len__(self) -> int:
        return max(self._layers, default=-1) + 1

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.materialize(layer_idx)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for idx in range(len(self)):
            yield self.materialize(idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer = self._layers.get(0 if layer_idx is None else layer_idx)
        return 0 if layer is None else layer.seq_length

    def get_usable_length(self, new_seq_length: int, layer_idx: int | None = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> None:
        return None

    def get_max_length(self) -> None:
        """Compatibility with older Transformers cache interfaces."""
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + int(cache_position.numel()), 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._validate_update(key_states, value_states, layer_idx, cache_kwargs)
        if layer is None:
            layer = _Layer(tuple(key_states.shape), key_states.dtype, key_states.device)
            full_key, full_value = key_states, value_states
        else:
            old_key, old_value = self.materialize(layer_idx)
            full_key = torch.cat((old_key, key_states), dim=2)
            full_value = torch.cat((old_value, value_states), dim=2)
        self._append(layer, key_states, value_states)
        self._layers[layer_idx] = layer
        self.record_peak(layer_idx)
        if layer_idx == 0:
            self._seen_tokens = layer.seq_length
        return full_key, full_value

    def _validate_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
        layer_idx: int, cache_kwargs: dict[str, Any] | None,
    ) -> _Layer | None:
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer")
        if key_states.ndim != 4 or value_states.shape != key_states.shape:
            raise ValueError("K/V must have matching [batch, KV_heads, tokens, head_dim] shapes")
        if key_states.shape[0] != 1:
            raise NotImplementedError("InQuantCache currently supports batch size 1 only")
        if not key_states.is_floating_point() or not value_states.is_floating_point():
            raise ValueError("K/V must be floating-point tensors")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("K/V must share dtype and device")
        if any(size < 1 for size in key_states.shape):
            raise ValueError("K/V dimensions must be nonempty")
        if key_states.shape[3] % 2:
            raise ValueError("Packed cache requires an even head dimension")
        layer = self._layers.get(layer_idx)
        old_length = 0 if layer is None else layer.seq_length
        if layer is not None:
            if (key_states.shape[0], key_states.shape[1], key_states.shape[3]) != (
                layer.shape[0], layer.shape[1], layer.shape[3]
            ) or key_states.dtype != layer.dtype or key_states.device != layer.device:
                raise ValueError("Layer shape, dtype and device must remain unchanged")
        position = (cache_kwargs or {}).get("cache_position")
        if position is not None:
            if not isinstance(position, torch.Tensor) or position.ndim != 1 or (
                position.numel() != key_states.shape[2] or position.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("cache_position must be a 1D integer tensor with one position per new token")
            if self.config.validate_positions:
                expected = torch.arange(
                    old_length, old_length + key_states.shape[2], device=position.device, dtype=position.dtype
                )
                if not torch.equal(position, expected):
                    raise ValueError("Only contiguous append positions are supported; reset before a new request")

        return layer

    @torch.no_grad()
    def append_for_attention(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
        layer_idx: int, cache_kwargs: dict[str, Any] | None = None,
    ) -> PackedAttentionView:
        """Append one token without materializing history, preserving update semantics.

        The returned snapshot contains old quantized blocks and an exact new
        token. Blocks sealed by this append are used on the *next* decode step,
        just as for :meth:`update`. Only the bounded dense tail is concatenated.
        """
        layer = self._validate_update(key_states, value_states, layer_idx, cache_kwargs)
        if key_states.shape[2] != 1 or layer is None:
            raise ValueError("append_for_attention requires one new token after an existing prefill")
        tail_key, tail_value = key_states, value_states
        if layer.residual_key is not None:
            tail_key = torch.cat((layer.residual_key, tail_key), dim=2)
            tail_value = torch.cat((layer.residual_value, tail_value), dim=2)
        view = PackedAttentionView(
            tuple(layer.key_blocks), tuple(layer.value_blocks),
            layer.sink_key, layer.sink_value, tail_key, tail_value,
            layer.seq_length + 1,
        )
        self._append(layer, key_states, value_states)
        self.record_peak(layer_idx)
        if layer_idx == 0:
            self._seen_tokens = layer.seq_length
        return view

    def _append(self, layer: _Layer, keys: torch.Tensor, values: torch.Tensor) -> None:
        sink_count = 0 if layer.sink_key is None else layer.sink_key.shape[2]
        take_sink = min(self.config.sink_tokens - sink_count, keys.shape[2])
        # Build new storage first. An invalid codec configuration must not leave a
        # partially updated layer or stale logical length behind.
        sink_key, sink_value = layer.sink_key, layer.sink_value
        if take_sink:
            new_key, new_value = keys[:, :, :take_sink], values[:, :, :take_sink]
            if sink_key is not None:
                new_key = torch.cat((sink_key, new_key), dim=2)
                new_value = torch.cat((sink_value, new_value), dim=2)
            sink_key, sink_value = new_key.clone(), new_value.clone()
        pending_key, pending_value = keys[:, :, take_sink:], values[:, :, take_sink:]
        if layer.residual_key is not None:
            pending_key = torch.cat((layer.residual_key, pending_key), dim=2)
            pending_value = torch.cat((layer.residual_value, pending_value), dim=2)
        blocks = max(0, (pending_key.shape[2] - self.config.residual_length) // self.config.block_size)
        new_keys = _quantize_temporal_blocks(pending_key, blocks, self.config.block_size, self.config.codec)
        if self.config.value_bits is None:
            new_values = _quantize_temporal_blocks(pending_value, blocks, self.config.block_size, self.config.codec)
        else:
            from .value_codec import quantize_value_blocks
            new_values = quantize_value_blocks(pending_value, blocks, self.config.block_size,
                                               self.config.value_bits, self.config.value_group_size)
        offset = blocks * self.config.block_size
        # clone(), including the small sink, avoids retaining the original full
        # prefill allocation through a view. No full FP cache is kept here.
        residual_key = pending_key[:, :, offset:].clone()
        residual_value = pending_value[:, :, offset:].clone()
        layer.sink_key, layer.sink_value = sink_key, sink_value
        layer.residual_key, layer.residual_value = residual_key, residual_value
        layer.key_blocks.extend(new_keys)
        layer.value_blocks.extend(new_values)
        if self.config.track_peak_bytes:
            layer.packed_nbytes += sum(block.nbytes for block in new_keys + new_values)
        layer.seq_length += keys.shape[2]

    def record_peak(self, layer_idx):
        """Track live persistent bytes in O(1) per layer update, including tails.

        Packed bytes are accumulated only when a block is sealed. Old default
        configurations retain their historical measurement behavior.
        """
        if not self.config.track_peak_bytes:
            return
        layer = self._layers[layer_idx]
        owned = layer.packed_nbytes + sum(_tensor_bytes(t) for t in (
            layer.sink_key, layer.sink_value, layer.residual_key, layer.residual_value))
        self._tracked_live_bytes += owned - self._tracked_storage.get(layer_idx, 0)
        self._tracked_storage[layer_idx] = owned
        state = self._attention_states.get(layer_idx)
        aux = 0 if state is None else state.nbytes
        self._tracked_aux_bytes += aux - self._tracked_auxiliary.get(layer_idx, 0)
        self._tracked_auxiliary[layer_idx] = aux
        shared = 0 if self._shared_decode_workspace is None else self._shared_decode_workspace.nbytes
        self.peak_nbytes = max(self.peak_nbytes, self._tracked_live_bytes + self._tracked_aux_bytes + shared)

    @torch.no_grad()
    def materialize(self, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct this layer; returned floating-point tensors are not retained."""
        if layer_idx not in self._layers:
            raise KeyError(f"No cached layer {layer_idx}")
        layer = self._layers[layer_idx]

        def restore(sink: torch.Tensor | None, blocks: list[PackedTensor], tail: torch.Tensor | None) -> torch.Tensor:
            pieces = [] if sink is None else [sink]
            pieces.extend(block.dequantize(dtype=layer.dtype) for block in blocks)
            if tail is not None:
                pieces.append(tail)
            return torch.cat(pieces, dim=2)

        return (
            restore(layer.sink_key, layer.key_blocks, layer.residual_key),
            restore(layer.sink_value, layer.value_blocks, layer.residual_value),
        )

    @property
    def payload_nbytes(self) -> int:
        return sum(b.payload_nbytes for layer in self._layers.values() for b in layer.key_blocks + layer.value_blocks)

    @property
    def metadata_nbytes(self) -> int:
        return sum(b.metadata_nbytes for layer in self._layers.values() for b in layer.key_blocks + layer.value_blocks)

    @property
    def residual_nbytes(self) -> int:
        return sum(_tensor_bytes(layer.residual_key) + _tensor_bytes(layer.residual_value) for layer in self._layers.values())

    @property
    def sink_nbytes(self) -> int:
        return sum(_tensor_bytes(layer.sink_key) + _tensor_bytes(layer.sink_value) for layer in self._layers.values())

    @property
    def auxiliary_nbytes(self) -> int:
        shared = 0 if self._shared_decode_workspace is None else self._shared_decode_workspace.nbytes
        return shared + sum(state.nbytes for state in self._attention_states.values())

    @property
    def nbytes(self) -> int:
        """Live persistent tensor bytes, including codec metadata and exact tails.

        Excludes Python objects, model weights, allocator reserves and temporary
        attention/materialization tensors. Measure CUDA peak separately.
        """
        return self.payload_nbytes + self.metadata_nbytes + self.residual_nbytes + self.sink_nbytes + self.auxiliary_nbytes

    @property
    def bf16_nbytes(self) -> int:
        return sum(2 * 2 * layer.shape[0] * layer.shape[1] * layer.seq_length * layer.shape[3] for layer in self._layers.values())

    def memory_stats(self) -> dict[str, int | float]:
        total, baseline = self.nbytes, self.bf16_nbytes
        return {
            "nbytes": total,
            "peak_nbytes": self.peak_nbytes if self.config.track_peak_bytes else None,
            "payload_nbytes": self.payload_nbytes,
            "metadata_nbytes": self.metadata_nbytes,
            "residual_nbytes": self.residual_nbytes,
            "sink_nbytes": self.sink_nbytes,
            "auxiliary_nbytes": self.auxiliary_nbytes,
            "bf16_nbytes": baseline,
            "quantized_layer_tokens": sum(block.shape[2] for layer in self._layers.values() for block in layer.key_blocks),
            "kv_reduction_vs_bf16": 1 - total / baseline if baseline else 0.0,
        }

    def reset(self) -> None:
        self._layers.clear()
        self._attention_states.clear()
        self._shared_decode_workspace = None
        self._tracked_storage.clear()
        self._tracked_auxiliary.clear()
        self._tracked_live_bytes = self._tracked_aux_bytes = self.peak_nbytes = 0
        self._seen_tokens = 0

    def release(self) -> None:
        """Release this cache's references; external materializations remain owned by callers."""
        self.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("Beam search/reordering is unsupported; use num_beams=1")

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("Cache rollback/speculative decoding is unsupported; use append-only generation")

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise NotImplementedError("Batch expansion is unsupported; use batch size 1")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        raise NotImplementedError("Batch selection is unsupported; use batch size 1")
