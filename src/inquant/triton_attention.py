"""Single-query GQA attention directly over the reference packed InQuant layout.

No full-sized floating-point K/V tensor is created. Independent partitions
compute online-softmax partials and a second kernel reduces those partials.
Keys already contain their original RoPE positions. Since this path accepts
only the newest query, all supplied keys are visible and their storage order
does not affect attention.

``PackedDecodeState`` owns pointer tables, channel decoding maps, and a reusable
FP32 partial-softmax workspace. ``nbytes`` includes all of these persistent GPU
allocations, but excludes input PackedTensors, exact tails, queries and outputs.
Append immutable blocks with ``extend`` instead of rebuilding old descriptors. Pass exact sink/tail
tensors on every decode; they can include the current, not-yet-sealed token.
The state is not safe for simultaneous calls from different CUDA streams.

Limitations: CUDA, batch=1, head_dim=128, one query token, uniform packed block
length, query_heads divisible by KV_heads, and no additive mask/dropout/ALiBi.
This module is optional: importing the core codec never imports Triton.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .codec import PackedTensor
from .value_codec import PackedValues


@triton.jit
def _decode_tile(PAYLOAD, SCALES, PADDING, MAP, token, channel,
                 valid_token, kv_head, H: tl.constexpr, T: tl.constexpr,
                 D: tl.constexpr, M: tl.constexpr):
    role = tl.load(MAP + kv_head * D + channel).to(tl.int32)
    packed_offset = kv_head * T * (D // 2) + token[:, None] * (D // 2)
    byte = tl.load(PAYLOAD + packed_offset + channel[None, :] // 2,
                   mask=valid_token[:, None], other=0).to(tl.int32)
    slot = (byte >> ((channel[None, :] % 2) * 4)) & 15
    partner = tl.maximum(role, 0)
    partner_byte = tl.load(PAYLOAD + packed_offset + partner[None, :] // 2,
                           mask=valid_token[:, None] & (role[None, :] >= 0), other=0).to(tl.int32)
    partner_slot = (partner_byte >> ((partner[None, :] % 2) * 4)) & 15
    integer = tl.where(role[None, :] >= 0, ((slot << 4) | partner_slot) - 128, slot - 8)
    scale = tl.load(SCALES + kv_head * D + channel)
    padding_index = tl.maximum(-role - 2, 0)
    padding = tl.load(PADDING + kv_head * M + padding_index, mask=role <= -2, other=0)
    return tl.where(role[None, :] <= -2, padding[None, :], integer.to(tl.float32) * scale[None, :])


@triton.jit
def _decode_value_tile(PAYLOAD, SCALES, OFFSETS, token, channel, valid_token, kv_head,
                       T: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr):
    per_byte: tl.constexpr = 8 // BITS
    packed_offset = (kv_head * T + token[:, None]) * (D // per_byte)
    byte = tl.load(PAYLOAD + packed_offset + channel[None, :] // per_byte,
                   mask=valid_token[:, None], other=0).to(tl.int32)
    code = (byte >> ((channel[None, :] % per_byte) * BITS)) & ((1 << BITS) - 1)
    parameter = (kv_head * T + token[:, None]) * (D // GROUP) + channel[None, :] // GROUP
    scale = tl.load(SCALES + parameter, mask=valid_token[:, None], other=0).to(tl.float32)
    offset = tl.load(OFFSETS + parameter, mask=valid_token[:, None], other=0).to(tl.float32)
    return code.to(tl.float32) * scale + offset


@triton.jit
def _attention_partials(
    Q, TABLES, K_MAP, V_MAP, SINK_K, SINK_V, TAIL_K, TAIL_V, PARTIAL,
    Q_STRIDE_H: tl.constexpr,
    SK_STRIDE_H, SK_STRIDE_T,
    SV_STRIDE_H, SV_STRIDE_T,
    TK_STRIDE_H, TK_STRIDE_T,
    TV_STRIDE_H, TV_STRIDE_T,
    SINK_T, TAIL_T, SCALE,
    N_BLOCKS: tl.constexpr, H_KV: tl.constexpr, GROUPS: tl.constexpr,
    PACKED_T: tl.constexpr, SALIENT: tl.constexpr,
    PARTITIONS: tl.constexpr, D: tl.constexpr,
    TILE_T: tl.constexpr, DENSE_SPLIT: tl.constexpr, QUERY_TILE: tl.constexpr,
    VALUE_BITS: tl.constexpr, VALUE_GROUP: tl.constexpr,
):
    kv_head = tl.program_id(0)
    partition = tl.program_id(1)
    channel = tl.arange(0, D)
    query_index = tl.arange(0, QUERY_TILE)
    head = kv_head * GROUPS + query_index
    q = tl.load(Q + head[:, None] * Q_STRIDE_H + channel[None, :],
                mask=query_index[:, None] < GROUPS, other=0)
    m = tl.full((QUERY_TILE,), float("-inf"), tl.float32)
    normalizer = tl.full((QUERY_TILE,), 0.0, tl.float32)
    accumulator = tl.full((QUERY_TILE, D), 0.0, tl.float32)

    if partition < N_BLOCKS:
        key_payload = tl.load(TABLES + partition * 6).to(tl.pointer_type(tl.uint8))
        key_scales = tl.load(TABLES + partition * 6 + 1).to(tl.pointer_type(tl.float32))
        key_padding = tl.load(TABLES + partition * 6 + 2).to(tl.pointer_type(tl.float32))
        val_payload = tl.load(TABLES + partition * 6 + 3).to(tl.pointer_type(tl.uint8))
        if VALUE_BITS:
            val_scales = tl.load(TABLES + partition * 6 + 4).to(tl.pointer_type(tl.float16))
            val_padding = tl.load(TABLES + partition * 6 + 5).to(tl.pointer_type(tl.float16))
        else:
            val_scales = tl.load(TABLES + partition * 6 + 4).to(tl.pointer_type(tl.float32))
            val_padding = tl.load(TABLES + partition * 6 + 5).to(tl.pointer_type(tl.float32))
        key_map = K_MAP + partition * H_KV * D
        val_map = V_MAP + partition * H_KV * D
        for base in range(triton.cdiv(PACKED_T, TILE_T)):
            token = base * TILE_T + tl.arange(0, TILE_T)
            valid = token < PACKED_T
            keys = _decode_tile(key_payload, key_scales, key_padding, key_map,
                                token, channel, valid, kv_head, H_KV, PACKED_T, D, SALIENT)
            # Reference SDPA receives a dequantized tensor in the model dtype;
            # reproduce that rounding before FP32 dot-product accumulation.
            keys = keys.to(Q.dtype.element_ty)
            scores = tl.dot(q, tl.trans(keys), input_precision="tf32x3") * SCALE
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - next_m)
            probabilities = tl.exp(scores - next_m[:, None])
            if VALUE_BITS:
                values = _decode_value_tile(val_payload, val_scales, val_padding,
                                            token, channel, valid, kv_head, PACKED_T, D, VALUE_BITS, VALUE_GROUP)
            else:
                values = _decode_tile(val_payload, val_scales, val_padding, val_map,
                                      token, channel, valid, kv_head, H_KV, PACKED_T, D, SALIENT)
            values = values.to(Q.dtype.element_ty)
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(Q.dtype.element_ty), values, input_precision="tf32x3")
            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
            m = next_m
    else:
        dense_partition = partition - N_BLOCKS
        sink_partitions = tl.cdiv(SINK_T, DENSE_SPLIT)
        if dense_partition < sink_partitions:
            dense_k, dense_v = SINK_K, SINK_V
            stride_kh, stride_kt = SK_STRIDE_H, SK_STRIDE_T
            stride_vh, stride_vt = SV_STRIDE_H, SV_STRIDE_T
            start = dense_partition * DENSE_SPLIT
            length = SINK_T
        else:
            dense_k, dense_v = TAIL_K, TAIL_V
            stride_kh, stride_kt = TK_STRIDE_H, TK_STRIDE_T
            stride_vh, stride_vt = TV_STRIDE_H, TV_STRIDE_T
            start = (dense_partition - sink_partitions) * DENSE_SPLIT
            length = TAIL_T
        for base in range(triton.cdiv(DENSE_SPLIT, TILE_T)):
            token = start + base * TILE_T + tl.arange(0, TILE_T)
            valid = token < length
            keys = tl.load(dense_k + kv_head * stride_kh + token[:, None] * stride_kt + channel[None, :],
                           mask=valid[:, None], other=0)
            values = tl.load(dense_v + kv_head * stride_vh + token[:, None] * stride_vt + channel[None, :],
                             mask=valid[:, None], other=0)
            scores = tl.dot(q, tl.trans(keys), input_precision="tf32x3") * SCALE
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - next_m)
            probabilities = tl.exp(scores - next_m[:, None])
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(Q.dtype.element_ty), values, input_precision="tf32x3")
            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
            m = next_m

    result = PARTIAL + (head * PARTITIONS + partition) * (D + 2)
    tl.store(result[:, None] + channel[None, :], accumulator, mask=query_index[:, None] < GROUPS)
    tl.store(result + D, m, mask=query_index < GROUPS)
    tl.store(result + D + 1, normalizer, mask=query_index < GROUPS)


@triton.jit
def _reduce_partials(PARTIAL, OUTPUT, PARTITIONS: tl.constexpr, D: tl.constexpr,
                     BLOCK_PARTITIONS: tl.constexpr, OUTPUT_TILE: tl.constexpr):
    head = tl.program_id(0)
    channel = tl.program_id(1) * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
    partition = tl.arange(0, BLOCK_PARTITIONS)
    pointer = PARTIAL + (head * PARTITIONS + partition) * (D + 2)
    m = tl.load(pointer + D, mask=partition < PARTITIONS, other=float("-inf"))
    normalizer = tl.load(pointer + D + 1, mask=partition < PARTITIONS, other=0.0)
    weights = tl.exp(m - tl.max(m, axis=0))
    values = tl.load(pointer[:, None] + channel[None, :],
                     mask=(partition[:, None] < PARTITIONS) & (channel[None, :] < D), other=0.0)
    result = tl.sum(values * weights[:, None], axis=0) / tl.sum(normalizer * weights, axis=0)
    tl.store(OUTPUT + head * D + channel, result, mask=channel < D)


class SharedDecodeWorkspace:
    """One scratch allocation per request, usable only on one CUDA stream.

    Layers consume the scratch in stream order and never retain its views.
    The cache owns/counts this allocation once; layer states exclude it.
    """

    def __init__(self):
        self.tensor = None
        self._stream_identity = None

    @property
    def nbytes(self):
        return 0 if self.tensor is None else self.tensor.untyped_storage().nbytes()

    def acquire(self, shape, device):
        stream = torch.cuda.current_stream(device)
        identity = (stream.device, stream.cuda_stream)
        if self._stream_identity is not None and identity != self._stream_identity:
            raise RuntimeError("Shared decode workspace requires one CUDA stream per cache")
        self._stream_identity = identity
        required = math.prod(shape)
        if self.tensor is None or self.tensor.numel() < required:
            self.tensor = torch.empty(required, dtype=torch.float32, device=device)
        return self.tensor[:required].view(shape)


class PackedDecodeState:
    """Reusable descriptors for immutable blocks; exact tails are per-call inputs."""

    def __init__(self, key_blocks: list[PackedTensor], value_blocks: list[PackedTensor], *, workspace_pool=None):
        if len(key_blocks) != len(value_blocks):
            raise ValueError("Key and value block counts must match")
        self.key_blocks = ()
        self.value_blocks = ()
        self._workspace: torch.Tensor | None = None
        self._workspace_pool = workspace_pool
        self._tables: torch.Tensor | None = None
        self._key_map: torch.Tensor | None = None
        self._value_map: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._kv_heads = None
        self._packed_tokens = 256
        self._salient = 0
        self._value_bits = 0
        self._value_group = 64
        self._capacity = 0
        self._stream_identity = None
        self.extend(key_blocks, value_blocks)

    def extend(self, key_blocks, value_blocks):
        """Append immutable blocks, building descriptors only for the new suffix.

        Spare capacity is bounded to at most seven blocks once the sequence
        grows beyond eight blocks. nbytes counts all reserved table capacity.
        """
        if len(key_blocks) != len(value_blocks):
            raise ValueError("Key and value block counts must match")
        old_count = len(self.key_blocks)
        if len(key_blocks) < old_count or any(a is not b for a, b in zip(self.key_blocks, key_blocks)) or any(
                a is not b for a, b in zip(self.value_blocks, value_blocks)):
            raise ValueError("PackedDecodeState.extend requires an immutable append-only prefix")
        if len(key_blocks) == old_count:
            return
        first = key_blocks[0]
        if first.shape[0] != 1 or first.shape[-1] != 128 or first.payload.device.type != "cuda":
            raise ValueError("Packed decode requires CUDA blocks with batch=1 and head_dim=128")
        stream = torch.cuda.current_stream(first.payload.device)
        identity = (stream.device, stream.cuda_stream)
        if self._stream_identity is not None and self._stream_identity != identity:
            raise RuntimeError("Descriptor updates require one CUDA stream per state")
        self._stream_identity = identity
        self._device = first.payload.device
        self._kv_heads = first.shape[1]
        self._packed_tokens = first.shape[2]
        self._salient = first.salient_channels.shape[-1]
        for block in key_blocks:
            if not isinstance(block, PackedTensor) or block.shape != first.shape or block.salient_channels.shape[-1] != self._salient:
                raise ValueError("All packed K/V blocks must share shape and salient-channel count")
        self._value_bits = value_blocks[0].bits if isinstance(value_blocks[0], PackedValues) else 0
        self._value_group = value_blocks[0].group_size if self._value_bits else 64
        for block in value_blocks:
            if block.shape != first.shape:
                raise ValueError("All packed blocks must share shape")
            if self._value_bits:
                if not isinstance(block, PackedValues) or (block.bits, block.group_size) != (self._value_bits, self._value_group):
                    raise ValueError("All V blocks must share the same quantizer")
            elif not isinstance(block, PackedTensor) or block.salient_channels.shape[-1] != self._salient:
                raise ValueError("All InQuant V blocks must share the salient-channel count")
        for block in (*key_blocks, *value_blocks):
            if any(t.device != self._device for t in block.tensors.values()):
                raise ValueError("All packed tensors must share the same CUDA device")
        new_keys, new_values = key_blocks[old_count:], value_blocks[old_count:]
        pointers = [[k.payload.data_ptr(), k.scales.data_ptr(), k.padding.data_ptr(),
                     v.payload.data_ptr(), v.scales.data_ptr(),
                     (v.offsets if self._value_bits else v.padding).data_ptr()]
                    for k, v in zip(new_keys, new_values)]
        count = len(key_blocks)
        if count > self._capacity:
            capacity = count if not self._capacity else max(count, min(self._capacity * 2, count + 7))
            tables = torch.empty((capacity, 6), dtype=torch.int64, device=self._device)
            key_map = torch.empty((capacity, self._kv_heads, 128), dtype=torch.int8, device=self._device)
            value_map = (torch.empty((0,), dtype=torch.int8, device=self._device)
                         if self._value_bits else torch.empty_like(key_map))
            if old_count:
                tables[:old_count].copy_(self._tables[:old_count])
                key_map[:old_count].copy_(self._key_map[:old_count])
                if not self._value_bits:
                    value_map[:old_count].copy_(self._value_map[:old_count])
            self._tables, self._key_map, self._value_map = tables, key_map, value_map
            self._capacity = capacity
        self._tables[old_count:count].copy_(torch.tensor(pointers, dtype=torch.int64, device=self._device))
        self._write_maps(self._key_map[old_count:count], new_keys)
        if not self._value_bits:
            self._write_maps(self._value_map[old_count:count], new_values)
        self.key_blocks, self.value_blocks = tuple(key_blocks), tuple(value_blocks)

    def _write_maps(self, result, blocks):
        # Roles: donor channels 0..127, normal=-1, padding indices -2..-65.
        result.fill_(-1)
        padding_index = -torch.arange(self._salient, dtype=torch.int8, device=self._device) - 2
        sources = torch.cat([block.salient_channels for block in blocks], dim=0).long()
        donors = torch.cat([block.donor_channels for block in blocks], dim=0).long()
        result.scatter_(-1, sources, donors.to(torch.int8))
        result.scatter_(-1, donors, padding_index.expand_as(donors))
        return result

    @property
    def nbytes(self) -> int:
        """Auxiliary live GPU allocations; input packed blocks are counted elsewhere."""
        return sum(t.untyped_storage().nbytes() for t in
                   (self._tables, self._key_map, self._value_map, self._workspace) if t is not None)

    @torch.no_grad()
    def decode(self, query: torch.Tensor, *, sink_key: torch.Tensor | None = None,
               sink_value: torch.Tensor | None = None, residual_key: torch.Tensor | None = None,
               residual_value: torch.Tensor | None = None, scale: float | None = None) -> torch.Tensor:
        if query.ndim != 4 or query.shape[0] != 1 or query.shape[1] < 1 or query.shape[-2:] != (1, 128):
            raise ValueError("query must have shape [1, query_heads, 1, 128]")
        if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("query must be a CUDA FP16, BF16 or FP32 tensor")
        if query.stride(-1) != 1:
            raise ValueError("query head dimension must be contiguous")
        if self._device is not None and query.device != self._device:
            raise ValueError("query and packed blocks must share a device")
        kv_heads = self._kv_heads
        dense = []
        for name, key, value in (("sink", sink_key, sink_value), ("residual", residual_key, residual_value)):
            if (key is None) != (value is None):
                raise ValueError(f"{name} key/value must both be provided")
            if key is None:
                dense.append((query, query, 0))
                continue
            if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1 or key.shape[-1] != 128:
                raise ValueError(f"{name} K/V must have matching [1, KV_heads, tokens, 128] shapes")
            if key.dtype != query.dtype or value.dtype != query.dtype or key.device != query.device or value.device != query.device:
                raise ValueError(f"{name} K/V must match query dtype and device")
            if key.stride(-1) != 1 or value.stride(-1) != 1:
                raise ValueError(f"{name} head dimension must be contiguous")
            kv_heads = key.shape[1] if kv_heads is None else kv_heads
            if key.shape[1] != kv_heads:
                raise ValueError("All inputs must share their KV head count")
            dense.append((key, value, key.shape[2]))
        if kv_heads is None or query.shape[1] % kv_heads:
            raise ValueError("query_heads must be divisible by the nonempty KV head count")
        (sink_key, sink_value, sink_t), (residual_key, residual_value, residual_t) = dense
        partitions = len(self.key_blocks) + triton.cdiv(sink_t, 256) + triton.cdiv(residual_t, 256)
        if not partitions:
            raise ValueError("Attention requires at least one key/value token")
        if scale is None:
            scale = 128 ** -0.5
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attention scale must be positive and finite")
        workspace_shape = (query.shape[1], partitions, 130)
        if self._workspace_pool is not None:
            workspace = self._workspace_pool.acquire(workspace_shape, query.device)
        else:
            if self._workspace is None or self._workspace.shape != workspace_shape or self._workspace.device != query.device:
                self._workspace = torch.empty(workspace_shape, dtype=torch.float32, device=query.device)
            workspace = self._workspace
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        # Triton type-checks both sides of the runtime partition branch. Empty
        # placeholders must therefore have integer table/map dtypes even though
        # a dense-only invocation never dereferences them. Zero elements allocate
        # no GPU storage and are still included in nbytes.
        if self._tables is None:
            self._tables = torch.empty((0, 6), dtype=torch.int64, device=query.device)
            self._key_map = torch.empty((0, kv_heads, 128), dtype=torch.int8, device=query.device)
            self._value_map = torch.empty_like(self._key_map)
            self._device = query.device
        _attention_partials[(kv_heads, partitions)](
            query, self._tables, self._key_map, self._value_map, sink_key, sink_value, residual_key, residual_value,
            workspace, query.stride(1), sink_key.stride(1), sink_key.stride(2),
            sink_value.stride(1), sink_value.stride(2), residual_key.stride(1), residual_key.stride(2),
            residual_value.stride(1), residual_value.stride(2), sink_t, residual_t, scale,
            len(self.key_blocks), kv_heads, query.shape[1] // kv_heads, self._packed_tokens,
            self._salient, partitions, 128, 32, 256,
            max(16, triton.next_power_of_2(query.shape[1] // kv_heads)),
            self._value_bits, self._value_group, num_warps=4, num_stages=1,
        )
        _reduce_partials[(query.shape[1], 4)](
            workspace, output, partitions, 128, triton.next_power_of_2(partitions), 32,
            num_warps=4,
        )
        return output


def packed_decode(query: torch.Tensor, key_blocks: list[PackedTensor], value_blocks: list[PackedTensor],
                  sink_key: torch.Tensor | None = None, sink_value: torch.Tensor | None = None,
                  residual_key: torch.Tensor | None = None, residual_value: torch.Tensor | None = None,
                  *, scale: float | None = None) -> torch.Tensor:
    """One-shot convenience wrapper; reuse PackedDecodeState for repeated decoding."""
    return PackedDecodeState(key_blocks, value_blocks).decode(
        query, sink_key=sink_key, sink_value=sink_value,
        residual_key=residual_key, residual_value=residual_value, scale=scale,
    )
