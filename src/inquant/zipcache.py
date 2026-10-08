"""Qwen/HF4.53 port of the official ZipCache packed reference implementation.

Upstream: ThisisBillhe/ZipCache, 8833f675a938b019fccc531bfdf932abe6e622ad.
The unmodified MIT-licensed quantization functions are loaded from third_party.
This is NOT a fused ZipCache kernel: decode materializes old KV as upstream does.
The port changes model plumbing, keeps indices on-device, chunks saliency probes,
and uses a per-request CPU RNG so resume/repeat order cannot change predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from transformers.cache_utils import Cache

UPSTREAM_REVISION = "8833f675a938b019fccc531bfdf932abe6e622ad"
CODEC_SHA256 = "a35c5b0aa6acaea4350e6ef9af13d817c27ff7a22be81658812154762c1595a7"


@lru_cache(maxsize=1)
def official_codec():
    path = Path(__file__).resolve().parents[2] / "third_party/ZipCache/zipcache/models/CompressUtils/compress_function.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CODEC_SHA256:
        raise RuntimeError("ZipCache official codec source differs from the pinned revision")
    spec = importlib.util.spec_from_file_location("_inquant_zipcache_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.source_sha256 = digest
    return module


@dataclass(frozen=True)
class ZipCacheConfig:
    important_bits: int = 4
    unimportant_bits: int = 2
    unimportant_ratio: float = .4
    streaming_gap: int = 100
    probe_chunk_size: int = 32

    def __post_init__(self):
        if self.important_bits not in (2, 4, 8) or self.unimportant_bits not in (2, 4, 8):
            raise ValueError("Official ZipCache packing supports 2/4/8 bits")
        if self.important_bits < self.unimportant_bits or not 0 < self.unimportant_ratio < 1:
            raise ValueError("Invalid ZipCache precision allocation")
        if self.streaming_gap < 1 or self.probe_chunk_size < 1:
            raise ValueError("ZipCache gap and probe chunk size must be positive")
        if int(self.streaming_gap * self.unimportant_ratio) < 1:
            raise ValueError("Streaming block must contain at least one unimportant token")


def important_complement(ids: torch.Tensor, length: int):
    """Equivalent to upstream's CPU mask loop, without CPU/GPU index mixing."""
    mask = torch.ones((*ids.shape[:2], length), dtype=torch.bool, device=ids.device)
    mask.scatter_(-1, ids, False)
    all_ids = torch.arange(length, device=ids.device).expand_as(mask)
    return all_ids[mask].reshape(*ids.shape[:2], length - ids.shape[-1]).contiguous()


@dataclass
class ZipPackedTensor:
    parts: tuple[torch.Tensor, ...]
    important_ids: torch.Tensor
    unimportant_ids: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    kind: str
    config: ZipCacheConfig

    @classmethod
    def pack(cls, x, low_ids, kind, config):
        if kind not in ("key", "value"):
            raise ValueError("kind must be key or value")
        if not 0 < low_ids.shape[-1] < x.shape[-2]:
            raise ValueError("ZipCache needs nonempty important/unimportant groups")
        high_ids = important_complement(low_ids, x.shape[-2])
        source = official_codec()
        fn = source.true_mixedprec_compress_channelwise if kind == "key" else source.true_channel_separate_mixedprec_tokenwise_compress
        parts = fn(x, high_ids, low_ids, config.important_bits, config.unimportant_bits)
        # Upstream does not guard zero channel scales. Never silently record
        # nonfinite quantization as a successful model evaluation.
        if not all(bool(torch.isfinite(t).all()) for t in parts if t.is_floating_point()):
            raise ValueError("Official ZipCache codec produced nonfinite scales; degenerate input unsupported")
        return cls(tuple(parts), high_ids, low_ids, tuple(x.shape), x.dtype, kind, config)

    def dequantize(self):
        source = official_codec()
        hi, hi_min, hi_step, lo, lo_min, lo_step, *extra = self.parts
        args = (hi, hi_min, hi_step, self.config.important_bits,
                lo, lo_min, lo_step, self.config.unimportant_bits)
        tail = (self.important_ids, self.unimportant_ids, self.dtype, self.shape)
        if self.kind == "key":
            return source.true_mixedprec_decompress_channelwise(*args, *tail)
        return source.true_channel_separate_mixedprec_tokenwise_decompress(*args, *extra, *tail)


def probe_attention_scores(query, keys, probes, chunk_size, *, causal, normalize):
    """Official recent+random query scoring with bounded temporary allocations.

    GQA-group sums round to the model dtype as upstream. The probe reduction
    accumulates in FP32 across chunks, then rounds once before normalization.
    All probes are retained; chunking is not a reduced-probe approximation.
    """
    batch, query_heads, _, dim = query.shape
    kv_heads, length = keys.shape[1:3]
    groups = query_heads // kv_heads
    total = torch.zeros((batch, kv_heads, length), device=query.device, dtype=torch.float32)
    key_positions = torch.arange(length, device=query.device)
    for start in range(0, probes.numel(), chunk_size):
        positions = probes[start:start + chunk_size]
        q = query[:, :, positions, :].reshape(batch, kv_heads, groups * positions.numel(), dim)
        weights = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(dim)
        weights = weights.view(batch, kv_heads, groups, positions.numel(), length)
        if causal:
            weights.masked_fill_(key_positions[None, None, None, None, :] > positions[None, None, None, :, None],
                                 torch.finfo(query.dtype).min)
        probabilities = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        total += probabilities.sum(dim=2).float().sum(dim=2)
    total = total.to(query.dtype)
    if normalize:
        total = total / torch.arange(length, 0, -1, device=query.device)
    return total


@dataclass
class _Layer:
    key: ZipPackedTensor
    value: ZipPackedTensor
    length: int
    sealed_length: int
    decode_steps: int = 0
    tail_key: torch.Tensor | None = None
    tail_value: torch.Tensor | None = None
    packed_nbytes: int = 0
    current_nbytes: int = 0


class ZipCache(Cache):
    """Batch-one append-only cache; full history is recompressed every 100 steps."""
    is_compileable = False

    def __init__(self, config=None, *, seed=42):
        super().__init__()
        self.config = config or ZipCacheConfig()
        self.layers_by_id: dict[int, _Layer] = {}
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.seed = seed
        self.compressions = 0
        self.decode_calls = 0
        self.live_nbytes = 0
        self.peak_nbytes = 0

    def _record_layer_bytes(self, layer, *, recompressed):
        if recompressed:
            storages = {}
            for packed in (layer.key, layer.value):
                for tensor in (*packed.parts, packed.important_ids, packed.unimportant_ids):
                    storage = tensor.untyped_storage()
                    storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
            layer.packed_nbytes = sum(storages.values())
        current = layer.packed_nbytes + sum(t.untyped_storage().nbytes() for t in
                                             (layer.tail_key, layer.tail_value) if t is not None)
        self.live_nbytes += current - layer.current_nbytes
        layer.current_nbytes = current
        self.peak_nbytes = max(self.peak_nbytes, self.live_nbytes)

    def get_seq_length(self, layer_idx=0):
        layer = self.layers_by_id.get(0 if layer_idx is None else layer_idx)
        return layer.length if layer else 0

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.get_seq_length(layer_idx) + cache_position.numel(), 0

    def get_max_cache_shape(self):
        return None

    def update(self, *args, **kwargs):
        raise RuntimeError("ZipCache requires enable_zipcache_qwen2() to receive saliency queries")

    @torch.no_grad()
    def append(self, query, key, value, layer_idx):
        if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
            raise ValueError("ZipCache requires matching batch-one [B,H,T,D] K/V")
        if query.shape[1] % key.shape[1] or query.shape[-2:] != key.shape[-2:]:
            raise ValueError("Query shape or GQA grouping mismatch")
        config = self.config
        layer = self.layers_by_id.get(layer_idx)
        if layer is None:
            length = key.shape[-2]
            low_count = int(config.unimportant_ratio * length)
            if low_count < 1 or low_count >= length:
                raise ValueError("Prompt too short for the ZipCache mixed-precision groups")
            recent_start = int(length - .05 * length)
            probes = torch.cat((torch.arange(recent_start, length),
                                torch.randint(0, recent_start, (int(.05 * length),), generator=self.rng))).to(query.device)
            scores = probe_attention_scores(query, key, probes, config.probe_chunk_size, causal=True, normalize=True)
            low = scores.topk(low_count, dim=-1, largest=False).indices
            self.layers_by_id[layer_idx] = _Layer(ZipPackedTensor.pack(key, low, "key", config),
                                                   ZipPackedTensor.pack(value, low, "value", config), length, length)
            self._record_layer_bytes(self.layers_by_id[layer_idx], recompressed=True)
            self.compressions += 1
            return key, value  # Prefill attention uses original KV, as upstream.
        if key.shape[-2] != 1:
            raise ValueError("Only single-token decode after prefill is supported")
        if key.shape[:2] != layer.key.shape[:2] or key.shape[-1] != layer.key.shape[-1] or key.dtype != layer.key.dtype:
            raise ValueError("Cache dimensions/dtype must remain unchanged")
        tail_key = key if layer.tail_key is None else torch.cat((layer.tail_key, key), dim=2)
        tail_value = value if layer.tail_value is None else torch.cat((layer.tail_value, value), dim=2)
        full_key = torch.cat((layer.key.dequantize(), tail_key), dim=2)
        full_value = torch.cat((layer.value.dequantize(), tail_value), dim=2)
        layer.decode_steps += 1
        layer.length += 1
        self.decode_calls += 1
        if layer.decode_steps % config.streaming_gap == 0:
            probes = torch.zeros(1, device=query.device, dtype=torch.long)
            scores = probe_attention_scores(query, full_key, probes, config.probe_chunk_size, causal=False, normalize=False)
            tail_scores = scores[..., layer.sealed_length:]
            new_low = tail_scores.topk(int(config.streaming_gap * config.unimportant_ratio), dim=-1, largest=False).indices + layer.sealed_length
            low = torch.cat((layer.key.unimportant_ids, new_low), dim=-1)
            layer.key = ZipPackedTensor.pack(full_key, low, "key", config)
            layer.value = ZipPackedTensor.pack(full_value, low, "value", config)
            layer.sealed_length = layer.length
            layer.tail_key = layer.tail_value = None
            self.compressions += 1
        else:
            # Own compact allocations; do not retain model projections/views.
            layer.tail_key = tail_key.clone(memory_format=torch.contiguous_format)
            layer.tail_value = tail_value.clone(memory_format=torch.contiguous_format)
        self._record_layer_bytes(layer, recompressed=layer.decode_steps % config.streaming_gap == 0)
        return full_key, full_value

    def memory_stats(self):
        storages = {}
        def add(tensor, category):
            if tensor is None:
                return
            storage = tensor.untyped_storage()
            storages.setdefault((str(tensor.device), storage.data_ptr()), (storage.nbytes(), category))
        for layer in self.layers_by_id.values():
            for packed in (layer.key, layer.value):
                for index, part in enumerate(packed.parts):
                    add(part, "payload_nbytes" if index in (0, 3) else "metadata_nbytes")
                add(packed.important_ids, "metadata_nbytes")
                add(packed.unimportant_ids, "metadata_nbytes")
            add(layer.tail_key, "residual_nbytes")
            add(layer.tail_value, "residual_nbytes")
        stats = {name: sum(n for n, category in storages.values() if category == name)
                 for name in ("payload_nbytes", "metadata_nbytes", "residual_nbytes")}
        return {**stats, "nbytes": sum(stats.values()), "peak_nbytes": self.peak_nbytes,
                "layers": len(self.layers_by_id),
                "seq_length": self.get_seq_length(), "compressions": self.compressions,
                "decode_calls": self.decode_calls, "probe_seed": self.seed,
                "max_tail_tokens": max((layer.length - layer.sealed_length for layer in self.layers_by_id.values()), default=0)}

    @property
    def nbytes(self):
        return self.memory_stats()["nbytes"]


@dataclass
class ZipCacheHandle:
    originals: list[tuple[Any, Any]] = field(default_factory=list)

    def restore(self):
        for module, forward in self.originals:
            module.forward = forward
            delattr(module, "_zipcache_original_forward")
        self.originals.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def enable_zipcache_qwen2(model):
    """Instance-local adapter preserving Qwen projection biases, RoPE and SDPA."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if model.config.model_type != "qwen2" or model.config._attn_implementation != "sdpa" or model.training:
        raise ValueError("ZipCache port requires dense Qwen2/Qwen2.5 in eval mode with SDPA")
    modules = [layer.self_attn for layer in model.model.layers]
    if any(not isinstance(m, Qwen2Attention) or m.sliding_window is not None or hasattr(m, "_zipcache_original_forward") for m in modules):
        raise ValueError("Expected unpatched Qwen2 full-attention modules")

    def forward(attention, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
        if not isinstance(past_key_value, ZipCache):
            return attention._zipcache_original_forward(hidden_states, position_embeddings, attention_mask,
                                                       past_key_value=past_key_value, cache_position=cache_position, **kwargs)
        if torch.is_grad_enabled() or attention.training or attention_mask is not None or kwargs.get("output_attentions", False):
            raise ValueError("ZipCache port supports inference-only unpadded causal SDPA without output_attentions")
        shape = hidden_states.shape[:-1]
        projected = (*shape, -1, attention.head_dim)
        query = attention.q_proj(hidden_states).view(projected).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(projected).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(projected).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        key, value = past_key_value.append(query, key, value, attention.layer_idx)
        result, _ = sdpa_attention_forward(attention, query, key, value, None, dropout=0.0, scaling=attention.scaling)
        return attention.o_proj(result.reshape(*shape, -1).contiguous()), None

    handle = ZipCacheHandle()
    for module in modules:
        handle.originals.append((module, module.forward))
        module._zipcache_original_forward = module.forward
        module.forward = MethodType(forward, module)
    return handle
