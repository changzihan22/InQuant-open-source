"""Single-active-request vLLM V1 backend with bounded exact tail storage."""
from dataclasses import dataclass
import math
import os
import torch
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm.vllm_flash_attn import flash_attn_varlen_func
from inquant.triton_attention import SharedDecodeWorkspace
from .layout import PageLayout, pack_pages, pack_pages_into
from .kernels import paged_decode

# V1 worker executes layers in stream order; the platform rejects concurrency,
# graphs and parallel ranks. Shared workspace verifies stream identity itself.
WORKSPACE = SharedDecodeWorkspace()


@dataclass
class InQuantMetadata:
    num_actual_tokens: int
    seq_len: int
    context_len: int
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    request_id: str


class InQuantMetadataBuilder:
    def __init__(self, runner, kv_cache_spec, block_table):
        self.runner, self.block_table = runner, block_table

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def use_cascade_attention(self, *args, **kwargs):
        return False

    def build(self, num_reqs, num_actual_tokens, max_query_len, common_prefix_len, common_attn_metadata):
        if num_reqs != 1 or common_prefix_len:
            raise RuntimeError("InQuant requires one active request and no shared prefix")
        seq_len = int(self.runner.seq_lens_np[0])
        context_len = seq_len - num_actual_tokens
        if context_len and num_actual_tokens != 1:
            raise RuntimeError("Chunked prefill/speculative multi-token decode is unsupported")
        return InQuantMetadata(num_actual_tokens, seq_len, context_len,
                               common_attn_metadata.query_start_loc,
                               self.block_table.get_device_tensor()[0],
                               self.runner.input_batch.req_ids[0])


class InQuantBackend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return "INQUANT_PAGED_K4V2_V1"

    @staticmethod
    def get_supported_head_sizes():
        return [128]

    @staticmethod
    def get_impl_cls():
        return InQuantImpl

    @staticmethod
    def get_metadata_cls():
        return InQuantMetadata

    @staticmethod
    def get_builder_cls():
        return InQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return (num_blocks, PageLayout(block_size, num_kv_heads, head_size).page_bytes)

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1)


class InQuantImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.alibi_slopes is not None or self.sliding_window != (-1, -1) or self.logits_soft_cap:
            raise ValueError("InQuant supports full causal attention without ALiBi/softcap")
        if self.kv_sharing_target_layer_name is not None or self.use_irope:
            raise ValueError("KV sharing and iRoPE are not implemented")
        if self.num_queries_per_kv > 16:
            raise ValueError("At most 16 query heads per KV head")
        config = get_current_vllm_config()
        self.layout = PageLayout(config.cache_config.block_size, self.num_kv_heads, self.head_size)
        self.max_pages = math.ceil(config.model_config.max_model_len / self.layout.block_size)
        self.dense = None
        self.request_id = None
        self.tokens = 0
        self.max_tokens_seen = 0
        self.sealed_pages = 0
        self.prefill_calls = self.decode_calls = self.seal_calls = 0

    def _allocate_aux(self, query):
        if self.dense is None:
            self.dense = torch.empty((2, self.num_kv_heads, self.layout.block_size + 4, 128),
                                     dtype=query.dtype, device=query.device)
            WORKSPACE.acquire((self.num_heads, self.max_pages + 1, 130), query.device)

    def _seal(self, keys, values, cache, block_table, start_page):
        # Pack all new full pages together; one scatter follows vLLM's physical
        # block table, including arbitrary page order and reused page IDs.
        ids = block_table[start_page:start_page + keys.shape[0]]
        if os.environ.get("INQUANT_DIRECT_WRITE") == "1":
            pack_pages_into(keys, values, self.layout, cache, ids,
                            donor_backend=os.environ.get("INQUANT_DONOR_BACKEND", "torch"))
        else:
            packed = pack_pages(keys, values, self.layout)
            cache.index_copy_(0, ids.long(), packed)
        self.seal_calls += 1

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        if output is None:
            raise ValueError("vLLM must supply an output buffer")
        self._allocate_aux(query)
        if attn_metadata is None:
            # Aux storage is allocated during memory profiling, before vLLM
            # budgets its pool. Do not mutate request state on profile calls.
            return output
        m, b = attn_metadata, self.layout.block_size
        n = m.num_actual_tokens
        if kv_cache.dtype != torch.uint8 or kv_cache.shape[1] != self.layout.page_bytes:
            raise RuntimeError("InQuant cache spec was not installed; refusing dense fallback")
        if m.context_len == 0:
            self.request_id = m.request_id
            self.tokens = n
            self.sealed_pages = n // b
            sink = min(4, n)
            self.dense[0, :, :sink].copy_(key[:sink].transpose(0, 1))
            self.dense[1, :, :sink].copy_(value[:sink].transpose(0, 1))
            # Full prefill attends to fresh uncompressed K/V, like the HF
            # implementation. No dense history is retained after this call.
            flash_attn_varlen_func(
                q=query[:n], k=key[:n], v=value[:n], out=output[:n],
                cu_seqlens_q=m.query_start_loc, cu_seqlens_k=m.query_start_loc,
                max_seqlen_q=n, max_seqlen_k=n, causal=True,
                softmax_scale=self.scale, fa_version=self.vllm_flash_attn_version,
            )
            if self.sealed_pages:
                full = self.sealed_pages * b
                keys = key[:full].view(self.sealed_pages, b, self.num_kv_heads, 128).transpose(1, 2)
                values = value[:full].view(self.sealed_pages, b, self.num_kv_heads, 128).transpose(1, 2)
                self._seal(keys, values, kv_cache, m.block_table, 0)
            rem = n % b
            if rem:
                self.dense[0, :, 4:4 + rem].copy_(key[n - rem:n].transpose(0, 1))
                self.dense[1, :, 4:4 + rem].copy_(value[n - rem:n].transpose(0, 1))
            self.prefill_calls += 1
        else:
            if m.request_id != self.request_id or m.context_len != self.tokens or n != 1:
                raise RuntimeError("InQuant request state mismatch; refusing stale tail reuse")
            tail = self.tokens % b
            self.dense[0, :, 4 + tail].copy_(key[0])
            self.dense[1, :, 4 + tail].copy_(value[0])
            if self.tokens < 4:
                self.dense[0, :, self.tokens].copy_(key[0])
                self.dense[1, :, self.tokens].copy_(value[0])
            self.tokens += 1
            paged_decode(query[:1], kv_cache, m.block_table, self.dense,
                         self.sealed_pages, tail + 1, min(4, self.tokens),
                         self.layout, WORKSPACE, self.scale, output[:1])
            # The query that fills a page still uses its exact values; sealing
            # occurs after attention so only subsequent queries see quantization.
            if tail + 1 == b:
                self._seal(self.dense[0, :, 4:].unsqueeze(0),
                           self.dense[1, :, 4:].unsqueeze(0), kv_cache,
                           m.block_table, self.sealed_pages)
                self.sealed_pages += 1
            self.decode_calls += 1
        self.max_tokens_seen = max(self.max_tokens_seen, self.tokens)
        return output
