"""Fused paged K4/V2 decode. No dense history or host page lookup."""
import torch
import triton
import triton.language as tl
from inquant.triton_attention import _decode_tile, _decode_value_tile, _reduce_partials


@triton.jit
def _paged_partials(Q, CACHE, BLOCKS, DENSE, PARTIAL,
                    N_PAGES, TAIL_T, SINK_T, PARTITIONS,
                    SCALE: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr,
                    B: tl.constexpr, PAGE_BYTES: tl.constexpr,
                    KS: tl.constexpr, KM: tl.constexpr, KP: tl.constexpr,
                    VP: tl.constexpr, VS: tl.constexpr, VO: tl.constexpr,
                    TILE: tl.constexpr = 32, Q_TILE: tl.constexpr = 16):
    kvh, part = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    g = tl.arange(0, Q_TILE)
    head = kvh * GROUPS + g
    q = tl.load(Q + head[:, None] * 128 + d[None, :], mask=g[:, None] < GROUPS, other=0)
    m = tl.full((Q_TILE,), float('-inf'), tl.float32)
    norm = tl.full((Q_TILE,), 0., tl.float32)
    acc = tl.full((Q_TILE, 128), 0., tl.float32)
    if part < N_PAGES:
        block = tl.load(BLOCKS + part).to(tl.int64)
        page = CACHE + block * PAGE_BYTES
        scales = (page + KS).to(tl.pointer_type(tl.float32))
        roles = (page + KM).to(tl.pointer_type(tl.int16))
        padding = (page + KP).to(tl.pointer_type(tl.float32))
        vscales = (page + VS).to(tl.pointer_type(tl.float16))
        offsets = (page + VO).to(tl.pointer_type(tl.float16))
        for base in range(B // TILE):
            t = base * TILE + tl.arange(0, TILE)
            valid = (part * B + t) >= SINK_T
            keys = _decode_tile(page, scales, padding, roles, t, d, valid, kvh, H, B, 128, 16)
            values = _decode_value_tile(page + VP, vscales, offsets, t, d, valid, kvh, B, 128, 2, 64)
            score = tl.dot(q, tl.trans(keys.to(q.dtype)), input_precision='tf32x3') * SCALE
            score = tl.where(valid[None, :], score, float('-inf'))
            next_m = tl.maximum(m, tl.max(score, 1))
            alpha = tl.exp(m - next_m)
            p = tl.exp(score - next_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), values.to(q.dtype), input_precision='tf32x3')
            norm = norm * alpha + tl.sum(p, 1)
            m = next_m
    else:
        for base in range(triton.cdiv(B + 4, TILE)):
            t = base * TILE + tl.arange(0, TILE)
            valid = (t < SINK_T) | ((t >= 4) & (t < 4 + TAIL_T) & (N_PAGES * B + t - 4 >= SINK_T))
            pos = (kvh * (B + 4) + t[:, None]) * 128 + d[None, :]
            keys = tl.load(DENSE + pos, mask=valid[:, None], other=0)
            values = tl.load(DENSE + H * (B + 4) * 128 + pos, mask=valid[:, None], other=0)
            score = tl.dot(q, tl.trans(keys), input_precision='tf32x3') * SCALE
            score = tl.where(valid[None, :], score, float('-inf'))
            next_m = tl.maximum(m, tl.max(score, 1))
            alpha = tl.exp(m - next_m)
            p = tl.exp(score - next_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), values, input_precision='tf32x3')
            norm = norm * alpha + tl.sum(p, 1)
            m = next_m
    ptr = PARTIAL + (head * PARTITIONS + part) * 130
    tl.store(ptr[:, None] + d[None, :], acc, mask=g[:, None] < GROUPS)
    tl.store(ptr + 128, m, mask=g < GROUPS)
    tl.store(ptr + 129, norm, mask=g < GROUPS)


def paged_decode(query, cache, blocks, dense, num_pages, tail_tokens,
                 sink_tokens, layout, workspace, scale, output=None):
    if query.shape[0] != 1 or query.shape[-1] != 128 or query.dtype != torch.bfloat16:
        raise ValueError("Paged decode expects one BF16 query [1, heads, 128]")
    hq, hkv = query.shape[1], layout.heads
    if hq % hkv or hq // hkv > 16:
        raise ValueError("Expected 1..16 query heads per KV head")
    if not query.is_contiguous():
        query = query.contiguous()
    parts = num_pages + 1
    scratch = workspace.acquire((hq, parts, 130), query.device)
    if output is None:
        output = torch.empty_like(query)
    offsets = layout.offsets
    _paged_partials[(hkv, parts)](
        query, cache, blocks, dense, scratch, num_pages, tail_tokens, sink_tokens, parts,
        scale, hkv, hq // hkv, layout.block_size, layout.page_bytes,
        offsets['key_scales'], offsets['key_map'], offsets['key_padding'],
        offsets['value_payload'], offsets['value_scales'], offsets['value_offsets'],
        num_warps=4,
    )
    _reduce_partials[(hq, 4)](scratch, output, parts, 128, triton.next_power_of_2(parts), 32, num_warps=4)
    return output
