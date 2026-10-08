"""One GPU program per head/group for deterministic reference donor selection.

Statistics and quantization remain unchanged. Unique integer sort keys reproduce
stable float32 ordering, including ties, without per-donor Python dispatch.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _select(SALIENCY, ERROR, SOURCES, DONORS, COUNT: tl.constexpr, USE_ERROR: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, 128)
    saliency = tl.load(SALIENCY + row * 128 + channel)
    # Scores are nonnegative; IEEE bits preserve order. Lower channels win ties.
    key = (saliency.to(tl.uint32, bitcast=True).to(tl.uint64) << 8) | (127-channel).to(tl.uint64)
    ordered = tl.sort(key, descending=True)
    threshold = tl.sum(tl.where(channel == COUNT-1, ordered, 0), axis=0)
    selected = key >= threshold
    available = ~selected
    if USE_ERROR:
        error = tl.load(ERROR + row * 128 + channel)
        error = tl.where(selected, float('inf'), error)
        error_key = (error.to(tl.uint32, bitcast=True).to(tl.uint64) << 8) | channel.to(tl.uint64)
        ordered_error = tl.sort(error_key, descending=False)
        maximum = tl.sum(tl.where(channel == COUNT-1, ordered_error, 0), axis=0)
        available = (~selected) & (error_key <= maximum)
    source_order = tl.sort(tl.where(selected, channel, 128), descending=False)
    for i in range(COUNT):
        source = tl.sum(tl.where(channel == i, source_order, 0), axis=0)
        immediate = available & (tl.abs(channel-source) == 1)
        left = tl.max(tl.where((channel < source) & ~selected, channel, -1), axis=0) + 1
        right = tl.min(tl.where((channel > source) & ~selected, channel, 128), axis=0) - 1
        distance = tl.minimum(tl.abs(channel-left), tl.abs(channel-right))
        nearest = tl.min(tl.where(available, distance, 129), axis=0)
        fallback = available & (distance == nearest)
        candidates = tl.where(tl.sum(immediate.to(tl.int32), axis=0) > 0, immediate, fallback)
        score = tl.min(tl.where(candidates, saliency, float('inf')), axis=0)
        donor = tl.min(tl.where(candidates & (saliency == score), channel, 128), axis=0)
        tl.store(SOURCES + row * COUNT + i, source)
        tl.store(DONORS + row * COUNT + i, donor)
        available = available & (channel != donor)


def reuse_map(saliency, count, donor_error=None):
    if saliency.device.type != 'cuda' or saliency.dtype != torch.float32 or saliency.shape[-1] != 128:
        raise ValueError('Fused selector requires CUDA float32 saliency with 128 channels')
    if not 0 <= count <= 64:
        raise ValueError('Expected 0..64 selected channels')
    if donor_error is not None and (donor_error.shape != saliency.shape or donor_error.dtype != saliency.dtype
                                   or donor_error.device != saliency.device):
        raise ValueError('Donor error must match saliency shape, dtype and device')
    sources = torch.empty((*saliency.shape[:-1], count), dtype=torch.int64, device=saliency.device)
    donors = torch.empty_like(sources)
    if count:
        _select[(saliency.numel()//128,)](saliency.contiguous(),
            saliency if donor_error is None else donor_error.contiguous(), sources, donors,
            count, donor_error is not None, num_warps=4)
    return sources, donors
