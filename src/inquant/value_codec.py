"""Experimental per-token affine INT2/INT4 values with physical byte packing.

This replaces only V; K retains its original InQuant representation. Parameters
are stored as FP16 scale and minimum per contiguous channel group. Codes are
computed against those stored parameters, and all owned storage is counted.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedValues:
    payload: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, int, int, int]
    original_dtype: torch.dtype
    bits: int
    group_size: int

    @property
    def tensors(self):
        return {"payload": self.payload, "scales": self.scales, "offsets": self.offsets}

    @property
    def payload_nbytes(self):
        return self.payload.untyped_storage().nbytes()

    @property
    def nbytes(self):
        return sum(t.untyped_storage().nbytes() for t in self.tensors.values())

    @property
    def metadata_nbytes(self):
        return self.nbytes - self.payload_nbytes

    @torch.no_grad()
    def dequantize(self, dtype=None):
        dtype = self.original_dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dequantize dtype must be floating point")
        mask = (1 << self.bits) - 1
        slots = torch.stack([(self.payload >> shift) & mask
                             for shift in range(0, 8, self.bits)], dim=-1).flatten(-2)
        grouped = slots.float().reshape(*self.shape[:-1], -1, self.group_size)
        dense = grouped * self.scales.float().unsqueeze(-1) + self.offsets.float().unsqueeze(-1)
        return dense.reshape(self.shape).to(dtype)


@torch.no_grad()
def quantize_values(x, bits=2, group_size=64):
    if type(bits) is not int or bits not in (2, 4):
        raise ValueError("Value bits must be 2 or 4")
    if type(group_size) is not int or group_size not in (32, 64, 128):
        raise ValueError("Value group size must be 32, 64 or 128")
    if x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point() or x.shape[-1] % group_size:
        raise ValueError("Expected nonempty floating [batch, heads, tokens, channels] divisible by group_size")
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    lower, upper = grouped.amin(-1), grouped.amax(-1)
    levels = (1 << bits) - 1
    offsets = lower.to(torch.float16).contiguous()
    scales = ((upper - lower) / levels).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16).contiguous()
    if not bool(torch.isfinite(grouped).all() & torch.isfinite(offsets).all() & torch.isfinite(scales).all()):
        raise ValueError("Values and stored FP16 quantization parameters must be finite")
    codes = ((grouped - offsets.float().unsqueeze(-1)) / scales.float().unsqueeze(-1))
    codes = codes.round().clamp(0, levels).to(torch.uint8).reshape(x.shape)
    per_byte = 8 // bits
    packed = codes[..., 0::per_byte].clone()
    for index in range(1, per_byte):
        packed |= codes[..., index::per_byte] << (index * bits)
    return PackedValues(packed.contiguous(), scales, offsets, tuple(x.shape), x.dtype, bits, group_size)


def quantize_value_blocks(tensor, block_count, block_size, bits, group_size):
    if not block_count:
        return []
    if block_count == 1:
        return [quantize_values(tensor[:, :, :block_size], bits, group_size)]
    _, heads, _, channels = tensor.shape
    blocked = tensor[0, :, :block_count * block_size].reshape(
        heads, block_count, block_size, channels).permute(1, 0, 2, 3)
    packed = quantize_values(blocked, bits, group_size)
    return [PackedValues(
        **{name: value[index:index + 1].clone(memory_format=torch.contiguous_format)
           for name, value in packed.tensors.items()},
        shape=(1, heads, block_size, channels), original_dtype=tensor.dtype, bits=bits, group_size=group_size,
    ) for index in range(block_count)]
