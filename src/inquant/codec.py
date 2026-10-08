"""A physically packed, device-resident InQuant reference representation.

The layout follows Algorithms 1/2 and Appendices D/E of the supplied paper:
every channel owns one nibble, and a salient channel borrows one other channel's
nibble for its 8-bit code. The descriptor is the sole authority for decoding.
There is no separate outlier-value buffer and no in-band marker sentinel.

This is a correctness reference, not the paper's fused CUDA implementation.
Explicit choices where the paper is underspecified are:

* saliency and top-k selection are per batch item/head, over every eighth token;
* a head's entire channel dimension is one group, with no cross-head fallback;
* immediate donors are preferred, then a unit-stride search expands from the
  contiguous salient group's boundaries; ties prefer lower sampled saliency;
* symmetric per-channel FP32 scales use full temporal-block maxima;
* every overwritten donor value uses that channel's sampled temporal mean;
* explicit source/donor indices replace bitmap/stride/marker optimizations.

The optional ``min_error`` donor policy is an engineering variation: it first
restricts nonsalient donors to the smallest full-block mean-square error under
sampled-mean reconstruction, then assigns them one-to-one. The default
``neighbors`` policy retains the paper-inspired local search described above.

Callers must bound temporal block size. All persistent arrays stay on the input
device. Python loops and validation can synchronize CUDA; no speed claim is made.
Memory accounting includes all persistent tensor storage, including descriptors,
scales, and donor padding, but excludes Python objects and temporary workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CodecConfig:
    sample_stride: int = 8
    salient_fraction: float = 0.125
    donor_policy: str = "neighbors"
    donor_backend: str = "torch"

    def __post_init__(self) -> None:
        if self.donor_backend not in ("torch", "triton", "auto"):
            raise ValueError("donor_backend must be torch, triton or auto")
        if isinstance(self.sample_stride, bool) or not isinstance(self.sample_stride, int):
            raise ValueError("sample_stride must be a positive integer")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be a positive integer")
        if not math.isfinite(self.salient_fraction) or not 0 <= self.salient_fraction <= 0.5:
            raise ValueError("salient_fraction must be finite and in [0, 0.5]")
        if self.donor_policy not in ("neighbors", "min_error"):
            raise ValueError("donor_policy must be neighbors or min_error")


@dataclass(frozen=True)
class PackedTensor:
    """Persistent packed values and a descriptor shared by every token in a block.

    ``salient_channels[..., i]`` borrows ``donor_channels[..., i]``. These
    disjoint, one-to-one indices are sufficient to recover the layout even if
    any ordinary data nibble happens to match a proposed marker pattern.
    """

    payload: torch.Tensor
    scales: torch.Tensor
    salient_channels: torch.Tensor
    donor_channels: torch.Tensor
    padding: torch.Tensor
    shape: tuple[int, int, int, int]
    original_dtype: torch.dtype

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "payload": self.payload,
            "scales": self.scales,
            "salient_channels": self.salient_channels,
            "donor_channels": self.donor_channels,
            "padding": self.padding,
        }

    @property
    def payload_nbytes(self) -> int:
        return self.payload.untyped_storage().nbytes()

    @property
    def nbytes(self) -> int:
        # quantize() creates independently owned, compact contiguous tensors.
        return sum(t.untyped_storage().nbytes() for t in self.tensors.values())

    @property
    def metadata_nbytes(self) -> int:
        return self.nbytes - self.payload_nbytes

    def storage_stats(self) -> dict[str, int | float]:
        source_nbytes = math.prod(self.shape) * self.original_dtype.itemsize
        return {
            "payload_nbytes": self.payload_nbytes,
            "scale_nbytes": self.scales.untyped_storage().nbytes(),
            "descriptor_nbytes": self.salient_channels.untyped_storage().nbytes()
            + self.donor_channels.untyped_storage().nbytes(),
            "padding_nbytes": self.padding.untyped_storage().nbytes(),
            "metadata_nbytes": self.metadata_nbytes,
            "nbytes": self.nbytes,
            "source_nbytes": source_nbytes,
            "compression_ratio": source_nbytes / self.nbytes,
        }

    @torch.no_grad()
    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Recover a dense tensor, including sampled-mean donor approximation."""
        dtype = self.original_dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dequantize dtype must be floating point")
        # The even channel's slot occupies the byte's low nibble. A salient
        # channel's slot, irrespective of parity, holds its code's high nibble.
        slots = torch.stack((self.payload & 15, self.payload >> 4), dim=-1)
        slots = slots.flatten(-2)
        decoded = (slots.to(torch.float32) - 8) * self.scales.unsqueeze(-2)
        if self.salient_channels.shape[-1]:
            sources = self.salient_channels.to(torch.int64)
            donors = self.donor_channels.to(torch.int64)
            source_tokens = sources.unsqueeze(-2).expand(-1, -1, self.shape[-2], -1)
            donor_tokens = donors.unsqueeze(-2).expand_as(source_tokens)
            high = slots.gather(-1, source_tokens)
            low = slots.gather(-1, donor_tokens)
            codes = ((high << 4) | low).to(torch.float32) - 128
            source_scales = self.scales.gather(-1, sources).unsqueeze(-2)
            decoded.scatter_(-1, source_tokens, codes * source_scales)
            decoded.scatter_(-1, donor_tokens, self.padding.unsqueeze(-2).expand_as(codes))
        return decoded.to(dtype=dtype)


def _reuse_map(saliency: torch.Tensor, count: int, donor_error: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose disjoint sources/donors using only tensors on saliency.device."""
    *prefix, channels = saliency.shape
    positions = torch.arange(channels, device=saliency.device)
    # A stable ordering makes all-zero blocks and ties reproducible.
    sources = torch.argsort(saliency, dim=-1, descending=True, stable=True)[..., :count]
    sources = sources.sort(dim=-1).values
    selected = torch.zeros_like(saliency, dtype=torch.bool)
    selected.scatter_(-1, sources, True)
    available = ~selected
    if donor_error is not None:
        # Engineering variant: keep the paper's L1 source selection, but only
        # borrow from the nonsalient channels with the smallest actual error
        # under sampled-mean reconstruction. The neighbor search then assigns
        # these slots one-to-one. Metadata and physical payload are unchanged.
        pool = torch.argsort(donor_error.masked_fill(selected, float("inf")), dim=-1, stable=True)[..., :count]
        available = torch.zeros_like(selected)
        available.scatter_(-1, pool, True)
    donors = torch.empty((*prefix, count), dtype=torch.int64, device=saliency.device)

    for i in range(count):
        source = sources[..., i, None]
        distance = (positions - source).abs()
        immediate = available & (distance == 1)

        # Find the boundaries of this contiguous run of salient channels. All
        # candidates lie outside the run; expand from either boundary in steps
        # of one until an unused low-saliency slot is reached.
        left = torch.where((positions < source) & ~selected, positions, -1).amax(-1) + 1
        right = torch.where((positions > source) & ~selected, positions, channels).amin(-1) - 1
        boundary_distance = torch.minimum(
            (positions - left.unsqueeze(-1)).abs(),
            (positions - right.unsqueeze(-1)).abs(),
        )
        nearest = boundary_distance.masked_fill(~available, channels + 1).amin(-1, keepdim=True)
        fallback = available & (boundary_distance == nearest)
        candidates = torch.where(immediate.any(-1, keepdim=True), immediate, fallback)
        donor = saliency.masked_fill(~candidates, float("inf")).argmin(-1)
        donors[..., i] = donor
        available.scatter_(-1, donor.unsqueeze(-1), False)

    return sources, donors


@torch.no_grad()
def quantize(x: torch.Tensor, config: CodecConfig = CodecConfig()) -> PackedTensor:
    """Pack one ``[batch, heads, tokens, even_head_dim]`` temporal KV block.

    ``floor(head_dim * salient_fraction)`` channels receive 8-bit precision;
    the same number of low-saliency channels are replaced by sampled means.
    Keys and values must be passed independently. Empty or nonfinite inputs
    and odd channel dimensions are rejected explicitly.
    """
    if x.ndim != 4 or any(size < 1 for size in x.shape):
        raise ValueError("x must have nonempty [batch, heads, tokens, channels] dimensions")
    if not x.is_floating_point():
        raise ValueError("x must have a floating-point dtype")
    if x.shape[-1] % 2:
        raise ValueError("head dimension must be even for two nibbles per byte")
    x32 = x.to(torch.float32)
    if not bool(torch.isfinite(x32).all()):
        raise ValueError("x must contain finite values representable in float32")

    batch, heads, tokens, channels = x.shape
    count = math.floor(channels * config.salient_fraction)
    samples = x32[..., ::config.sample_stride, :]
    # Mean(abs) avoids overflow in the full-token L1 sum. The omitted scalar
    # factor is shared by all channels and cannot alter their ranking.
    saliency = samples.abs().mean(dim=-2)
    sampled_mean = samples.mean(dim=-2)
    donor_error = None
    if config.donor_policy == "min_error":
        variance, full_mean = torch.var_mean(x32, dim=-2, correction=0)
        donor_error = variance + (full_mean - sampled_mean).square()
    use_triton = config.donor_backend == "triton" or (
        config.donor_backend == "auto" and x.device.type == "cuda" and channels == 128)
    if use_triton:
        from .triton_selection import reuse_map
        sources, donors = reuse_map(saliency, count, donor_error)
    else:
        sources, donors = _reuse_map(saliency, count, donor_error)
    selected = torch.zeros_like(saliency, dtype=torch.bool)
    selected.scatter_(-1, sources, True)
    denominator = torch.where(selected, 127.0, 7.0)
    scales = (x32.abs().amax(dim=-2) / denominator).clamp_min(torch.finfo(torch.float32).tiny)

    slots = (torch.round(x32 / scales.unsqueeze(-2)).clamp(-7, 7) + 8).to(torch.uint8)
    if count:
        source_tokens = sources.unsqueeze(-2).expand(batch, heads, tokens, count)
        donor_tokens = donors.unsqueeze(-2).expand_as(source_tokens)
        values = x32.gather(-1, source_tokens)
        source_scales = scales.gather(-1, sources).unsqueeze(-2)
        codes = (torch.round(values / source_scales).clamp(-127, 127) + 128).to(torch.uint8)
        slots.scatter_(-1, source_tokens, codes >> 4)
        slots.scatter_(-1, donor_tokens, codes & 15)
    payload = (slots[..., 0::2] | (slots[..., 1::2] << 4)).contiguous()
    padding = sampled_mean.gather(-1, donors).contiguous()
    descriptor_dtype = torch.uint8 if channels <= 256 else torch.int32
    return PackedTensor(
        payload=payload,
        scales=scales.contiguous(),
        salient_channels=sources.to(descriptor_dtype).contiguous(),
        donor_channels=donors.to(descriptor_dtype).contiguous(),
        padding=padding,
        shape=tuple(x.shape),
        original_dtype=x.dtype,
    )
