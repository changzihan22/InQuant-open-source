"""Versioned byte layout. Every packed page owns all its decoding metadata."""
from dataclasses import dataclass
import torch
from inquant.codec import CodecConfig, quantize
from inquant.value_codec import quantize_values


@dataclass(frozen=True)
class PageLayout:
    block_size: int
    heads: int
    dim: int = 128

    def __post_init__(self):
        if self.block_size not in (64, 128, 256) or self.dim != 128 or self.heads < 1:
            raise ValueError("Expected block_size 64/128/256, head_dim 128, positive KV heads")

    @property
    def fields(self):
        b, h, d = self.block_size, self.heads, self.dim
        return (
            ("key_payload", torch.uint8, (h, b, d // 2)),
            ("key_scales", torch.float32, (h, d)),
            ("key_map", torch.int16, (h, d)),
            ("key_padding", torch.float32, (h, d // 8)),
            ("value_payload", torch.uint8, (h, b, d // 4)),
            ("value_scales", torch.float16, (h, b, d // 64)),
            ("value_offsets", torch.float16, (h, b, d // 64)),
        )

    @property
    def offsets(self):
        import math
        offset, result = 0, {}
        for name, dtype, shape in self.fields:
            result[name] = offset
            offset += math.prod(shape) * dtype.itemsize
        return result

    @property
    def page_bytes(self):
        return self.heads * (104 * self.block_size + 832)

    def views(self, pages):
        import math
        if pages.dtype != torch.uint8 or pages.ndim != 2 or pages.shape[1] != self.page_bytes:
            raise ValueError("Expected [pages, page_bytes] uint8 storage")
        out = {}
        for name, dtype, shape in self.fields:
            offset = self.offsets[name]
            size = math.prod(shape) * dtype.itemsize
            out[name] = pages[:, offset:offset + size].view(dtype).view(-1, *shape)
        return out


@torch.no_grad()
def pack_pages(keys, values, layout):
    """Reference sealing path, batched over full temporal pages [P,H,B,D].

    Only freshly sealed pages are quantized. Existing packed history is never
    dequantized/requantized. Temporary PyTorch quantization allocations are not
    part of persistent cache; worker CUDA peak counters include their peak.
    """
    if keys.shape != values.shape or tuple(keys.shape[1:]) != (layout.heads, layout.block_size, 128):
        raise ValueError("Invalid full-page K/V shape")
    k = quantize(keys, CodecConfig(donor_policy="min_error"))
    v = quantize_values(values, bits=2, group_size=64)
    result = torch.empty((keys.shape[0], layout.page_bytes), dtype=torch.uint8, device=keys.device)
    fields = layout.views(result)
    fields["key_payload"].copy_(k.payload)
    fields["key_scales"].copy_(k.scales)
    fields["key_padding"].copy_(k.padding)
    role = fields["key_map"]
    role.fill_(-1)
    role.scatter_(-1, k.salient_channels.long(), k.donor_channels.to(torch.int16))
    donor_roles = -2 - torch.arange(16, dtype=torch.int16, device=keys.device)
    role.scatter_(-1, k.donor_channels.long(), donor_roles.expand_as(k.donor_channels))
    fields["value_payload"].copy_(v.payload)
    fields["value_scales"].copy_(v.scales)
    fields["value_offsets"].copy_(v.offsets)
    return result


def unpack_pages_reference(pages, layout, dtype=torch.bfloat16):
    """Testing/inspection only. The production decode never calls this."""
    fields = layout.views(pages)
    role = fields["key_map"].long()
    slots = torch.stack((fields["key_payload"] & 15, fields["key_payload"] >> 4), -1).flatten(-2)
    partner = slots.gather(-1, role.clamp_min(0).unsqueeze(-2).expand_as(slots))
    code = torch.where(role.unsqueeze(-2) >= 0,
                       (slots.int() << 4 | partner.int()) - 128, slots.int() - 8)
    keys = code.float() * fields["key_scales"].unsqueeze(-2)
    padding = fields["key_padding"].gather(-1, (-role - 2).clamp_min(0))
    keys = torch.where(role.unsqueeze(-2) <= -2, padding.unsqueeze(-2), keys)
    vp = fields["value_payload"]
    codes = torch.stack([(vp >> i) & 3 for i in (0, 2, 4, 6)], -1).flatten(-2)
    values = codes.float().reshape(*codes.shape[:-1], 2, 64)
    values = (values * fields["value_scales"].float().unsqueeze(-1)
              + fields["value_offsets"].float().unsqueeze(-1))
    return keys.to(dtype), values.flatten(-2).to(dtype)


@torch.no_grad()
def pack_pages_into(keys, values, layout, destination, page_ids, *, donor_backend="torch"):
    """Seal fresh pages directly into their assigned physical slots.

    The page allocator owns unique in-range IDs. No packed staging page pool
    or host page lookup is created; codec temporaries remain separately owned.
    """
    if keys.shape != values.shape or tuple(keys.shape[1:]) != (layout.heads, layout.block_size, 128):
        raise ValueError("Invalid full-page K/V shape")
    from .packing import write_codec_pages
    k = quantize(keys, CodecConfig(donor_policy="min_error", donor_backend=donor_backend))
    v = quantize_values(values, bits=2, group_size=64)
    write_codec_pages(k, v, layout, destination, page_ids)
