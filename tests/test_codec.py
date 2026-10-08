"""Layout and error checks for the executable CPU reference implementation."""

import math

import pytest
import torch

from inquant.codec import CodecConfig, quantize


def _fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(2, 3, 41, 16, generator=generator, dtype=dtype) * 0.1
    x[..., 6:10] *= 100  # A consecutive salient run needs nonadjacent donors.
    return x


def test_physical_payload_and_complete_memory_accounting():
    x = _fixture(torch.float16)
    packed = quantize(x, CodecConfig(salient_fraction=0.25))
    assert packed.payload.shape == (2, 3, 41, 8)
    assert packed.payload.dtype == torch.uint8
    assert packed.payload_nbytes == math.prod(x.shape) // 2
    assert set(packed.tensors) == {
        "payload", "scales", "salient_channels", "donor_channels", "padding"
    }
    assert all(t.device == x.device and t.is_contiguous() for t in packed.tensors.values())
    expected_metadata = 2 * 3 * (16 * 4 + 4 * (1 + 1 + 4))
    assert packed.metadata_nbytes == expected_metadata
    assert packed.nbytes == packed.payload_nbytes + expected_metadata
    assert packed.nbytes == sum(t.numel() * t.element_size() for t in packed.tensors.values())
    assert packed.storage_stats()["source_nbytes"] == x.numel() * x.element_size()
    assert packed.storage_stats()["compression_ratio"] < 4  # Metadata is not free.


def test_adjacent_sources_have_unique_nonsalient_donors():
    x = _fixture()
    packed = quantize(x, CodecConfig(salient_fraction=0.25))
    sources, donors = packed.salient_channels.long(), packed.donor_channels.long()
    assert torch.equal(sources, torch.tensor([6, 7, 8, 9]).expand_as(sources))
    assert torch.all(torch.sort(donors, dim=-1).values.diff(dim=-1) > 0)
    assert not torch.any((sources.unsqueeze(-1) == donors.unsqueeze(-2)))
    assert torch.any((sources - donors).abs() > 1)
    # The first source has an available immediate left neighbor.
    assert torch.all(donors[..., 0] == 5)


def test_descriptor_recovers_exact_int8_codes_and_sampled_padding():
    x = _fixture()
    packed = quantize(x, CodecConfig(salient_fraction=0.25))
    restored = packed.dequantize()
    source = packed.salient_channels.long()
    donor = packed.donor_channels.long()
    source_tokens = source.unsqueeze(-2).expand(2, 3, 41, 4)
    donor_tokens = donor.unsqueeze(-2).expand_as(source_tokens)
    source_scales = packed.scales.gather(-1, source).unsqueeze(-2)
    expected_sources = (x.gather(-1, source_tokens) / source_scales).round().clamp(-127, 127)
    expected_sources *= source_scales
    torch.testing.assert_close(restored.gather(-1, source_tokens), expected_sources, rtol=0, atol=0)
    expected_padding = x[..., ::8, :].mean(-2).gather(-1, donor)
    torch.testing.assert_close(
        restored.gather(-1, donor_tokens), expected_padding.unsqueeze(-2).expand_as(expected_sources),
        rtol=0, atol=0,
    )
    # Remaining channels obey the standard symmetric INT4 rounding bound.
    normal = torch.ones((2, 3, 16), dtype=torch.bool)
    normal.scatter_(-1, source, False)
    normal.scatter_(-1, donor, False)
    bound = packed.scales.unsqueeze(-2) / 2 + 1e-7
    assert torch.all(((restored - x).abs() <= bound) | ~normal.unsqueeze(-2))


@pytest.mark.parametrize("fraction", [0.0, 0.125, 0.5])
def test_all_zero_is_finite_and_exact(fraction):
    x = torch.zeros(1, 2, 17, 16, dtype=torch.bfloat16)
    packed = quantize(x, CodecConfig(salient_fraction=fraction))
    restored = packed.dequantize()
    assert restored.dtype == x.dtype
    assert torch.isfinite(restored).all()
    assert torch.count_nonzero(restored) == 0
    assert packed.dequantize(torch.float32).dtype == torch.float32


def test_zero_fraction_is_uniform_int4_with_no_donor_storage():
    x = _fixture()
    packed = quantize(x, CodecConfig(salient_fraction=0))
    expected = (x / packed.scales.unsqueeze(-2)).round().clamp(-7, 7) * packed.scales.unsqueeze(-2)
    torch.testing.assert_close(packed.dequantize(), expected, rtol=0, atol=0)
    assert packed.salient_channels.numel() == packed.donor_channels.numel() == packed.padding.numel() == 0
    assert packed.metadata_nbytes == packed.scales.numel() * 4


def test_ordinary_nibble_patterns_never_become_markers():
    # Exercise all signed INT4 codes without any descriptor; the nibble pattern
    # cannot change channel roles. A -8 code is unused by symmetric quantization.
    x = torch.arange(-7, 8, dtype=torch.float32).view(1, 1, 15, 1).expand(1, 1, 15, 16)
    packed = quantize(x, CodecConfig(salient_fraction=0))
    torch.testing.assert_close(packed.dequantize(), x, rtol=0, atol=0)


def test_saliency_uses_the_configured_sample_positions():
    x = torch.ones(1, 1, 16, 8)
    x[..., 1::2, 0] = 1000  # Unsampled outliers must not influence ranking.
    x[..., ::8, 6] = 10
    packed = quantize(x, CodecConfig(sample_stride=8, salient_fraction=0.125))
    assert packed.salient_channels.tolist() == [[[6]]]


def test_noncontiguous_input_is_supported_and_output_detached():
    x = torch.randn(1, 2, 16, 12, requires_grad=True).transpose(-2, -1)
    packed = quantize(x)
    assert packed.dequantize().shape == x.shape
    assert all(not t.requires_grad for t in packed.tensors.values())
    assert torch.isfinite(packed.dequantize()).all()


@pytest.mark.parametrize("channels", [2, 128, 258])
def test_half_salient_fraction_always_has_enough_donors(channels):
    packed = quantize(torch.randn(1, 1, 3, channels), CodecConfig(salient_fraction=0.5))
    assert packed.donor_channels.unique().numel() == channels // 2
    assert torch.isfinite(packed.dequantize()).all()
    if channels > 256:
        assert packed.donor_channels.dtype == torch.int32


@pytest.mark.parametrize("bad", [-0.1, 0.51, float("nan")])
def test_invalid_fraction_rejected(bad):
    with pytest.raises(ValueError, match="salient_fraction"):
        CodecConfig(salient_fraction=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_sample_stride_rejected(bad):
    with pytest.raises(ValueError, match="sample_stride"):
        CodecConfig(sample_stride=bad)


def test_odd_dimension_and_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="even"):
        quantize(torch.ones(1, 1, 8, 7))
    with pytest.raises(ValueError, match="nonempty"):
        quantize(torch.ones(1, 1, 0, 8))
    with pytest.raises(ValueError, match="floating"):
        quantize(torch.ones(1, 1, 8, 8, dtype=torch.int32))
    for bad in [float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            quantize(torch.full((1, 1, 8, 8), bad))


def test_min_error_donor_avoids_destroying_a_varying_neighbor():
    generator = torch.Generator().manual_seed(119)
    x = torch.randn(1, 2, 128, 8, generator=generator)
    x[..., 0] = 100  # The same salient source in both policies.
    x[..., 1] *= 10  # Nearby, but costly to replace with its mean.
    x[..., 7] = 2    # Nonadjacent, exactly reconstructable from the mean.
    local = quantize(x, CodecConfig(salient_fraction=0.125))
    selected = quantize(x, CodecConfig(salient_fraction=0.125, donor_policy="min_error"))
    assert torch.all(local.donor_channels == 1)
    assert torch.all(selected.donor_channels == 7)
    assert torch.equal(selected.salient_channels, local.salient_channels)
    assert selected.nbytes == local.nbytes
    assert ((selected.dequantize() - x) ** 2).mean() < ((local.dequantize() - x) ** 2).mean()
