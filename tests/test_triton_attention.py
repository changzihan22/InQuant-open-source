"""CUDA/reference equivalence; skipped honestly when no CUDA device is available."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")

from inquant.codec import CodecConfig, quantize
from inquant.triton_attention import PackedDecodeState, packed_decode, _attention_partials


def _random(shape, dtype):
    return torch.randn(shape, device="cuda", dtype=dtype)


def _reference(query, keys, values, groups):
    key = torch.cat(keys, dim=2).repeat_interleave(groups, dim=1)
    value = torch.cat(values, dim=2).repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(query, key, value, is_causal=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fraction", [0.0, 0.125, 0.5])
@pytest.mark.parametrize("donor_policy", ["neighbors", "min_error"])
def test_packed_gqa_with_sink_and_unsealed_tail(dtype, fraction, donor_policy):
    torch.manual_seed(921)
    cfg = CodecConfig(salient_fraction=fraction, donor_policy=donor_policy)
    keys = [quantize(_random((1, 4, 256, 128), dtype), cfg) for _ in range(3)]
    values = [quantize(_random((1, 4, 256, 128), dtype), cfg) for _ in range(3)]
    sink_k, sink_v = [_random((1, 4, 4, 128), dtype) for _ in range(2)]
    tail_k, tail_v = [_random((1, 4, 281, 128), dtype) for _ in range(2)]
    query = _random((1, 28, 1, 128), dtype)
    state = PackedDecodeState(keys, values)
    descriptor_bytes = state.nbytes
    actual = state.decode(query, sink_key=sink_k, sink_value=sink_v, residual_key=tail_k, residual_value=tail_v)
    expected = _reference(query, [sink_k] + [k.dequantize() for k in keys] + [tail_k],
                          [sink_v] + [v.dequantize() for v in values] + [tail_v], 7)
    tolerance = 1e-3 if dtype != torch.float32 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.02)
    assert torch.isfinite(actual).all()
    assert actual.shape == query.shape and actual.dtype == dtype
    assert state.nbytes == descriptor_bytes + state._workspace.numel() * 4
    pointer = state._workspace.data_ptr()
    state.decode(query, sink_key=sink_k, sink_value=sink_v, residual_key=tail_k, residual_value=tail_v)
    assert state._workspace.data_ptr() == pointer


@pytest.mark.parametrize("tokens", [1, 4, 257])
def test_dense_only_before_any_block_is_sealed(tokens):
    query = _random((1, 14, 1, 128), torch.bfloat16)
    key, value = [_random((1, 2, tokens, 128), torch.bfloat16) for _ in range(2)]
    actual = packed_decode(query, [], [], residual_key=key, residual_value=value)
    expected = _reference(query, [key], [value], 7)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)


def test_no_dense_tensors_and_nondefault_packed_block_length():
    cfg = CodecConfig(salient_fraction=0.125)
    keys, values = [[quantize(_random((1, 1, 41, 128), torch.float16), cfg)] for _ in range(2)]
    query = _random((1, 3, 1, 128), torch.float16)
    actual = packed_decode(query, keys, values)
    expected = _reference(query, [keys[0].dequantize()], [values[0].dequantize()], 3)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.02)


def test_extreme_softmax_and_zero_blocks_remain_finite():
    key = torch.zeros((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    key[..., 0, :] = 30
    value = _random(key.shape, key.dtype)
    query = torch.full((1, 7, 1, 128), 20, device="cuda", dtype=torch.bfloat16)
    keys, values = [quantize(key)], [quantize(value)]
    actual = packed_decode(query, keys, values)
    expected = _reference(query, [keys[0].dequantize()], [values[0].dequantize()], 7)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.02)


def test_growing_tail_reuses_compiled_kernel_for_64_decode_steps():
    # Real cache tails are compact allocations: their head stride changes at
    # every appended token. Those strides must be runtime kernel arguments.
    query = _random((1, 28, 1, 128), torch.bfloat16)
    key = quantize(_random((1, 4, 256, 128), torch.bfloat16))
    value = quantize(_random((1, 4, 256, 128), torch.bfloat16))
    state = PackedDecodeState([key], [value])
    tail_k, tail_v = [_random((1, 4, 192, 128), torch.bfloat16) for _ in range(2)]
    device = query.device.index
    before = len(_attention_partials.device_caches[device][0])
    for length in range(129, 193):
        actual = state.decode(query, residual_key=tail_k[:, :, :length].contiguous(),
                              residual_value=tail_v[:, :, :length].contiguous())
    torch.cuda.synchronize()
    after = len(_attention_partials.device_caches[device][0])
    # Triton may distinguish scalar alignment classes, but never each length.
    assert after - before <= 3
    expected = _reference(query, [key.dequantize(), tail_k], [value.dequantize(), tail_v], 7)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.02)
