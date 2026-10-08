import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from inquant.cache import CacheConfig, InQuantCache
from inquant.codec import CodecConfig, quantize
from inquant.value_codec import quantize_values, quantize_value_blocks


@pytest.mark.parametrize("bits", [2, 4])
def test_known_affine_codes_physical_packing_and_zero_range(bits):
    codes = torch.arange(1 << bits).repeat(128 // (1 << bits))
    x = (codes.float() * 2 - 3).view(1, 1, 1, 128).expand(2, 3, 7, 128).clone()
    q = quantize_values(x, bits, 64)
    wanted_byte = sum(i << (bits * i) for i in range(8 // bits))
    assert q.payload[0, 0, 0, 0].item() == wanted_byte
    assert q.payload.dtype == torch.uint8
    assert q.payload_nbytes == x.numel() * bits // 8
    assert q.metadata_nbytes == x.numel() // 64 * 4
    torch.testing.assert_close(q.dequantize(), x, rtol=0, atol=0)
    for constant in (0., 3., -7.):
        y = torch.full((1, 2, 17, 128), constant, dtype=torch.bfloat16)
        torch.testing.assert_close(quantize_values(y, bits).dequantize(), y, rtol=0, atol=0)


def test_compact_batched_blocks_match_individually_packed_noncontiguous_input():
    torch.manual_seed(52)
    x = torch.randn(1, 3, 103, 128, dtype=torch.bfloat16)
    blocks = quantize_value_blocks(x, 3, 32, 2, 64)
    pointers = []
    for i, block in enumerate(blocks):
        expected = quantize_values(x[:, :, i*32:(i+1)*32], 2, 64)
        for name, tensor in block.tensors.items():
            torch.testing.assert_close(tensor, expected.tensors[name], rtol=0, atol=0)
            assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size()
            pointers.append(tensor.data_ptr())
    assert len(set(pointers)) == len(pointers)


def test_value_codec_rejects_unrepresentable_parameters():
    for value in (float('nan'), float('inf'), 1e6):
        with pytest.raises(ValueError, match="finite"):
            quantize_values(torch.full((1, 1, 2, 128), value))
    with pytest.raises(ValueError):
        quantize_values(torch.zeros(1, 1, 2, 48))


def test_k4v2_append_semantics_and_owned_bytes_across_seal_boundary():
    torch.manual_seed(21)
    cfg = CacheConfig(CodecConfig(donor_policy="min_error"), block_size=8,
                      residual_length=3, sink_tokens=2, value_bits=2, track_peak_bytes=True)
    a, b = InQuantCache(cfg), InQuantCache(cfg)
    k, v = [torch.randn(1, 2, 51, 128) for _ in range(2)]
    for cache in (a, b):
        cache.update(k[:, :, :27], v[:, :, :27], 0)
    expected_peak = a.nbytes
    for step in range(27, 51):
        expected = a.update(k[:, :, step:step+1], v[:, :, step:step+1], 0)
        view = b.append_for_attention(k[:, :, step:step+1], v[:, :, step:step+1], 0)
        for wanted, sink, blocks, tail in zip(expected, (view.sink_key, view.sink_value),
                                             (view.key_blocks, view.value_blocks),
                                             (view.residual_key, view.residual_value)):
            actual = torch.cat([sink]+[block.dequantize() for block in blocks]+[tail], dim=2)
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert a.nbytes == b.nbytes
        expected_peak = max(expected_peak, a.nbytes)
        assert a.peak_nbytes == b.peak_nbytes == expected_peak
    stats = a.memory_stats()
    assert stats['nbytes'] == sum(stats[k] for k in
        ['payload_nbytes', 'metadata_nbytes', 'sink_nbytes', 'residual_nbytes', 'auxiliary_nbytes'])


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("bits,group", [(2,32), (2,64), (2,128), (4,64)])
def test_fused_asymmetric_gqa_equals_dequantized_sdpa(bits, group):
    from inquant.triton_attention import PackedDecodeState
    torch.manual_seed(125)
    cfg = CodecConfig(donor_policy="min_error")
    keys = [quantize(torch.randn(1, 4, 256, 128, device='cuda', dtype=torch.bfloat16), cfg) for _ in range(2)]
    vals = [quantize_values(torch.randn(1, 4, 256, 128, device='cuda', dtype=torch.bfloat16), bits, group) for _ in range(2)]
    tail_k, tail_v = [torch.randn(1, 4, 267, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    q = torch.randn(1, 28, 1, 128, device='cuda', dtype=torch.bfloat16)
    state = PackedDecodeState(keys, vals)
    actual = state.decode(q, residual_key=tail_k, residual_value=tail_v)
    k = torch.cat([b.dequantize() for b in keys]+[tail_k],dim=2)
    v = torch.cat([b.dequantize() for b in vals]+[tail_v],dim=2)
    expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.0015)
    assert state._value_map.numel() == 0


@cuda
def test_workspace_pool_is_shared_once_and_rejects_other_stream():
    from inquant.triton_attention import PackedDecodeState, SharedDecodeWorkspace
    pool = SharedDecodeWorkspace()
    q = torch.randn(1, 28, 1, 128, device='cuda', dtype=torch.bfloat16)
    k, v = [torch.randn(1, 4, 19, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    states = [PackedDecodeState([], [], workspace_pool=pool) for _ in range(3)]
    for i, state in enumerate(states):
        actual = state.decode(q, residual_key=k, residual_value=v+i)
        expected = F.scaled_dot_product_attention(q, k, v+i, enable_gqa=True)
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.008)
        assert state._workspace is None
        assert state.nbytes == 0
    assert pool.nbytes == 28*130*4
    first_pointer = pool.tensor.data_ptr()
    states[0].decode(q, residual_key=k, residual_value=v)
    assert first_pointer == pool.tensor.data_ptr()
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="one CUDA stream"):
            states[0].decode(q, residual_key=k, residual_value=v)


@cuda
@pytest.mark.parametrize("value_bits", [None, 2])
def test_qwen_three_layers_shared_workspace_and_cache_reset(value_bits):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from inquant.qwen_fused import enable_fused_qwen2
    torch.manual_seed(87)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=97, hidden_size=512, intermediate_size=1024,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        attn_implementation='sdpa')).eval().to('cuda',torch.bfloat16)
    cfg = CacheConfig(CodecConfig(donor_policy='min_error'),block_size=32,residual_length=4,
                      sink_tokens=2,value_bits=value_bits,shared_workspace=True,track_peak_bytes=True)
    packed = InQuantCache(cfg)
    independent = InQuantCache(replace(cfg, shared_workspace=False))
    dense_ref = InQuantCache(cfg)
    original_model = copy.deepcopy(model)
    tokens = torch.randint(0,97,(1,99),device='cuda')
    with enable_fused_qwen2(model), torch.inference_mode():
        for cache in (packed, independent):
            model(tokens,past_key_values=cache)
        original_model(tokens,past_key_values=dense_ref)
        for step in range(10):
            token = torch.tensor([[step+3]],device='cuda')
            expected = model(token,past_key_values=independent).logits
            oracle = original_model(token,past_key_values=dense_ref).logits
            with patch.object(packed,'materialize',side_effect=AssertionError('dense history not allowed')):
                actual = model(token,past_key_values=packed).logits
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(actual.float(),oracle.float(),rtol=.035,atol=.015)
    per_layer = next(iter(independent._attention_states.values())).kernel._workspace.untyped_storage().nbytes()
    assert independent.nbytes-packed.nbytes == 2*per_layer
    for cache in (packed, independent):
        assert cache._tracked_live_bytes + cache._tracked_aux_bytes + (
            cache._shared_decode_workspace.nbytes if cache._shared_decode_workspace else 0) == cache.nbytes
        assert cache.peak_nbytes >= cache.nbytes
    old_pool = packed._shared_decode_workspace
    packed.reset()
    assert packed.nbytes==0 and packed._shared_decode_workspace is None
    assert old_pool is not InQuantCache(cfg)._shared_decode_workspace
