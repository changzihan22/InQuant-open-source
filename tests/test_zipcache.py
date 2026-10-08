"""ZipCache port checked against unmodified official codec and attention math."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from inquant.zipcache import (ZipCache, ZipCacheConfig, ZipPackedTensor, important_complement,
                             probe_attention_scores, enable_zipcache_qwen2, official_codec)


def upstream_union():
    path = Path(__file__).parents[1] / "third_party/ZipCache/zipcache/models/CompressUtils"
    spec = importlib.util.spec_from_file_location("_zipcache_test_upstream", path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MixedPrecisionCompressUnion


def test_chunked_selector_matches_full_upstream_gqa_formula():
    torch.manual_seed(821)
    q, k = torch.randn(1, 6, 41, 8), torch.randn(1, 2, 41, 8)
    probes = torch.tensor([39, 40, 2, 7])
    scores = q[:, :, probes] @ k.repeat_interleave(3, dim=1).transpose(-1, -2) / 8**.5
    mask = torch.arange(41)[None, :] > probes[:, None]
    scores.masked_fill_(mask[None, None], torch.finfo(q.dtype).min)
    expected = scores.softmax(-1).reshape(1, 2, 3, len(probes), 41).sum(2).sum(2)
    expected /= torch.arange(41, 0, -1)
    actual = probe_attention_scores(q, k, probes, 2, causal=True, normalize=True)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual.topk(16, largest=False).indices, expected.topk(16, largest=False).indices)


@pytest.mark.parametrize("kind,mode", [("key", "mixed_channelwiseQ"), ("value", "channel_separate_mixed_tokenwiseQ")])
def test_packed_storage_exactly_matches_official_streaming_union(kind, mode):
    torch.manual_seed(822)
    cfg = ZipCacheConfig(streaming_gap=4)
    union = upstream_union()({"compress_mode": mode, "quantize_bit_important": 4,
                              "quantize_bit_unimportant": 2, "stream": True, "streaming_gap": 4})
    x = torch.randn(1, 2, 30, 8)
    low = torch.stack((torch.arange(0, 12), torch.arange(12, 24)))[None]
    packed = ZipPackedTensor.pack(x, low, kind, cfg)
    union.compress(x, low)
    torch.testing.assert_close(packed.dequantize(), union.decompress(), rtol=0, atol=0)
    assert packed.parts[0].dtype == packed.parts[3].dtype == torch.uint8
    assert sum(t.numel() for t in (packed.parts[0], packed.parts[3])) == (18 * 2 * 8 // 2 + 12 * 2 * 8 // 4)
    for step in range(1, 6):
        token = torch.randn(1, 2, 1, 8)
        dense = torch.cat((x if step == 1 else expected, token), dim=2)
        if step == 1:
            dense = torch.cat((packed.dequantize(), token), dim=2)
        if step == 4:
            low = torch.cat((low, torch.full((1, 2, 1), 30, dtype=torch.long)), dim=2)
            packed = ZipPackedTensor.pack(dense, low, kind, cfg)
            tail = None
        else:
            tail = dense[..., packed.shape[-2]:, :].clone()
        union.compress(dense, low)
        expected = union.decompress()
        actual = packed.dequantize() if tail is None else torch.cat((packed.dequantize(), tail), dim=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cache_recompression_tail_accounting_and_seed_replay():
    torch.manual_seed(823)
    cfg = ZipCacheConfig(streaming_gap=4)
    a, b = ZipCache(cfg, seed=15), ZipCache(cfg, seed=15)
    observed = []
    for length in (30, 1, 1, 1, 1, 1):
        q, k, v = torch.randn(1, 6, length, 8), torch.randn(1, 2, length, 8), torch.randn(1, 2, length, 8)
        actual, expected = a.append(q, k, v, 0), b.append(q, k, v, 0)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        stats = a.memory_stats()
        observed.append(a.nbytes)
        assert a.nbytes == stats['payload_nbytes'] + stats['metadata_nbytes'] + stats['residual_nbytes']
        assert a.live_nbytes == a.nbytes
        assert a.peak_nbytes == max(observed)
        assert stats['max_tail_tokens'] < 4
    assert a.get_seq_length() == 35 and a.compressions == 2 and a.decode_calls == 5
    assert a.layers_by_id[0].key.unimportant_ids.shape[-1] == 13


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_qwen_prefill_identical_and_manual_quantized_decode_oracle(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
    torch.manual_seed(824)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa")).to(device=device, dtype=dtype).eval()
    reference = copy.deepcopy(model)
    tokens = torch.randint(2, 90, (1, 31), device=device)
    cache = ZipCache(ZipCacheConfig(streaming_gap=4))
    with torch.inference_mode(), enable_zipcache_qwen2(model):
        got = model(tokens, past_key_values=cache).logits
        expected = reference(tokens).logits
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for step in range(6):
            layer = cache.layers_by_id[0]
            dense = DynamicCache()
            k, v = layer.key.dequantize(), layer.value.dequantize()
            if layer.tail_key is not None:
                k, v = torch.cat((k, layer.tail_key), 2), torch.cat((v, layer.tail_value), 2)
            dense.update(k, v, 0)
            token = torch.tensor([[3 + step]], device=device)
            expected = reference(token, past_key_values=dense).logits
            got = model(token, past_key_values=cache).logits
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        assert cache.compressions == 2


def test_nonfinite_upstream_value_scales_are_not_silently_accepted():
    x = torch.zeros(1, 2, 30, 8)
    ids = torch.arange(12)[None, None].expand(1, 2, -1)
    with pytest.raises(ValueError, match="nonfinite"):
        ZipPackedTensor.pack(x, ids, "value", ZipCacheConfig())
