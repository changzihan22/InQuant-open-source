import unittest
from unittest.mock import patch

import torch

from inquant.cache import CacheConfig, InQuantCache
from inquant.codec import CodecConfig, quantize


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = CacheConfig(CodecConfig(salient_fraction=0), block_size=8, residual_length=3, sink_tokens=2)

    def test_prefill_exact_small_tail_storage_and_accounting(self):
        cache = InQuantCache(self.config)
        key, value = torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 8)
        returned_key, returned_value = cache.update(key, value, 0)
        self.assertIs(returned_key, key)
        self.assertIs(returned_value, value)
        layer = cache._layers[0]
        for item in (layer.sink_key, layer.sink_value, layer.residual_key, layer.residual_value):
            self.assertEqual(item.untyped_storage().nbytes(), item.numel() * item.element_size())
            self.assertNotEqual(item.untyped_storage().data_ptr(), key.untyped_storage().data_ptr())
        packed_bytes = sum(block.nbytes for block in layer.key_blocks + layer.value_blocks)
        self.assertEqual(cache.nbytes, packed_bytes + cache.sink_nbytes + cache.residual_nbytes)
        self.assertEqual(cache.bf16_nbytes, 2 * key.numel() * 2)
        self.assertEqual(cache.memory_stats()["quantized_layer_tokens"], 32)
        self.assertEqual(cache.get_seq_length(), 37)
        self.assertEqual(cache.get_usable_length(2), 37)
        self.assertIsNone(cache.get_max_cache_shape())

    def test_multihead_append_chunking_and_old_blocks_immutable(self):
        cache = InQuantCache(self.config)
        key, value = torch.randn(1, 3, 61, 8), torch.randn(1, 3, 61, 8)
        cache.update(key[:, :, :21], value[:, :, :21], 0)
        old_block = cache._layers[0].key_blocks[0]
        old_decoded = old_block.dequantize().clone()
        seen = 21
        for count in (1, 7, 12, 20):
            prior_key, prior_value = cache.materialize()
            out_key, out_value = cache.update(key[:, :, seen:seen + count], value[:, :, seen:seen + count], 0)
            torch.testing.assert_close(out_key, torch.cat((prior_key, key[:, :, seen:seen + count]), dim=2))
            torch.testing.assert_close(out_value, torch.cat((prior_value, value[:, :, seen:seen + count]), dim=2))
            seen += count
        layer = cache._layers[0]
        self.assertIs(layer.key_blocks[0], old_block)
        torch.testing.assert_close(old_block.dequantize(), old_decoded, rtol=0, atol=0)
        for original, actual in zip((key, value), cache.materialize()):
            blocks = (seen - self.config.sink_tokens - self.config.residual_length) // self.config.block_size
            pieces = [original[:, :, :2]]
            for index in range(blocks):
                start = 2 + index * 8
                pieces.append(quantize(original[:, :, start:start + 8], self.config.codec).dequantize(dtype=original.dtype))
            pieces.append(original[:, :, 2 + blocks * 8:])
            torch.testing.assert_close(actual, torch.cat(pieces, dim=2), rtol=0, atol=0)
        self.assertEqual(seen, 61)
        self.assertEqual(cache.get_seq_length(), 61)
        self.assertLess(layer.residual_key.shape[2], self.config.residual_length + self.config.block_size)

    def test_sink_accumulates_across_short_prefills(self):
        cache = InQuantCache(self.config)
        x = torch.randn(1, 2, 7, 8)
        for pos in range(7):
            result, _ = cache.update(x[:, :, pos:pos + 1], x[:, :, pos:pos + 1], 0)
            torch.testing.assert_close(result, x[:, :, :pos + 1], rtol=0, atol=0)
        torch.testing.assert_close(cache.materialize()[0], x, rtol=0, atol=0)

    def test_batched_prefill_matches_serial_blocks_and_owns_compact_storage(self):
        # Different heads and temporal blocks must retain independent saliency,
        # scale and donor statistics after moving the block axis into batch.
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                config = CacheConfig(CodecConfig(sample_stride=3, salient_fraction=.25),
                                     block_size=8, residual_length=3, sink_tokens=2)
                key = torch.randn(1, 3, 16, 55, dtype=dtype).transpose(2, 3)
                value = torch.randn(1, 3, 16, 55, dtype=dtype).transpose(2, 3)
                self.assertFalse(key.is_contiguous())
                for index in range(6):
                    key[:, index % 3, 2 + index * 8:10 + index * 8, index] *= 20
                    value[:, (index + 1) % 3, 2 + index * 8:10 + index * 8, 15 - index] *= 17
                cache = InQuantCache(config)
                with patch("inquant.cache.quantize", wraps=quantize) as observed:
                    cache.update(key, value, 0)
                self.assertEqual(observed.call_count, 2)
                self.assertEqual(tuple(observed.call_args_list[0].args[0].shape), (6, 3, 8, 16))
                layer = cache._layers[0]
                expected_bytes = 0
                owned_addresses = set()
                for original, blocks in ((key, layer.key_blocks), (value, layer.value_blocks)):
                    self.assertEqual(len(blocks), 6)
                    for index, actual in enumerate(blocks):
                        start = 2 + index * 8
                        serial = quantize(original[:, :, start:start + 8], config.codec)
                        self.assertEqual(actual.shape, serial.shape)
                        self.assertEqual(actual.original_dtype, dtype)
                        self.assertEqual(actual.nbytes, serial.nbytes)
                        expected_bytes += serial.nbytes
                        for name, tensor in actual.tensors.items():
                            torch.testing.assert_close(tensor, serial.tensors[name], rtol=0, atol=0)
                            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
                            self.assertTrue(tensor.is_contiguous())
                            if tensor.numel():
                                address = tensor.untyped_storage().data_ptr()
                                self.assertNotIn(address, owned_addresses)
                                owned_addresses.add(address)
                        torch.testing.assert_close(actual.dequantize(), serial.dequantize(), rtol=0, atol=0)
                self.assertEqual(cache.nbytes, expected_bytes + cache.sink_nbytes + cache.residual_nbytes)

    def test_positions_layers_reset_and_unsupported_operations(self):
        cache = InQuantCache(self.config)
        x = torch.randn(1, 2, 7, 8)
        cache.update(x, x, 0, {"cache_position": torch.arange(7)})
        cache.update(x, x, 1, {"cache_position": torch.arange(7)})
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.seen_tokens, 7)
        self.assertEqual(cache.get_seq_length(5), 0)
        self.assertEqual(cache.get_mask_sizes(torch.arange(7, 9), 1), (9, 0))
        with self.assertRaises(ValueError):
            cache.update(x, x, 0, {"cache_position": torch.arange(7)})
        self.assertEqual(cache.get_seq_length(), 7)
        with self.assertRaises(NotImplementedError):
            cache.update(x.expand(2, -1, -1, -1), x.expand(2, -1, -1, -1), 0)
        with self.assertRaises(NotImplementedError):
            cache.reorder_cache(torch.tensor([0]))
        with self.assertRaises(NotImplementedError):
            cache.crop(3)
        with self.assertRaises(ValueError):
            cache.update(x[:, :, :, :7], x[:, :, :, :7], 2)
        self.assertEqual(len(cache), 2)
        cache.release()
        self.assertEqual(cache.nbytes, 0)
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertEqual(len(cache), 0)

    def test_controlled_position_validation_opt_out_keeps_shape_checks(self):
        cache = InQuantCache(CacheConfig(validate_positions=False))
        x = torch.randn(1, 2, 3, 8)
        with patch("inquant.cache.torch.equal", side_effect=AssertionError("Unexpected value synchronization")):
            cache.update(x, x, 0, {"cache_position": torch.arange(3)})
            cache.append_for_attention(x[:, :, :1], x[:, :, :1], 0, {"cache_position": torch.tensor([3])})
        self.assertEqual(cache.get_seq_length(), 4)
        for bad_position in (torch.tensor([[4]]), torch.tensor([4, 5]), torch.tensor([4.0]), [4]):
            with self.subTest(position=bad_position), self.assertRaises(ValueError):
                cache.append_for_attention(x[:, :, :1], x[:, :, :1], 0, {"cache_position": bad_position})
            self.assertEqual(cache.get_seq_length(), 4)
        with self.assertRaises(ValueError):
            CacheConfig(validate_positions="false")

    def test_over_32k_cache_shape_and_persistent_bytes(self):
        cache = InQuantCache(CacheConfig(CodecConfig(salient_fraction=0), block_size=256, residual_length=128, sink_tokens=4))
        x = torch.randn(1, 2, 32768, 8, dtype=torch.bfloat16)
        cache.update(x, x, 0)
        tail = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
        out, _ = cache.update(tail, tail, 0)
        self.assertEqual(tuple(out.shape), (1, 2, 32771, 8))
        self.assertEqual(cache.get_seq_length(), 32771)
        self.assertLess(cache.nbytes, cache.bf16_nbytes * 0.8)
        self.assertGreater(cache.metadata_nbytes, 0)

    def test_tiny_qwen2_forward_and_greedy_generation(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest("Transformers/Qwen2 unavailable")
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=65536, bos_token_id=1, eos_token_id=96, pad_token_id=0,
            attn_implementation="sdpa",
        )).eval()
        tokens = torch.randint(2, 90, (1, 19))
        cache = InQuantCache(self.config)
        with torch.no_grad():
            reference = model(tokens, use_cache=False).logits
            actual = model(tokens, past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            self.assertEqual(cache.get_seq_length(), 19)
            output = model(torch.tensor([[3]]), past_key_values=cache, use_cache=True)
            self.assertEqual(tuple(output.logits.shape), (1, 1, 97))
            self.assertEqual(cache.get_seq_length(1), 20)
            self.assertTrue(torch.isfinite(output.logits).all())
            generated = model.generate(
                tokens, attention_mask=torch.ones_like(tokens),
                past_key_values=InQuantCache(self.config), do_sample=False,
                num_beams=1, max_new_tokens=3, min_new_tokens=3,
            )
            self.assertEqual(tuple(generated.shape), (1, 22))

            # With quantization delayed, chunking must preserve the exact causal
            # attention semantics, absolute positions and GQA head grouping.
            exact_cache = InQuantCache(CacheConfig(residual_length=128))
            model(tokens[:, :7], past_key_values=exact_cache, use_cache=True)
            chunked = model(tokens[:, 7:], past_key_values=exact_cache, use_cache=True).logits
            torch.testing.assert_close(chunked, reference[:, 7:], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
