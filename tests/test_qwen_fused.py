import copy
import unittest
from unittest.mock import patch

import torch

from inquant.cache import CacheConfig, InQuantCache
from inquant.codec import CodecConfig
from inquant.qwen_fused import enable_fused_qwen2

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    HAVE_QWEN = True
except ImportError:
    HAVE_QWEN = False


def _dense_view(view, dtype):
    def restore(sink, blocks, tail):
        chunks = [] if sink is None else [sink]
        chunks.extend(block.dequantize(dtype=dtype) for block in blocks)
        chunks.append(tail)
        return torch.cat(chunks, dim=2)
    return (
        restore(view.sink_key, view.key_blocks, view.residual_key),
        restore(view.sink_value, view.value_blocks, view.residual_value),
    )


class PackedAppendTests(unittest.TestCase):
    def test_append_snapshot_matches_reference_across_block_boundary(self):
        torch.manual_seed(27)
        config = CacheConfig(CodecConfig(salient_fraction=.125), block_size=8, residual_length=2, sink_tokens=2)
        reference, packed = InQuantCache(config), InQuantCache(config)
        key, value = torch.randn(1, 2, 32, 16), torch.randn(1, 2, 32, 16)
        reference.update(key[:, :, :19], value[:, :, :19], 0)
        packed.update(key[:, :, :19], value[:, :, :19], 0)
        for token in range(19, 32):
            expected = reference.update(key[:, :, token:token + 1], value[:, :, token:token + 1], 0)
            with patch.object(packed, "materialize", side_effect=AssertionError("Unexpected full materialization")):
                view = packed.append_for_attention(key[:, :, token:token + 1], value[:, :, token:token + 1], 0)
            for actual, wanted in zip(_dense_view(view, key.dtype), expected):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            for actual, wanted in zip(packed.materialize(), reference.materialize()):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            self.assertEqual(packed.nbytes, reference.nbytes)
            self.assertEqual(view.seq_length, token + 1)

    def test_append_while_sink_is_still_filling(self):
        config = CacheConfig(sink_tokens=4, block_size=8, residual_length=2)
        reference, packed = InQuantCache(config), InQuantCache(config)
        x = torch.randn(1, 2, 9, 8)
        for cache in (reference, packed):
            cache.update(x[:, :, :1], x[:, :, :1], 0)
        for index in range(1, 9):
            expected = reference.update(x[:, :, index:index + 1], x[:, :, index:index + 1], 0)
            view = packed.append_for_attention(x[:, :, index:index + 1], x[:, :, index:index + 1], 0)
            for actual, wanted in zip(_dense_view(view, x.dtype), expected):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@unittest.skipUnless(HAVE_QWEN, "Transformers Qwen2 is unavailable")
class FusedQwenTests(unittest.TestCase):
    def _model(self, head_dim=8, layers=2):
        return Qwen2ForCausalLM(Qwen2Config(
            vocab_size=97, hidden_size=4 * head_dim, intermediate_size=8 * head_dim,
            num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=65536, bos_token_id=1, eos_token_id=96, pad_token_id=0,
            attn_implementation="sdpa",
        )).eval()

    def test_cpu_instance_patch_rope_gqa_and_decode_equivalence(self):
        torch.manual_seed(31)
        model = self._model()
        reference_model = copy.deepcopy(model)
        config = CacheConfig(CodecConfig(salient_fraction=.125), block_size=4, residual_length=2, sink_tokens=2)
        reference, packed = InQuantCache(config), InQuantCache(config)
        tokens = torch.randint(2, 90, (1, 19))
        original_forward = model.model.layers[0].self_attn.forward
        handle = enable_fused_qwen2(model, allow_cpu_reference=True)
        self.assertFalse(hasattr(reference_model.model.layers[0].self_attn, "_inquant_original_forward"))
        with torch.no_grad():
            expected = reference_model(tokens, past_key_values=reference).logits
            actual = model(tokens, past_key_values=packed).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for step in range(7):
                token = torch.tensor([[3 + step]])
                expected = reference_model(token, past_key_values=reference).logits
                with patch.object(packed, "materialize", side_effect=AssertionError("Unexpected full materialization")):
                    actual = model(token, past_key_values=packed).logits
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                self.assertEqual(packed.get_seq_length(), reference.get_seq_length())
            self.assertEqual(handle.cpu_reference_calls, 14)
            self.assertEqual(handle.fused_decode_calls, 0)
            generated = model.generate(
                tokens, attention_mask=torch.ones_like(tokens), past_key_values=InQuantCache(config),
                do_sample=False, num_beams=1, max_new_tokens=3, min_new_tokens=3,
            )
            self.assertEqual(tuple(generated.shape), (1, 22))
        with self.assertRaises(ValueError):
            enable_fused_qwen2(model, allow_cpu_reference=True)
        handle.restore()
        handle.restore()
        self.assertEqual(model.model.layers[0].self_attn.forward, original_forward)

    def test_default_does_not_silently_use_cpu_reference(self):
        model = self._model(head_dim=128, layers=1)
        cache = InQuantCache()
        with enable_fused_qwen2(model), torch.no_grad():
            model(torch.tensor([[2, 3, 4]]), past_key_values=cache)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                model(torch.tensor([[5]]), past_key_values=cache)
            self.assertEqual(cache.get_seq_length(), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real fused kernel integration")
    def test_gpu_packed_kernel_logits_and_auxiliary_accounting(self):
        torch.manual_seed(37)
        model = self._model(head_dim=128, layers=1).to(device="cuda", dtype=torch.bfloat16)
        reference_model = copy.deepcopy(model)
        config = CacheConfig(CodecConfig(salient_fraction=.125), block_size=32, residual_length=4, sink_tokens=2)
        # Three tokens cover the dense-only kernel specialization; 98 tokens
        # cover batched prefill quantization and a later sealing boundary.
        for prefill_tokens in (3, 98):
            with self.subTest(prefill_tokens=prefill_tokens), enable_fused_qwen2(model) as handle, torch.no_grad():
                reference, packed = InQuantCache(config), InQuantCache(config)
                tokens = torch.randint(2, 90, (1, prefill_tokens), device="cuda")
                expected = reference_model(tokens, past_key_values=reference).logits
                actual = model(tokens, past_key_values=packed).logits
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for step in range(7):
                    token = torch.tensor([[3 + step]], device="cuda")
                    expected = reference_model(token, past_key_values=reference).logits
                    with patch.object(packed, "materialize", side_effect=AssertionError("Unexpected full materialization")):
                        actual = model(token, past_key_values=packed).logits
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=.035, atol=.012)
                self.assertEqual(handle.fused_decode_calls, 7)
                self.assertEqual(handle.cpu_reference_calls, 0)
                self.assertGreater(packed.auxiliary_nbytes, 0)
                stats = packed.memory_stats()
                self.assertEqual(stats["nbytes"], sum(stats[name] for name in (
                    "payload_nbytes", "metadata_nbytes", "residual_nbytes", "sink_nbytes", "auxiliary_nbytes"
                )))
                if prefill_tokens == 3:
                    self.assertEqual(stats["quantized_layer_tokens"], 0)
                packed.release()
                self.assertEqual(packed.auxiliary_nbytes, 0)


if __name__ == "__main__":
    unittest.main()
