"""Mistral GQA/RoPE adaptation checked independently of Qwen model classes."""
import copy
from unittest.mock import patch
import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM
from inquant.cache import CacheConfig, InQuantCache
from inquant.codec import CodecConfig
from inquant.qwen_fused import enable_fused_mistral


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_mistral_cache_logits_across_sealing_boundaries(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(673)
    d=128 if device=='cuda' else 32
    cfg=MistralConfig(vocab_size=97,hidden_size=4*d,intermediate_size=8*d,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=32768,sliding_window=None,rope_theta=1e6,
        bos_token_id=1,eos_token_id=96,pad_token_id=0,attn_implementation='sdpa')
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    model=MistralForCausalLM(cfg).to(device=device,dtype=dtype).eval()
    ref=copy.deepcopy(model)
    cachecfg=CacheConfig(CodecConfig(donor_policy='min_error'),block_size=32,residual_length=4,sink_tokens=4,value_bits=2,value_group_size=64 if device=='cuda' else 32)
    original=model.model.layers[0].self_attn.forward
    with enable_fused_mistral(model,allow_cpu_reference=device=='cpu') as handle,torch.no_grad():
        a,b=InQuantCache(cachecfg),InQuantCache(cachecfg)
        prompt=torch.randint(2,90,(1,66),device=device)
        torch.testing.assert_close(model(prompt,past_key_values=a).logits,ref(prompt,past_key_values=b).logits,atol=0,rtol=0)
        for i in range(8):
            token=torch.tensor([[i+5]],device=device)
            expected=ref(token,past_key_values=b).logits
            with patch.object(a,'materialize',side_effect=AssertionError('dense fallback')):
                actual=model(token,past_key_values=a).logits
            torch.testing.assert_close(actual.float(),expected.float(),atol=.012 if device=='cuda' else 2e-6,rtol=.035 if device=='cuda' else 2e-5)
        assert handle.fused_decode_calls+handle.cpu_reference_calls==16
    assert model.model.layers[0].self_attn.forward==original


def test_windowed_mistral_is_not_silently_changed_to_full_attention():
    cfg=MistralConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                      num_attention_heads=2,num_key_value_heads=1,sliding_window=16)
    model=MistralForCausalLM(cfg).eval()
    with pytest.raises(ValueError,match='sliding-window'):
        enable_fused_mistral(model,allow_cpu_reference=True)
