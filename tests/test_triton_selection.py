"""Fused selection must match stable reference assignments and every packed field."""
import pytest
import torch
from inquant.codec import CodecConfig, quantize, _reuse_map

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('count', [0, 1, 8, 16, 32, 64])
@pytest.mark.parametrize('policy', ['neighbors','min_error'])
def test_selector_matches_reference_with_ties_and_adjacent_outliers(count, policy):
    from inquant.triton_selection import reuse_map
    torch.manual_seed(112)
    score = torch.rand(3, 4, 128, device='cuda')
    score[0] = 0
    score[1, :, 30:70] = 5
    score[2, :, 127] = 9
    error = torch.randint(0, 4, score.shape, device='cuda').float() if policy == 'min_error' else None
    expected = _reuse_map(score, count, error)
    actual = reuse_map(score, count, error)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.parametrize('policy', ['neighbors','min_error'])
@pytest.mark.parametrize('tokens',[32,64,256])
def test_packed_fields_are_bitwise_identical(policy,tokens):
    torch.manual_seed(513)
    x=torch.randn(3,4,tokens,128,device='cuda',dtype=torch.bfloat16)
    x[0,0] = 0
    x[..., 32:48] *= 20
    a=quantize(x,CodecConfig(donor_policy=policy,donor_backend='torch'))
    b=quantize(x,CodecConfig(donor_policy=policy,donor_backend='triton'))
    for name in a.tensors:
        torch.testing.assert_close(a.tensors[name],b.tensors[name],atol=0,rtol=0)
    torch.testing.assert_close(a.dequantize(),b.dequantize(),atol=0,rtol=0)
