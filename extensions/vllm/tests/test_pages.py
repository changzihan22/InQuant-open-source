"""Run with the extension environment; CUDA cases require a real device."""
import pytest
import torch
from inquant.codec import quantize, CodecConfig
from inquant.value_codec import quantize_values
from inquant_vllm.layout import PageLayout, pack_pages, unpack_pages_reference


@pytest.mark.parametrize("block", [64, 128, 256])
def test_page_layout_matches_existing_codecs(block):
    torch.manual_seed(7)
    layout = PageLayout(block, 2)
    k, v = torch.randn(2, 3, 2, block, 128, dtype=torch.bfloat16)
    pages = pack_pages(k, v, layout)
    assert pages.numel() == 3 * layout.page_bytes
    restored_k, restored_v = unpack_pages_reference(pages, layout)
    torch.testing.assert_close(restored_k, quantize(k, CodecConfig(donor_policy="min_error")).dequantize(), rtol=0, atol=0)
    torch.testing.assert_close(restored_v, quantize_values(v).dequantize(), rtol=0, atol=0)
    assert layout.page_bytes < block * 2 * 128 * 4 * 0.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block,pages,tail", [(64, 0, 1), (64, 0, 63), (64, 1, 0),
                                               (64, 1, 64), (128, 3, 17), (256, 128, 128)])
def test_paged_decode_random_page_order_and_reuse(block, pages, tail):
    from inquant.triton_attention import SharedDecodeWorkspace
    from inquant_vllm.kernels import paged_decode
    torch.manual_seed(71)
    device, h, hq = 'cuda', 4, 28
    layout = PageLayout(block, h)
    pool = torch.full((pages + 7, layout.page_bytes), 255, dtype=torch.uint8, device=device)
    ids = torch.randperm(pages + 7, device=device)[:pages].to(torch.int32)
    dense = torch.full((2, h, block + 4, 128), float('nan'), dtype=torch.bfloat16, device=device)
    for reuse in range(2):
        k, v = torch.randn(2, h, pages * block + tail, 128, device=device, dtype=torch.bfloat16)
        sink = min(4, k.shape[1])
        dense[0, :, :sink], dense[1, :, :sink] = k[:, :sink], v[:, :sink]
        if tail:
            dense[0, :, 4:4 + tail], dense[1, :, 4:4 + tail] = k[:, -tail:], v[:, -tail:]
        if pages:
            kt = k[:, :pages * block].reshape(h, pages, block, 128).transpose(0, 1)
            vt = v[:, :pages * block].reshape(h, pages, block, 128).transpose(0, 1)
            packed = pack_pages(kt, vt, layout)
            pool.index_copy_(0, ids.long(), packed)
            kr, vr = unpack_pages_reference(packed, layout)
            kr = kr.transpose(0, 1).reshape(h, -1, 128)
            vr = vr.transpose(0, 1).reshape(h, -1, 128)
            kr[:, :sink], vr[:, :sink] = k[:, :sink], v[:, :sink]
            kr, vr = torch.cat((kr, k[:, pages * block:]), 1), torch.cat((vr, v[:, pages * block:]), 1)
        else:
            kr, vr = k, v
        q = torch.randn(1, hq, 128, device=device, dtype=torch.bfloat16)
        actual = paged_decode(q, pool, ids, dense, pages, tail, sink, layout, SharedDecodeWorkspace(), 128 ** -0.5)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), kr.unsqueeze(0), vr.unsqueeze(0), enable_gqa=True)
        expected = expected[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        assert torch.isfinite(actual).all()
