"""Direct writes preserve every page byte and leave unassigned pool pages intact."""
import pytest
import torch
from inquant_vllm.layout import PageLayout, pack_pages, pack_pages_into

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('block',[64,128,256])
@pytest.mark.parametrize('backend',['torch','triton'])
@pytest.mark.parametrize('heads',[4,8])
def test_direct_scattered_pages_match_reference_and_reuse(block,backend,heads):
    torch.manual_seed(915)
    layout=PageLayout(block,heads)
    pool=torch.full((7,layout.page_bytes),173,device='cuda',dtype=torch.uint8)
    ids=torch.tensor([5,0,3],device='cuda',dtype=torch.int32)
    for _ in range(2):
        k,v=torch.randn(2,3,heads,block,128,device='cuda',dtype=torch.bfloat16)
        reference=pack_pages(k,v,layout)
        pack_pages_into(k,v,layout,pool,ids,donor_backend=backend)
        torch.testing.assert_close(pool[ids.long()],reference,rtol=0,atol=0)
        assert (pool[torch.tensor([1,2,4,6],device='cuda')]==173).all()


def test_page_write_and_decode_beyond_signed_32bit_byte_offset():
    from inquant.triton_attention import SharedDecodeWorkspace
    from inquant_vllm.kernels import paged_decode
    layout = PageLayout(64, 4)
    far = 2**31 // layout.page_bytes + 1
    needed = (far + 1) * layout.page_bytes
    if torch.cuda.mem_get_info()[0] < needed + 512 * 1024**2:
        pytest.skip('Requires approximately 2.6 GiB of free GPU memory')
    pool = torch.empty((far + 1, layout.page_bytes), device='cuda', dtype=torch.uint8)
    ids = torch.tensor([0, far], device='cuda', dtype=torch.int32)
    torch.manual_seed(811)
    k, v = torch.randn(2, 1, 4, 64, 128, device='cuda', dtype=torch.bfloat16)
    reference = pack_pages(k, v, layout)
    pack_pages_into(k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1), layout, pool, ids,
                    donor_backend='triton')
    torch.testing.assert_close(pool[ids.long()], reference.expand(2, -1), atol=0, rtol=0)
    query = torch.randn(1, 28, 128, device='cuda', dtype=torch.bfloat16)
    tail = torch.randn(2, 4, 68, 128, device='cuda', dtype=torch.bfloat16)
    workspace = SharedDecodeWorkspace()
    near_output = paged_decode(query, pool, ids[:1], tail, 1, 1, 4, layout, workspace, 128**-.5)
    far_output = paged_decode(query, pool, ids[1:], tail, 1, 1, 4, layout, workspace, 128**-.5)
    torch.testing.assert_close(near_output, far_output, atol=0, rtol=0)
