"""Write codec output directly to physical pages, without a packed staging pool."""
import torch
import triton
import triton.language as tl


@triton.jit
def _write_pages(K, KS, KP, SOURCE, DONOR, V, VS, VO, POOL, IDS,
                 H: tl.constexpr, B: tl.constexpr, PAGE_BYTES: tl.constexpr,
                 OKS: tl.constexpr, OKM: tl.constexpr, OKP: tl.constexpr,
                 OVP: tl.constexpr, OVS: tl.constexpr, OVO: tl.constexpr,
                 TILE: tl.constexpr = 2048):
    p,h,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    physical=tl.load(IDS+p).to(tl.int64)
    page=POOL+physical*PAGE_BYTES
    i=tile*TILE+tl.arange(0,TILE)
    k=tl.load(K+(p*H+h)*B*64+i,mask=i<B*64,other=0)
    tl.store(page+h*B*64+i,k,mask=i<B*64)
    v=tl.load(V+(p*H+h)*B*32+i,mask=i<B*32,other=0)
    tl.store(page+OVP+h*B*32+i,v,mask=i<B*32)
    if tile == 0:
        d=tl.arange(0,128)
        j=tl.arange(0,16)
        source=tl.load(SOURCE+(p*H+h)*16+j).to(tl.int32)
        donor=tl.load(DONOR+(p*H+h)*16+j).to(tl.int32)
        source_role=tl.max(tl.where(d[:,None]==source[None,:],donor[None,:],-1),axis=1)
        donor_role=tl.min(tl.where(d[:,None]==donor[None,:],-2-j[None,:],0),axis=1)
        role=tl.where(donor_role<0,donor_role,source_role)
        tl.store((page+OKM).to(tl.pointer_type(tl.int16))+h*128+d,role)
        scales=tl.load(KS+(p*H+h)*128+d)
        tl.store((page+OKS).to(tl.pointer_type(tl.float32))+h*128+d,scales)
        padding=tl.load(KP+(p*H+h)*16+j)
        tl.store((page+OKP).to(tl.pointer_type(tl.float32))+h*16+j,padding)
        parameter=tl.arange(0,512)
        vs=tl.load(VS+(p*H+h)*B*2+parameter,mask=parameter<B*2,other=0)
        vo=tl.load(VO+(p*H+h)*B*2+parameter,mask=parameter<B*2,other=0)
        tl.store((page+OVS).to(tl.pointer_type(tl.float16))+h*B*2+parameter,vs,mask=parameter<B*2)
        tl.store((page+OVO).to(tl.pointer_type(tl.float16))+h*B*2+parameter,vo,mask=parameter<B*2)


def write_codec_pages(keys, values, layout, pool, page_ids):
    """Caller owns allocation and supplies unique, in-range GPU page IDs."""
    if pool.dtype!=torch.uint8 or pool.ndim!=2 or pool.shape[1]!=layout.page_bytes or not pool.is_contiguous():
        raise ValueError('Expected contiguous uint8 destination page pool')
    if page_ids.ndim!=1 or page_ids.numel()!=keys.shape[0] or page_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('One integer physical page ID is required per input page')
    if page_ids.device!=pool.device or keys.payload.device!=pool.device or values.payload.device!=pool.device:
        raise ValueError('Codec tensors, page IDs and pool must share a CUDA device')
    offsets=layout.offsets
    _write_pages[(page_ids.numel(),layout.heads,triton.cdiv(layout.block_size*64,2048))](
        keys.payload,keys.scales,keys.padding,keys.salient_channels,keys.donor_channels,
        values.payload,values.scales,values.offsets,pool,page_ids.contiguous(),
        layout.heads,layout.block_size,layout.page_bytes,
        offsets['key_scales'],offsets['key_map'],offsets['key_padding'],
        offsets['value_payload'],offsets['value_scales'],offsets['value_offsets'],num_warps=4)
