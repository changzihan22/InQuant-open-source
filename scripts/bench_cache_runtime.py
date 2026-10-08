#!/usr/bin/env python3
"""Profile current K4/V2 setup and decode; synthetic, not model accuracy evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, nargs='+', default=[2048, 8192, 32768])
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--donor-backend', choices=['torch','triton'], default='torch')
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--query-heads', type=int, default=28)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root / 'src'))
    import torch
    from inquant.codec import CodecConfig, quantize, _reuse_map
    from inquant.value_codec import quantize_values, quantize_value_blocks
    from inquant.cache import _quantize_temporal_blocks
    from inquant.triton_attention import PackedDecodeState
    torch.set_grad_enabled(False)
    torch.manual_seed(20260925)
    report = {'scope': 'Synthetic one-layer K4/V2 runtime profiling; not model E2E',
              'source_root': str(args.source_root.resolve()), 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'source_sha256': {}, 'workloads': []}
    for name in ['codec.py', 'value_codec.py', 'cache.py', 'triton_attention.py']:
        report['source_sha256'][name] = hashlib.sha256((args.source_root/'src/inquant'/name).read_bytes()).hexdigest()

    def timed(fn):
        for _ in range(2):
            result = fn()
        del result
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        walls, events = [], []
        for _ in range(args.repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter(); a.record(); result = fn(); b.record(); b.synchronize()
            walls.append(1000*(time.perf_counter()-t)); events.append(a.elapsed_time(b)); del result
        return {'wall_ms': statistics.median(walls), 'cuda_ms': statistics.median(events),
                'wall_samples_ms': walls, 'peak_extra_allocated_bytes': torch.cuda.max_memory_allocated()-base}

    for tokens in args.tokens:
        block = 256
        n = tokens//block
        k = torch.randn(n, args.kv_heads, block, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        c = CodecConfig(donor_policy='min_error', donor_backend=args.donor_backend) if args.donor_backend != 'torch' else CodecConfig(donor_policy='min_error')
        if args.donor_backend == 'triton':
            from inquant.triton_selection import reuse_map as selector
        else:
            selector = _reuse_map
        x = k.float(); samples=x[..., ::8, :]
        saliency=samples.abs().mean(-2); padding=samples.mean(-2)
        var, mean=torch.var_mean(x, dim=-2, correction=0); error=var+(mean-padding).square()
        dense_k=k.transpose(0,1).reshape(1,args.kv_heads,n*block,128)
        dense_v=v.transpose(0,1).reshape_as(dense_k)
        keys = _quantize_temporal_blocks(dense_k,n,block,c)
        values=quantize_value_blocks(dense_v,n,block,2,64)
        state=PackedDecodeState(keys,values)
        q=torch.randn(1,args.query_heads,1,128,device='cuda',dtype=torch.bfloat16)
        stages={
            'sample_statistics':lambda:(k.float()[...,::8,:].abs().mean(-2),k.float()[...,::8,:].mean(-2)),
            'donor_statistics':lambda:torch.var_mean(k.float(),dim=-2,correction=0),
            'donor_selection':lambda:selector(saliency,16,error),
            'key_quantize':lambda:quantize(k,c),
            'value_quantize':lambda:quantize_values(v),
            'key_blocks_including_export':lambda:_quantize_temporal_blocks(dense_k,n,block,c),
            'value_blocks_including_export':lambda:quantize_value_blocks(dense_v,n,block,2,64),
            'descriptor_build':lambda:PackedDecodeState(keys,values),
            'decode':lambda:state.decode(q),
        }
        item={'tokens':tokens,'blocks':n,'stages':{}}
        for name,fn in stages.items():
            item['stages'][name]=timed(fn)
            print(tokens,name,round(item['stages'][name]['wall_ms'],3),flush=True)
        item['descriptor_and_workspace_bytes']=state.nbytes
        report['workloads'].append(item)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        del state,keys,values,k,v,x,samples,dense_k,dense_v


if __name__=='__main__':
    main()
