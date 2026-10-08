#!/usr/bin/env python3
"""Interleaved old/new decode on identical packed tensors and the same GPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from inquant.codec import CodecConfig
    from inquant.cache import _quantize_temporal_blocks
    from inquant.value_codec import quantize_value_blocks
    from inquant.triton_attention import PackedDecodeState
    spec=importlib.util.spec_from_file_location('inquant._frozen_triton',args.baseline_root/'src/inquant/triton_attention.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    torch.set_grad_enabled(False);torch.manual_seed(567)
    result=[]
    for blocks in [8,32,128]:
        k,v=torch.randn(2,1,4,blocks*256,128,device='cuda',dtype=torch.bfloat16)
        keys=_quantize_temporal_blocks(k,blocks,256,CodecConfig(donor_policy='min_error',donor_backend='triton'))
        vals=quantize_value_blocks(v,blocks,256,2,64)
        old,new=module.PackedDecodeState(keys,vals),PackedDecodeState(keys,vals)
        q=torch.randn(1,28,1,128,device='cuda',dtype=torch.bfloat16)
        for _ in range(5):old.decode(q);new.decode(q)
        torch.testing.assert_close(old.decode(q),new.decode(q),atol=0,rtol=0)
        samples={name:[] for name in ['baseline','candidate']}
        for i in range(40):
            for name,state in ([('baseline',old),('candidate',new)] if i%2 else [('candidate',new),('baseline',old)]):
                torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter();a.record();state.decode(q);b.record();b.synchronize()
                samples[name].append({'wall_ms':1000*(time.perf_counter()-t),'cuda_ms':a.elapsed_time(b)})
        item={'blocks':blocks,'tokens':blocks*256,'outputs_bitwise_equal':True,
              'baseline_bytes':old.nbytes,'candidate_bytes':new.nbytes,'samples':samples,
              'medians':{name:{key:statistics.median(r[key] for r in rows) for key in ['wall_ms','cuda_ms']} for name,rows in samples.items()}}
        result.append(item);print(json.dumps({k:v for k,v in item.items() if k!='samples'}),flush=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
