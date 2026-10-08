#!/usr/bin/env python3
"""Interleaved page sealing comparison on one GPU, including temporary memory."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'extensions/vllm/src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('Use at least three repetitions')
    import torch
    from inquant_vllm.layout import PageLayout, pack_pages, pack_pages_into
    torch.set_grad_enabled(False)
    torch.manual_seed(567)
    rows = []
    for block in (64, 256):
        for count in (1, 8, 128):
            layout = PageLayout(block, 4)
            key, value = torch.randn(2, count, 4, block, 128, device='cuda', dtype=torch.bfloat16)
            pool = torch.empty((count + 3, layout.page_bytes), device='cuda', dtype=torch.uint8)
            ids = torch.randperm(count, device='cuda').to(torch.int32)
            ids64 = ids.long()
            calls = {
                'staged_torch': lambda: pool.index_copy_(0, ids64, pack_pages(key, value, layout)),
                'direct_torch': lambda: pack_pages_into(key, value, layout, pool, ids, donor_backend='torch'),
                'direct_triton': lambda: pack_pages_into(key, value, layout, pool, ids, donor_backend='triton'),
            }
            expected = pack_pages(key, value, layout)
            for call in calls.values():
                for _ in range(3):
                    call()
                torch.testing.assert_close(pool[ids64], expected, rtol=0, atol=0)
            del expected
            samples = {name: [] for name in calls}
            for repeat in range(args.repeats):
                order = list(calls) if repeat % 2 else list(reversed(calls))
                for name in order:
                    torch.cuda.synchronize()
                    start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    t0 = time.perf_counter()
                    start_event.record()
                    calls[name]()
                    end_event.record()
                    end_event.synchronize()
                    samples[name].append({'wall_ms': (time.perf_counter() - t0) * 1000,
                                          'cuda_ms': start_event.elapsed_time(end_event)})
            temporary = {}
            for name, call in calls.items():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                call()
                torch.cuda.synchronize()
                temporary[name] = torch.cuda.max_memory_allocated() - allocated
            row = {'block_size': block, 'pages': count, 'bytes_equal': True,
                   'temporary_peak_bytes': temporary, 'samples': samples,
                   'medians': {name: {key: statistics.median(r[key] for r in values)
                                      for key in ('wall_ms', 'cuda_ms')}
                               for name, values in samples.items()}}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                                      'repeats': args.repeats, 'results': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
