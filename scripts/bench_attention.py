#!/usr/bin/env python3
"""Synthetic one-layer packed-attention benchmark; not a model E2E result.

Measures original BF16 SDPA, already-dequantized BF16 SDPA, and reused packed
Triton state with CUDA events and host wall time. Includes both warm-cache and
64MiB-flushed measurements, actual persistent bytes, and reference parity.
Model quality, complete generation, append/sealing and setup are separate costs.
"""
import argparse
import hashlib
import json
import sys
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inquant.cache import _quantize_temporal_blocks
from inquant.codec import CodecConfig
from inquant.triton_attention import PackedDecodeState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--query-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1457)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/triton_microbench.json")
    args = parser.parse_args()
    if args.tokens < 388 or args.repeats < 1:
        parser.error("Need at least 388 tokens and one measurement repeat")
    if args.kv_heads < 1 or args.query_heads < 1 or args.query_heads % args.kv_heads:
        parser.error("Positive query-head count must be divisible by KV-head count")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    tokens, kv_heads, query_heads, dimension = args.tokens, args.kv_heads, args.query_heads, 128
    block_size, sink_tokens = 256, 4
    block_count = (tokens - sink_tokens - 128) // block_size
    tail_tokens = tokens - sink_tokens - block_count * block_size
    key = torch.randn((1, kv_heads, tokens, dimension), device=device, dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn((1, query_heads, 1, dimension), device=device, dtype=dtype)
    sink_k, sink_v = key[:, :, :sink_tokens].clone(), value[:, :, :sink_tokens].clone()
    tail_k, tail_v = key[:, :, -tail_tokens:].clone(), value[:, :, -tail_tokens:].clone()
    torch.cuda.synchronize()
    start = time.perf_counter()
    keys = _quantize_temporal_blocks(key[:, :, sink_tokens:], block_count, block_size, CodecConfig())
    values = _quantize_temporal_blocks(value[:, :, sink_tokens:], block_count, block_size, CodecConfig())
    torch.cuda.synchronize()
    quantize_seconds = time.perf_counter() - start
    start = time.perf_counter()
    state = PackedDecodeState(keys, values)
    torch.cuda.synchronize()
    state_setup_seconds = time.perf_counter() - start


    def materialize():
        return (torch.cat([sink_k] + [x.dequantize() for x in keys] + [tail_k], dim=2),
                torch.cat([sink_v] + [x.dequantize() for x in values] + [tail_v], dim=2))


    reference_k, reference_v = materialize()


    def sdpa_original():
        return F.scaled_dot_product_attention(query, key, value, enable_gqa=True)


    def sdpa_dequantized():
        return F.scaled_dot_product_attention(query, reference_k, reference_v, enable_gqa=True)


    def packed():
        return state.decode(query, sink_key=sink_k, sink_value=sink_v, residual_key=tail_k, residual_value=tail_v)


    def materialize_and_sdpa():
        dk, dv = materialize()
        return F.scaled_dot_product_attention(query, dk, dv, enable_gqa=True)


    expected = sdpa_dequantized()
    actual = packed()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.02)
    max_error = (actual.float() - expected.float()).abs().max().item()
    rmse = (actual.float() - expected.float()).square().mean().sqrt().item()
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)


    def measure(fn, repetitions=30, cold_cache=False):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        durations = []
        wall = []
        for _ in range(repetitions):
            if cold_cache:
                flush.zero_()
                torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            fn()
            end_event.record()
            end_event.synchronize()
            durations.append(start_event.elapsed_time(end_event))
            wall.append(1000 * (time.perf_counter() - wall_start))
        return {"repetitions": repetitions, "cuda_event_median_ms": statistics.median(durations),
                "cuda_event_min_ms": min(durations), "cuda_event_max_ms": max(durations),
                "host_wall_median_ms": statistics.median(wall)}


    results = {}
    for cache_mode in ("warm", "flushed_64MiB"):
        results[cache_mode] = {}
        for name, fn in (("bf16_sdpa", sdpa_original), ("dequantized_bf16_sdpa", sdpa_dequantized),
                         ("packed_triton", packed)):
            measured = measure(fn, repetitions=args.repeats, cold_cache=cache_mode != "warm")
            results[cache_mode][name] = measured
            print(cache_mode, name, measured, flush=True)
    results["reference_materialize_plus_sdpa"] = measure(materialize_and_sdpa, repetitions=3)
    exact_bytes = sum(x.numel() * x.element_size() for x in (sink_k, sink_v, tail_k, tail_v))
    packed_block_bytes = sum(x.nbytes for x in keys + values)
    baseline_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
    report = {
        "status": "measured_microbenchmark_only_not_model_e2e",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda, "dtype": "bfloat16", "batch_size": 1,
        "query_heads": query_heads, "kv_heads": kv_heads, "head_dimension": dimension,
        "input_tokens": tokens, "query_tokens": 1, "packed_blocks": len(keys),
        "block_size": block_size, "sink_tokens": sink_tokens, "residual_tokens": tail_tokens,
        "salient_fraction": 0.125, "sample_stride": 8,
        "state_reused": True, "sdpa_enable_gqa": True, "seed": args.seed,
        "kernel_config": {"tile_tokens": 32, "num_warps": 4, "num_stages": 1, "shared_gqa_tile": True},
        "kernel_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/inquant/triton_attention.py").read_bytes()).hexdigest(),
        "quantize_path": "cache._quantize_temporal_blocks",
        "parity_max_absolute_error": max_error, "parity_rmse": rmse,
        "setup": {"quantize_wall_seconds": quantize_seconds, "descriptor_state_wall_seconds": state_setup_seconds},
        "bytes": {"bf16_kv": baseline_bytes, "packed_blocks_including_codec_metadata": packed_block_bytes,
                  "sink_and_residual": exact_bytes, "auxiliary_tables_maps_and_workspace": state.nbytes,
                  "total_persistent": packed_block_bytes + exact_bytes + state.nbytes,
                  "reduction_fraction": 1 - (packed_block_bytes + exact_bytes + state.nbytes) / baseline_bytes},
        "measurements": results,
        "limitations": ["Synthetic one-layer single-query attention only; no model E2E or accuracy evidence.",
                        "Setup and cache append/sealing excluded from decode timings and separately reported.",
                        "Each event measurement synchronizes; Python dispatch and synchronization are included in host wall time.",
                        "Warm-cache and 64MiB cache-flush measurements are both reported; multilayer behavior can differ.",
                        "GPU may be shared; clock and power state are not locked."]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
