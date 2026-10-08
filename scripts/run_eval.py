#!/usr/bin/env python3
"""Dense Qwen2.5/Mistral packed InQuant and cache comparison experiments.

Reference InQuant materializes KV for SDPA; it is NOT a fused serving backend.
Every successful measured sample includes its real prediction and cache bytes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inquant.evaluation import fingerprint, read_jsonl
from inquant.benchmarks import (generation_budget, grading_protocol, score_row,
                                tokenizer_fingerprint, validate_resume_row)


def cache_bytes(cache):
    if hasattr(cache, "nbytes"):
        return cache.nbytes
    # Count allocations, not just a potentially small view of a retained allocation.
    storages = {}
    for tensor in cache.key_cache + cache.value_cache:
        if hasattr(tensor, "untyped_storage"):
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def software_versions():
    result = {}
    for name in ("torch", "transformers", "kvpress", "math-verify"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def snapkv_skip_reason(input_length, window_size, compression_ratio):
    """The requested budget must retain the entire observation window.

    KVpress 0.2 uses top-k over tied maximum scores for that window. With a
    smaller budget it arbitrarily drops recent tokens, including the chat
    suffix. Keep short prompts intact rather than violating SnapKV's window.
    """
    if input_length <= window_size:
        return "prompt_not_longer_than_observation_window"
    if int(input_length * (1 - compression_ratio)) <= window_size:
        return "retained_budget_not_larger_than_observation_window"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen2.5_7b.json")
    p.add_argument("--data", required=True)
    p.add_argument("--method", choices=["bf16", "inquant", "inquant_fused", "uniform_int4", "snapkv", "knorm", "zipcache"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", help="Override model ID with a local checkpoint path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, help="Smoke test only; not valid for full accuracy acceptance")
    p.add_argument("--compression-ratio", type=float, help="KVpress fraction of tokens REMOVED")
    p.add_argument("--salient-fraction", type=float)
    p.add_argument("--donor-policy", choices=["neighbors", "min_error"])
    p.add_argument("--donor-backend", choices=["torch", "triton", "auto"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume matching complete JSONL records without overwriting")
    p.add_argument("--min-free-gib", type=float, default=24, help="Fail early if the selected GPU lacks this much free memory")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.method == "zipcache":
        config.setdefault("zipcache", {"important_bits": 4, "unimportant_bits": 2, "unimportant_ratio": .4,
                                      "streaming_gap": 100, "probe_chunk_size": 32})
    if args.model:
        config["model"] = args.model
    for name in ("compression_ratio", "salient_fraction", "donor_policy", "donor_backend"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    rows = read_jsonl(args.data)
    if args.limit is not None:
        if args.limit < 1:
            p.error("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        p.error("Dataset is empty")
    if len({(row["dataset"], row["id"]) for row in rows}) != len(rows):
        p.error("Duplicate dataset IDs")
    if config["max_new_tokens"] < 1 or config["warmup_runs"] < 0 or config["repeats"] < 1:
        p.error("Invalid generation/repetition configuration")
    for row in rows:
        generation_budget(row, config)
    if args.dry_run:
        print(json.dumps({"method": args.method, "config": config, "samples": len(rows),
                          "execution": "not_started", "output": args.output}, indent=2))
        return

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from inquant.cache import CacheConfig, InQuantCache
    from inquant.codec import CodecConfig
    if not transformers.__version__.startswith("4.53."):
        raise RuntimeError("This reference adapter requires transformers 4.53.x; use a separate environment")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no GPU measurements will be produced")
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < args.min_free_gib * 1024**3:
            raise RuntimeError(f"Selected GPU has only {free_bytes / 1024**3:.2f} GiB free; wait for an available GPU")
    if config["dtype"] != "bfloat16":
        raise ValueError("This experiment protocol requires BF16 model weights and activations")
    if any(row["dataset"] in ("math500", "math_train") for row in rows):
        import math_verify  # Fail before loading the model if the official grader is unavailable.
    press = None
    if args.method in ("snapkv", "knorm"):
        if version("kvpress") != "0.2.0":
            raise RuntimeError("Historical HF4.53 baseline requires kvpress==0.2.0")
        from kvpress import KnormPress, SnapKVPress
        cls = SnapKVPress if args.method == "snapkv" else KnormPress
        press = cls(compression_ratio=config["compression_ratio"])
    torch.manual_seed(config["seed"])
    load_options = {"revision": config["revision"],
                    "local_files_only": config.get("local_files_only", False)}
    if load_options["local_files_only"] and not Path(config["model"]).is_dir():
        raise ValueError("This preset requires a complete local checkpoint; pass --model /path/to/model")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], **load_options)
    prepared_tokenizer_hashes = {row["tokenizer_fingerprint"] for row in rows if "tokenizer_fingerprint" in row}
    if prepared_tokenizer_hashes and prepared_tokenizer_hashes != {tokenizer_fingerprint(tokenizer)}:
        raise ValueError("Prepared benchmark tokenizer does not match the loaded tokenizer")
    hf_config = AutoConfig.from_pretrained(config["model"], **load_options)
    if hf_config.model_type not in ("qwen2", "mistral") or getattr(hf_config, "num_experts", 0):
        raise ValueError("Expected a dense Qwen2 or Mistral model")
    if config.get("rope_scaling"):
        hf_config.rope_scaling = config["rope_scaling"]
        hf_config.max_position_embeddings = int(config["rope_scaling"]["original_max_position_embeddings"] * config["rope_scaling"]["factor"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], **load_options, config=hf_config,
        torch_dtype=torch.bfloat16, attn_implementation=config["attention"],
    ).to(device).eval()
    fused_handle = None
    if args.method == "inquant_fused":
        from inquant.qwen_fused import enable_fused_attention
        fused_handle = enable_fused_attention(model)
    if args.method == "zipcache":
        from inquant.zipcache import ZipCache, ZipCacheConfig, enable_zipcache_qwen2, official_codec, UPSTREAM_REVISION
        zip_config = ZipCacheConfig(**config["zipcache"])
        zip_handle = enable_zipcache_qwen2(model)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    codec = CodecConfig(sample_stride=config["sample_stride"],
                        salient_fraction=0.0 if args.method == "uniform_int4" else config["salient_fraction"],
                        donor_policy=config.get("donor_policy", "neighbors"),
                        donor_backend=config.get("donor_backend", "torch"))
    cache_config = CacheConfig(codec=codec, block_size=config["block_size"],
                               residual_length=config["residual_length"], sink_tokens=config["sink_tokens"],
                               validate_positions=config.get("validate_positions", False),
                               shared_workspace=config.get("shared_workspace", False),
                               value_bits=config.get("value_bits"),
                               value_group_size=config.get("value_group_size", 64),
                               track_peak_bytes=config.get("track_peak_bytes", False))
    metadata = {
        "model": config["model"], "model_revision": getattr(hf_config, "_commit_hash", None) or config["revision"],
        "backend": "transformers-reference", "dtype": config["dtype"], "attention": config["attention"],
        "batch_size": 1, "max_new_tokens": config["max_new_tokens"], "rope_scaling": config.get("rope_scaling"),
        "device_type": device.type, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "software": software_versions(), "grading": "explicit-boxed-gsm8k+math-verify-0.8.0",
        "method": args.method, "config": config, "warmup_runs": config["warmup_runs"],
        "fixed_output_length": config["fixed_output_length"],
        "production_backend": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_repo": config["model"],
        "model_architecture": {key: getattr(hf_config, key) for key in
                               ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size")},
    }
    if args.method == "zipcache":
        metadata["zipcache_implementation"] = {
            "upstream_repo": "https://github.com/ThisisBillhe/ZipCache", "upstream_revision": UPSTREAM_REVISION,
            "codec_sha256": official_codec().source_sha256, "port": "qwen2_sdpa_chunked_probes_v2",
            "fused_dequant_attention": False, "randomness": "per_request_cpu_generator_from_dataset_id_and_seed",
        }
    if prepared_tokenizer_hashes:
        source_root = Path(__file__).resolve().parents[1]
        source_files = [Path(__file__).resolve(), *sorted((source_root / 'src/inquant').glob('*.py'))]
        metadata['implementation_fingerprint'] = fingerprint({
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files})
    data_hash = fingerprint(rows)
    download_manifest = Path(config["model"]) / "inquant_download_manifest.json"
    if download_manifest.exists():
        downloaded = json.loads(download_manifest.read_text())
        metadata["model_revision"] = downloaded["revision"]
        metadata["model_repo"] = downloaded["repo"]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def generate(row):
        max_new_tokens = generation_budget(row, config)
        cache = InQuantCache(cache_config) if args.method in ("inquant", "inquant_fused", "uniform_int4") else DynamicCache()
        if args.method == "zipcache":
            request_seed = int(fingerprint({"seed": config["seed"], "dataset": row["dataset"], "id": row["id"]})[:15], 16)
            cache = ZipCache(zip_config, seed=request_seed)
        fused_before = fused_handle.fused_decode_calls if fused_handle else 0
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        if "input_ids" in row:
            if row.get("tokenizer_model") != config["model"] and not Path(config["model"]).exists():
                raise ValueError("Pre-tokenized workload tokenizer does not match model")
            token_ids = row["input_ids"]
        else:
            prompt = row["question"] + "\nSolve step by step. Put your final answer in \\boxed{...}."
            messages = [{"role": "user", "content": prompt}]
            if hf_config.model_type == "qwen2":
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_length = len(token_ids)
        if input_length + max_new_tokens > hf_config.max_position_embeddings:
            raise ValueError("Input + output exceeds the model context window; shorten the workload or use separately validated RoPE scaling")
        effective_cache_config = None
        if isinstance(cache, InQuantCache):
            effective_cache_config = cache_config
            short = config.get("short_context")
            if short and input_length + max_new_tokens <= short["max_context_tokens"]:
                effective_cache_config = replace(cache_config, block_size=short["block_size"],
                                                residual_length=short["residual_length"])
                cache = InQuantCache(effective_cache_config)
        inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
        # The official SnapKV window must be strictly shorter than q_len.
        # A skipped short prompt remains part of accuracy evaluation and is logged.
        skip_reason = (snapkv_skip_reason(input_length, press.window_size, press.compression_ratio)
                       if args.method == "snapkv" else None)
        short_snapkv = skip_reason is not None
        active_press = press if not short_snapkv else None
        with active_press(model) if active_press else contextlib.nullcontext():
            output = model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [int(token.item())]
        del output
        synchronize()
        ttft = time.perf_counter() - start
        prefill_bytes = cache_bytes(cache)
        prefill_stats = cache.memory_stats() if hasattr(cache, "memory_stats") else None
        decode_start = time.perf_counter()
        observer_seconds = decode_start - start - ttft
        for step in range(1, max_new_tokens):
            if not config["fixed_output_length"] and generated[-1] in eos_ids:
                break
            # Absolute RoPE positions are independent of the number of tokens retained by a press.
            position = torch.tensor([input_length + step - 1], device=device)
            output = model(input_ids=token, past_key_values=cache, use_cache=True,
                           position_ids=position.unsqueeze(0), cache_position=position, logits_to_keep=1)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            del output
        synchronize()
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        wall_seconds = time.perf_counter() - start
        elapsed = wall_seconds - observer_seconds
        final_bytes = cache_bytes(cache)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        score = score_row(row, prediction)
        record = {**metadata, "max_new_tokens": max_new_tokens, "grading": grading_protocol(row), "dataset": row["dataset"], "id": row["id"], "status": "ok",
                  "dataset_split": row.get("split", "synthetic" if row["dataset"] in ("passkey", "passkey32k") else "test"),
                  "dataset_fingerprint": row.get("dataset_fingerprint") or data_hash,
                  "prompt_sha256": fingerprint(token_ids), "input_tokens": input_length,
                  "generated_tokens": len(generated), "generated_token_ids": generated,
                  "prediction": prediction, "answer": row["answer"],
                  "correct": score == 1.0, "score": score,
                  "task": row.get("task"), "context_budget": row.get("context_budget"),
                  "benchmark_source": row.get("benchmark_source"),
                  "finish_reason": "fixed_length" if config["fixed_output_length"] else
                      ("eos" if generated[-1] in eos_ids else "length"),
                  "e2e_s": elapsed, "ttft_s": ttft,
                  "excluded_observer_s": observer_seconds, "wall_with_observer_s": wall_seconds,
                  "tpot_s": (elapsed - ttft) / max(1, len(generated) - 1),
                  "output_tokens_per_s": len(generated) / elapsed,
                  "prefill_kv_bytes": prefill_bytes, "final_kv_bytes": final_bytes,
                  "kv_cache_bytes": max(prefill_bytes, final_bytes, getattr(cache, "peak_nbytes", 0)), "peak_allocated_bytes": peak,
                  "prefill_cache_stats": prefill_stats,
                  "final_cache_stats": cache.memory_stats() if hasattr(cache, "memory_stats") else None,
                  "effective_cache_config": asdict(effective_cache_config) if effective_cache_config else None,
                  "press_skipped_short_prompt": short_snapkv,
                  "press_skip_reason": skip_reason,
                  "fused_decode_calls": fused_handle.fused_decode_calls - fused_before if fused_handle else 0,
                  "kv_measurement": "persistent_tensors_including_metadata_residual_and_packed_decode_state_workspace; transient_attention_allocations_reported_in_peak"}
        del cache
        return record

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and path.exists():
        allowed = {(row["dataset"], row["id"]): row for row in rows}
        for previous in read_jsonl(path):
            if previous.get("config") != config or previous.get("method") != args.method:
                raise ValueError("Resume file was generated with a different method/configuration")
            for key in ("model_revision", "model_repo", "model_architecture", "software", "device_name"):
                if previous.get(key) != metadata[key]:
                    raise ValueError(f"Resume identity mismatch: {key}")
            pair = (previous["dataset"], previous["id"])
            if pair not in allowed or previous["repeat"] >= config["repeats"]:
                raise ValueError("Resume file contains samples outside this run")
            if prepared_tokenizer_hashes and previous.get('implementation_fingerprint') != metadata['implementation_fingerprint']:
                raise ValueError('Resume implementation fingerprint mismatch')
            validate_resume_row(previous, allowed[pair], data_hash, config)
            key = (*pair, previous["repeat"])
            if key in completed:
                raise ValueError("Resume file has duplicate records")
            completed.add(key)
    if len(completed) == len(rows) * config["repeats"]:
        print("All requested records are already complete.", flush=True)
        return
    mode = "a" if args.resume and path.exists() else "x"
    with path.open(mode) as target:
        for _ in range(config["warmup_runs"]):
            generate(rows[0])
        for repeat in range(config["repeats"]):
            for row in rows:
                if (row["dataset"], row["id"], repeat) in completed:
                    continue
                result = generate(row)
                result["repeat"] = repeat
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                target.flush()
                print(json.dumps({k: result[k] for k in ("method", "dataset", "id", "repeat", "correct", "e2e_s", "kv_cache_bytes")}), flush=True)


if __name__ == "__main__":
    main()
