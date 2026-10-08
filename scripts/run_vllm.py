#!/usr/bin/env python3
"""Measure a native vLLM baseline or the separately installed InQuant extension."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inquant.evaluation import grade


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    engine = config["engine"]
    work = config["workload"]
    extension = config.get("backend") == "inquant_vllm"
    if config.get("backend") not in (None, "vllm_native_baseline", "inquant_vllm"):
        raise ValueError("Unknown backend")
    if extension and config.get("environment", {}).get("INQUANT_VLLM") != "1":
        raise ValueError("InQuant config requires INQUANT_VLLM=1")
    if engine.get("kv_cache_dtype", "auto") not in {
        "auto", "fp8", "fp8_e4m3", "fp8_e5m2", "int4_per_token_head"
    }:
        raise ValueError("Use native cache dtype names; InQuant uses its own byte cache spec with dtype=auto")
    if engine.get("enable_prefix_caching") is not False:
        raise ValueError("Set enable_prefix_caching=false for independent repeated measurements.")
    for key in ("input_tokens", "output_tokens", "batch_size", "repeats", "warmup_runs"):
        if isinstance(work[key], bool) or not isinstance(work[key], int) or work[key] <= 0:
            raise ValueError(f"workload.{key} must be a positive integer")
    if work["warmup_runs"] < 2:
        raise ValueError("At least two warmup runs are required by the experiment protocol")
    if work["input_tokens"] + work["output_tokens"] > engine["max_model_len"]:
        raise ValueError("input_tokens + output_tokens exceeds max_model_len")
    if work["batch_size"] > engine["max_num_seqs"]:
        raise ValueError("batch_size exceeds max_num_seqs")
    return config


def prepare_prompts(tokenizer, work: dict, path: Path | None):
    """Accept prepared input_ids without retokenization; retain dataset provenance."""
    prompts = []
    metadata = []
    if path is not None:
        for index, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if "input_ids" in row:
                tokens = row["input_ids"]
            elif "prompt_token_ids" in row:
                tokens = row["prompt_token_ids"]
            elif "token_ids" in row:
                tokens = row["token_ids"]
            elif "messages" in row:
                tokens = tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=True
                )
            else:
                tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
            if not isinstance(tokens, list) or not tokens or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens
            ):
                raise ValueError(f"Invalid prompt tokens at JSONL line {index + 1}")
            prompts.append({"prompt_token_ids": tokens})
            metadata.append({
                **{key: row[key] for key in (
                    "dataset", "answer", "dataset_fingerprint", "tokenizer_model", "tokenizer_revision"
                ) if key in row},
                "id": str(row.get("id", index)),
            })
        if not prompts:
            raise ValueError("JSONL contains no prompts")
    else:
        phrase = tokenizer.encode(
            "This is a deterministic long context throughput benchmark. ",
            add_special_tokens=False,
        )
        if not phrase:
            raise ValueError("Tokenizer returned no tokens for synthetic context")
        count = work["input_tokens"]
        tokens = (phrase * ((count + len(phrase) - 1) // len(phrase)))[:count]
        prompts = [{"prompt_token_ids": list(tokens)} for _ in range(work["batch_size"])]
        metadata = [{"id": str(i)} for i in range(work["batch_size"])]
    return prompts, metadata


def run(config: dict, prompts_path: Path | None) -> dict:
    actual_version = importlib.metadata.version("vllm")
    expected_version = config["vllm_version"]
    if actual_version != expected_version:
        raise RuntimeError(
            f"Expected vllm=={expected_version}, found {actual_version}. "
            "Use the pinned environment, or create and validate a separately versioned config."
        )
    for name, value in config.get("environment", {}).items():
        if not name.startswith("VLLM_") and name not in {"INQUANT_VLLM", "INQUANT_DIRECT_WRITE", "INQUANT_DONOR_BACKEND"}:
            raise ValueError("Only VLLM_* and explicit InQuant runtime options are allowed")
        if name == "INQUANT_DIRECT_WRITE" and str(value) not in ("0", "1"):
            raise ValueError("INQUANT_DIRECT_WRITE must be 0 or 1")
        if name == "INQUANT_DONOR_BACKEND" and value not in ("torch", "triton", "auto"):
            raise ValueError("INQUANT_DONOR_BACKEND must be torch, triton or auto")
        os.environ[name] = str(value)
    extension = config.get("backend") == "inquant_vllm"
    if extension != (os.environ.get("INQUANT_VLLM") == "1"):
        raise ValueError("INQUANT_VLLM does not match config backend; use a fresh process per method")
    # vLLM 0.9.1 may otherwise fork after the CUDA availability probe.
    # This script has a __main__ guard, so spawn is safe for engine workers.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable: vLLM performance measurements cannot run here")
    from vllm import LLM, SamplingParams

    setup_start = time.perf_counter()
    engine_args = dict(config["engine"])
    download_manifest = None
    manifest_path = Path(engine_args["model"]) / "inquant_download_manifest.json"
    if manifest_path.is_file():
        download_manifest = json.loads(manifest_path.read_text())
        for entry in download_manifest["files"]:
            checkpoint_file = manifest_path.parent / entry["file"]
            if not checkpoint_file.is_file() or checkpoint_file.stat().st_size != entry["bytes"]:
                raise ValueError(f"Checkpoint does not match finalized manifest: {entry['file']}")
    llm = LLM(**engine_args)
    tokenizer = llm.get_tokenizer()
    work = config["workload"]
    prompts, row_metadata = prepare_prompts(tokenizer, work, prompts_path)
    for row in row_metadata:
        if (
            row.get("tokenizer_model")
            and row["tokenizer_model"] != engine_args["model"]
            and not Path(engine_args["model"]).exists()
        ):
            raise ValueError("Prepared workload tokenizer_model does not match engine model")
    lengths = [len(row["prompt_token_ids"]) for row in prompts]
    if max(lengths) + work["output_tokens"] > engine_args["max_model_len"]:
        raise ValueError("Actual tokenized input plus output exceeds max_model_len")
    if min(lengths) < work["input_tokens"]:
        raise ValueError("Actual prompt length below configured input_tokens; 32K gate cannot be assumed")
    sampling = SamplingParams(
        temperature=0.0, max_tokens=work["output_tokens"], ignore_eos=True,
        seed=engine_args.get("seed", 0),
    )
    def generate_workload():
        outputs = []
        for offset in range(0, len(prompts), work["batch_size"]):
            outputs.extend(llm.generate(
                prompts[offset:offset + work["batch_size"]], sampling, use_tqdm=False
            ))
        return outputs

    setup_seconds = time.perf_counter() - setup_start
    for _ in range(work["warmup_runs"]):
        generate_workload()
    # LLM.generate is blocking even when its CUDA work runs in engine workers.
    # The parent process's CUDA memory counters do not describe worker memory.
    timings = []
    outputs = None
    for repeat in range(work["repeats"]):
        started = time.perf_counter()
        outputs = generate_workload()
        elapsed = time.perf_counter() - started
        produced = sum(len(result.outputs[0].token_ids) for result in outputs)
        expected = len(prompts) * work["output_tokens"]
        if produced != expected:
            raise RuntimeError(f"Output-length mismatch: expected {expected}, generated {produced}")
        timings.append({
            "repeat": repeat,
            "wall_seconds": elapsed,
            "output_tokens": produced,
            "output_tokens_per_second": produced / elapsed,
            "requests_per_second": len(prompts) / elapsed,
            "outputs": [{
                **metadata,
                "text": result.outputs[0].text,
                "token_ids": list(result.outputs[0].token_ids),
                "retrieval_correct": grade("passkey32k", result.outputs[0].text, metadata["answer"])
                if metadata.get("dataset") == "passkey32k" else None,
            } for metadata, result in zip(row_metadata, outputs)],
        })
    median_seconds = statistics.median(row["wall_seconds"] for row in timings)
    model_config = llm.llm_engine.model_config
    hf_config = model_config.hf_config
    manifest = {
        "python": platform.python_version(),
        "vllm": actual_version,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "model_revision_resolved": download_manifest["revision"] if download_manifest else getattr(hf_config, "_commit_hash", None),
        "model_repo": download_manifest["repo"] if download_manifest else engine_args["model"],
        "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if download_manifest else None,
        "model_type": getattr(hf_config, "model_type", None),
        "model_architecture": {
            key: getattr(hf_config, key, None) for key in (
                "architectures", "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads"
            )
        },
        "attention_backend_requested": config.get("environment", {}).get("VLLM_ATTENTION_BACKEND", "auto"),
        "actual_attention_backend": "see captured vLLM startup log",
        "worker_multiproc_method": os.environ["VLLM_WORKER_MULTIPROC_METHOD"],
    }
    prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    retrieval = [output["retrieval_correct"] for row in timings for output in row["outputs"]
                 if output["retrieval_correct"] is not None]
    worker_reports = llm.collective_rpc("inquant_memory_report") if extension else None
    if extension:
        manifest["actual_attention_backend"] = worker_reports[0]["backend"]
        manifest["inquant_vllm"] = importlib.metadata.version("inquant-vllm")
    report = {
        "status": "measured",
        "method": config["name"],
        "backend": "inquant_vllm" if extension else "vllm_native_baseline",
        "inquant_backend_implemented": extension,
        "config": config,
        "environment": manifest,
        "verified_download_manifest": download_manifest,
        "prompt_sha256": prompt_hash,
        "prepared_jsonl_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest() if prompts_path else None,
        "synthetic_workload": prompts_path is None,
        "requests_per_run": len(prompts),
        "request_batch_size": work["batch_size"],
        "input_tokens_per_request": lengths,
        "output_tokens_per_request": work["output_tokens"],
        "setup_seconds_excluded": setup_seconds,
        "timing_scope": "whole workload in sequential request batches: blocking generate wall time includes scheduler + prefill + decode; excludes initialization, warmup and tokenization",
        "runs": timings,
        "median_wall_seconds": median_seconds,
        "median_output_tokens_per_second": len(prompts) * work["output_tokens"] / median_seconds,
        "retrieval": {"correct": sum(retrieval), "total": len(retrieval), "all_correct": all(retrieval)} if retrieval else None,
        "measured_live_kv_bytes": None,
        "measured_peak_gpu_bytes": None,
        "worker_memory_reports": worker_reports,
        "acceptance_eligible": False,
        "acceptance_note": ("InQuant paged extension integration/long-context measurement; full GSM8K/MATH500 "
                            "have not been graded for this paged variant. Not full acceptance evidence."
                            if extension else "Native baseline only. No math grading or worker KV instrumentation. Do not treat as InQuant acceptance evidence."),
        "last_run_outputs": timings[-1]["outputs"],
    }
    source_root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__)]
    if extension:
        sources += sorted((source_root / "extensions/vllm/src/inquant_vllm").glob("*.py"))
        sources += [source_root / "src/inquant" / name for name in
                    ("codec.py", "value_codec.py", "triton_attention.py", "triton_selection.py")]
    report["source_sha256"] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources}
    # Close V1 workers while Python/native modules are still alive, rather than
    # leaving multiprocessing and CUDA cleanup to interpreter shutdown.
    llm.llm_engine.engine_core.shutdown()
    report["engine_shutdown"] = "explicit_completed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", help="Override the model with a local checkpoint path or Hub ID")
    parser.add_argument("--prompts", type=Path, help="Optional JSONL; input_ids, prompt_token_ids, token_ids, messages, or raw prompt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate JSON only; no GPU, model download or measurements")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.model:
            config["engine"]["model"] = args.model
        if args.dry_run:
            report = {
                "status": "validated_config_only", "config": config,
                "acceptance_eligible": False, "inquant_backend_implemented": config.get("backend") == "inquant_vllm",
            }
        else:
            report = run(config, args.prompts)
        code = 0
    except Exception as exc:
        report = {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "acceptance_eligible": False, "inquant_backend_implemented": False,
        }
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
