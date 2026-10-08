# InQuant for vLLM

An opt-in platform plugin for **vLLM 0.9.1**. It adds packed K4/V2 pages and Triton attention over compressed KV, without patching the installed vLLM source.

Start with the [main README](../../README.md#vllm-extension) to install the separate environment and download Qwen2.5-7B-Instruct. All commands below run from the repository root.

## Use the Python API

Save this example as a Python file and run it with `.venv-vllm/bin/python` from the repository root. The `__main__` guard is required because vLLM workers use process spawning.

```python
import os

os.environ["INQUANT_VLLM"] = "1"
os.environ["INQUANT_DIRECT_WRITE"] = "1"
os.environ["INQUANT_DONOR_BACKEND"] = "triton"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        model="models/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        block_size=64,
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.8,
    )
    try:
        ids = llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": "Calculate 17 times 23."}],
            tokenize=True,
            add_generation_prompt=True,
        )
        outputs = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=256),
        )
        print(outputs[0].outputs[0].text)
        print(llm.collective_rpc("inquant_memory_report")[0])
    finally:
        llm.llm_engine.engine_core.shutdown()
```

Enable the plugin before importing vLLM. Installation alone leaves the default backend unchanged. To disable it, start a new process with `INQUANT_VLLM=0`; switching platforms inside an existing process is not supported.

## Run with prepared 32K inputs

After generating `data/qwen_passkey32k.jsonl` with the example in the main README:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python scripts/run_vllm.py \
  --config configs/vllm_inquant_32k.json \
  --model models/Qwen2.5-7B-Instruct \
  --prompts data/qwen_passkey32k.jsonl \
  --output outputs/vllm/inquant_32k.json
```

For a native BF16 control, use `configs/vllm_bf16_extension_control_32k.json` in a fresh process and choose a different output path.

## Implementation

| File | Responsibility |
|---|---|
| `src/inquant_vllm/platform.py` | Platform selection and compatibility checks |
| `src/inquant_vllm/worker.py` | Page allocation specification and memory reporting |
| `src/inquant_vllm/layout.py` | Packed payload, scales, and decoding descriptors |
| `src/inquant_vllm/backend.py` | Page lifecycle, uncompressed sink, and active tail |
| `src/inquant_vllm/kernels.py` | Triton decode using vLLM block tables |
| `src/inquant_vllm/packing.py` | Direct writes to physical cache pages |

K uses 4-bit slots with 8-bit salient values and sampled-mean donor reconstruction. V uses 2-bit affine groups of 64 channels. Full pages are sealed once; the active partial page and the attention sink remain in BF16. Decode reads packed pages directly. The supplied presets combine fused GPU donor selection with direct page writes; statistics and quantization retain the reference codec rules.

vLLM's native manager still allocates and recycles physical pages. The extension registers its own cache specification with internal vLLM interfaces, so compatibility is deliberately restricted to version 0.9.1.

## Direct page sealing

The supplied presets set `INQUANT_DIRECT_WRITE=1` and `INQUANT_DONOR_BACKEND=triton`. Fresh pages are quantized with deterministic fused donor selection and written straight to their assigned physical slots. The packed page format is unchanged. Set `INQUANT_DIRECT_WRITE=0` to use the original staging-and-scatter path.

Correctness tests cover scattered and reused page IDs, 4/8 KV heads, page boundaries, request reset, and byte addresses above 2 GiB. Mistral integration is available through Hugging Face. See the root README for model support, runnable examples, and measured runtime results.

## Supported configuration

- V1 engine, CUDA, BF16 weights, Qwen2 full attention, head dimension 128.
- One GPU, one active request, eager execution.
- 64-token or 256-token pages in the supplied math and long-context configurations.
- No continuous batching, CUDA Graphs, prefix caching, actual chunked prefill, TP/PP/DP, speculative decoding, KV transfer, or sleep mode.

The plugin rejects unsupported combinations. Removing a compatibility check does not implement the missing feature. This backend has different page boundaries and tail retention from HF, so accuracy measurements cannot be transferred between the two backends. It has not demonstrated a latency improvement over its paired native vLLM control.

Memory reports distinguish reserved pool bytes from active pages and auxiliary storage. A smaller active KV representation does not imply the same percentage reduction in process memory: vLLM preallocates its cache pool according to `gpu_memory_utilization`.

## Tests

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python -m pytest extensions/vllm/tests -q
```

The tests cover page layout, reconstruction, fused attention, page boundaries, request reuse, and configuration checks. GPU tests require a compatible CUDA device and the pinned environment.

## License

Original extension code uses the [MIT License](LICENSE). Installed dependencies retain their own licenses.
