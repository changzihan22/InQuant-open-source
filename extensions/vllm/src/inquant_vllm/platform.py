"""Out-of-tree CUDA platform; no edits or monkey patches to vLLM."""
from . import check_version
check_version()

import torch
from vllm import envs
from vllm.platforms.cuda import CudaPlatform


class InQuantPlatform(CudaPlatform):
    @classmethod
    def check_and_update_config(cls, config):
        super().check_and_update_config(config)
        m, c, s, p = (config.model_config, config.cache_config,
                      config.scheduler_config, config.parallel_config)
        requirements = {
            "VLLM_USE_V1=1": envs.VLLM_USE_V1,
            "Qwen2ForCausalLM": m.hf_config.architectures == ["Qwen2ForCausalLM"],
            "dtype=bfloat16": m.dtype == torch.bfloat16,
            "enforce_eager=True": m.enforce_eager,
            "max_num_seqs=1": s.max_num_seqs == 1,
            "max_num_batched_tokens >= max_model_len": s.max_num_batched_tokens >= m.max_model_len,
            "long_prefill_token_threshold=0": s.long_prefill_token_threshold == 0,
            "enable_prefix_caching=False": not c.enable_prefix_caching,
            "tensor/pipeline/data parallel size=1": (
                p.tensor_parallel_size == p.pipeline_parallel_size == p.data_parallel_size == 1),
            "kv_cache_dtype=auto": c.cache_dtype == "auto",
            "block_size=64,128,256": c.block_size in (64, 128, 256),
            "no speculative decoding": config.speculative_config is None,
            "no KV transfer": config.kv_transfer_config is None,
            "no CPU offload": c.cpu_offload_gb == 0,
            "no sliding window": c.sliding_window is None,
            "no sleep mode": not m.enable_sleep_mode,
        }
        failed = [name for name, ok in requirements.items() if not ok]
        if failed:
            raise ValueError("InQuant vLLM 0.1 requires: " + "; ".join(failed))
        # EngineArgs in 0.9.1 forces V1's flag to True even when False was
        # requested. With one active request, a full-length token budget and
        # no threshold, the scheduler cannot split prefill. Align the flags
        # here; the metadata builder independently rejects any actual split.
        s.enable_chunked_prefill = False
        s.chunked_prefill_enabled = False
        if p.worker_cls not in ("vllm.v1.worker.gpu_worker.Worker", "inquant_vllm.worker.InQuantWorker"):
            raise ValueError("InQuant cannot combine with a different custom worker")
        p.worker_cls = "inquant_vllm.worker.InQuantWorker"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        if not use_v1 or use_mla or head_size != 128 or dtype != torch.bfloat16:
            raise ValueError("InQuant requires V1, BF16 Qwen2 head_dim=128, no MLA")
        return "inquant_vllm.backend.InQuantBackend"
