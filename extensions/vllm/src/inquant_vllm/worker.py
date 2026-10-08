"""vLLM Worker extension: packed allocator spec and worker-side telemetry."""
from dataclasses import dataclass
import math
import torch
from vllm.attention.layer import Attention
from vllm.config import get_layers_from_vllm_config
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu_worker import Worker
from .layout import PageLayout


@dataclass
class InQuantSpec(FullAttentionSpec):
    @property
    def page_size_bytes(self):
        return PageLayout(self.block_size, self.num_kv_heads, self.head_size).page_bytes

    @property
    def type_id(self):
        return f"full_attention_inquant_k4v2_v1_{self.block_size}_{self.page_size_bytes}"


# vLLM 0.9.1 dispatches on the exact spec class, not isinstance. Register
# only our new type; the built-in full-attention manager retains ownership of
# scheduling, page reuse and preemption. This private registry is why the
# extension is pinned to 0.9.1. No built-in class/function is replaced.
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, spec_manager_map
spec_manager_map[InQuantSpec] = FullAttentionManager


class InQuantWorker(Worker):
    def get_kv_cache_spec(self):
        specs = super().get_kv_cache_spec()
        result = {}
        for name, spec in specs.items():
            if type(spec) is not FullAttentionSpec or spec.use_mla:
                raise ValueError("InQuant requires ordinary full-attention cache layers")
            result[name] = InQuantSpec(block_size=spec.block_size,
                                      num_kv_heads=spec.num_kv_heads,
                                      head_size=spec.head_size,
                                      dtype=torch.uint8, use_mla=False)
        return result

    def inquant_memory_report(self):
        from .backend import InQuantImpl, WORKSPACE
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        report = {"backend": "INQUANT_PAGED_K4V2_V1", "layers": []}
        for name, layer in layers.items():
            impl = layer.impl
            if not isinstance(impl, InQuantImpl):
                raise RuntimeError(f"Layer {name} did not use InQuant")
            layout = impl.layout
            active_pages = math.ceil(impl.max_tokens_seen / layout.block_size)
            pool_bytes = sum(t.numel() * t.element_size() for t in layer.kv_cache)
            report["layers"].append({
                "name": name, "dtype": str(layer.kv_cache[0].dtype),
                "page_bytes": layout.page_bytes, "pool_bytes": pool_bytes,
                "pool_pages": pool_bytes // layout.page_bytes,
                "max_tokens_seen": impl.max_tokens_seen,
                "peak_active_page_bytes": active_pages * layout.page_bytes,
                "exact_sink_tail_bytes": impl.dense.numel() * impl.dense.element_size(),
                "bf16_equivalent_payload_bytes": impl.max_tokens_seen * layout.heads * 128 * 4,
                "prefill_calls": impl.prefill_calls, "decode_calls": impl.decode_calls,
                "seal_calls": impl.seal_calls,
            })
        for key in ("pool_bytes", "peak_active_page_bytes", "exact_sink_tail_bytes", "bf16_equivalent_payload_bytes"):
            report[key] = sum(row[key] for row in report["layers"])
        report["shared_decode_workspace_bytes"] = WORKSPACE.nbytes
        tables = self.model_runner.input_batch.block_table.block_tables
        report["scheduler_device_table_bytes"] = sum(
            t.block_table.numel() * t.block_table.element_size()
            + t.slot_mapping.numel() * t.slot_mapping.element_size() for t in tables)
        report["peak_persistent_active_cache_bytes"] = (report["peak_active_page_bytes"]
            + report["exact_sink_tail_bytes"] + WORKSPACE.nbytes + report["scheduler_device_table_bytes"])
        report["worker_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["worker_cuda_reserved_bytes"] = torch.cuda.memory_reserved()
        report["accounting"] = ("Active-page high-water including one unsealed reserved page, all page metadata, "
                                "exact sink/tail capacity, shared decode workspace and GPU scheduler tables; excludes "
                                "model weights and transient quantization tensors. CUDA peak is whole worker, "
                                "including initialization; pool reservation is reported separately.")
        return report
