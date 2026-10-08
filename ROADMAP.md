# Development Roadmap

This roadmap proposes the next implementation steps around InQuant's sampling, neighbor-slot reuse, and descriptor-guided recovery design. The stages below describe the broader design direction. The runtime work listed next is already implemented; other items remain planned. Each stage should produce a usable, independently validated change.

## Runtime iteration status

The current iteration adds a deterministic GPU donor selector, incremental INT8 decode maps, and direct vLLM physical-page writes. Selection and packed page contents are checked against the Torch reference; the floating-point quantization rules and page format stay the same. This is the first runtime step toward the GPU pipeline in stage 3.

The HF adapter also accepts full-attention Mistral and includes local-checkpoint presets. CPU/CUDA small-model tests cover its GQA, native RoPE, cache sealing, and restoration. Pretrained Mistral-7B evaluation is pending complete local weights. Sliding-window Mistral and Mistral vLLM are separate follow-ups.

The next algorithm experiment should introduce `paper_neighbors_v1` alongside `min_error`, then compare both K/V mixed precision and K4/V2 on a fixed development set. Keep the current policy unless the candidate improves the accuracy/storage/latency tradeoff. Continuous batching should follow request-state isolation and lifecycle tests, not a relaxed platform guard.

## 1. Specify the group layout and slot-selection policy

Start with a versioned format and a CPU reference for the paper's Algorithms 1 and 2. Define the group size, sampled saliency score, outlier selection, tie-breaking rules, scale representation, nibble order, donor mapping, and sampled padding statistics. Make every choice explicit where the manuscript leaves implementation details open.

For isolated outliers, choose an unused low-saliency neighbor. For adjacent outlier groups, search at stride distances determined by the run length. Record aligned fallback mappings when a search crosses group boundaries. A donor must never be an outlier or be assigned twice.

Add a separate policy such as `paper_neighbors_v1` and dedicated configurations. Keep existing policies available for controlled comparisons; do not silently change the meaning of saved configurations. Include an experiment preset that applies the mixed-precision codec to both K and V, with the K4/V2 preset retained as another configuration.

**Code entry points:** `CodecConfig`, `_reuse_map`, and `quantize` in `src/inquant/codec.py`; `CacheConfig` in `src/inquant/cache.py`; `configs/`.

**Validation:** isolated outliers, adjacent runs, tied saliency, edge channels, donor exhaustion, cross-group fallbacks, and deterministic assignments. Verify that every 8-bit code can be recovered exactly before applying its quantization scale.

## 2. Introduce compact descriptors and boundary hints

Define a descriptor with an outlier bitmap, bit-width information, reuse offsets or strides, optional fallback entries, and a format version. Specify byte alignment and ownership before writing GPU kernels.

Treat markers as descriptor-scoped hints. Do not reserve an ordinary nibble value as a global sentinel or overwrite payload bits without a defined representation. Sampled padding must reconstruct donor entries separately from outlier recovery.

Add a reference decoder for the new descriptor, then teach the fused attention reader to consume it. Measure descriptor size and access cost together: smaller metadata is useful only if decoding remains efficient.

**Code entry points:** `PackedTensor` in `src/inquant/codec.py`, `_decode_tile` in `src/inquant/triton_attention.py`, and `PageLayout` in `extensions/vllm/src/inquant_vllm/layout.py`.

**Validation:** all 16 ordinary nibble patterns, marker-like payloads, malformed descriptors, group boundaries, and agreement between reference reconstruction and fused attention. Count payload, scales, descriptors, padding, and alignment separately.

## 3. Fuse quantization and page writes on the GPU

Profile saliency estimation, selection, scale calculation, donor assignment, packing, and page sealing separately. Use the profile to choose fusion boundaries rather than assuming that one large kernel will be faster.

A practical starting point is a statistics kernel for sampled saliency and padding, followed by selection/descriptor construction and a packing kernel that writes directly into the destination block or page. Scale calculation may still require a full-block reduction. Keep intermediate tensors on the device and use bounded reusable workspace.

After the reference format is stable, remove avoidable Python loops, host synchronization, and temporary copies from the hot path. Add this path behind an explicit configuration switch until it matches the reference.

**Code entry points:** `quantize`, `_quantize_temporal_blocks`, `PackedDecodeState`, and `pack_pages` in the core codec, cache, Triton attention, and vLLM layout modules.

**Validation:** compare generated descriptors and quantized codes with the reference, check attention outputs, and measure both quantization latency and complete request latency. Report peak temporary memory as well as persistent storage.

## 4. Extend the vLLM request lifecycle

Integrate direct packed-page writes before adding concurrent requests. Move request-specific sink and tail state into explicitly owned structures, and verify page allocation, reuse, and cleanup across interleaved requests.

Add continuous batching and chunked prefill with dedicated lifecycle tests. Consider CUDA Graph support only after allocation patterns and workspace addresses are stable. Handle prefix-cache sharing and copy-on-write separately. Keep version checks and unsupported-configuration guards until each path has been implemented and tested.

**Code entry points:** `backend.py`, `worker.py`, `layout.py`, and `kernels.py` under `extensions/vllm/src/inquant_vllm/`.

**Validation:** interleaved requests, cancellation, page reuse, shared prefixes, partial prefill, and stream/workspace ownership. Measure the native and compressed backends with the same request workload and engine settings.

## Suggested implementation order

1. Submit the format specification, reference slot selector, and focused correctness tests.
2. Implement compact descriptors and a matching fused reader.
3. Profile and add fused quantization/page writes.
4. Expand serving features one at a time.

For each stage, record the configuration and source revision, compare accuracy on fixed inputs, and report cache storage and latency with clear measurement scopes. Change one major design choice at a time so its effects can be identified. Set performance targets from these measurements rather than carrying over historical project thresholds.
