from types import SimpleNamespace as NS
import pytest
import torch


def test_plugin_is_opt_in_and_version_checked(monkeypatch):
    import inquant_vllm
    monkeypatch.delenv('INQUANT_VLLM', raising=False)
    assert inquant_vllm.register() is None
    monkeypatch.setenv('INQUANT_VLLM', '1')
    monkeypatch.setattr(inquant_vllm.importlib.metadata, 'version', lambda _: '0.9.1')
    assert inquant_vllm.register() == 'inquant_vllm.platform.InQuantPlatform'
    inquant_vllm.check_version()
    monkeypatch.setattr(inquant_vllm.importlib.metadata, 'version', lambda _: '0.10.0')
    with pytest.raises(RuntimeError, match='requires vllm'):
        inquant_vllm.check_version()


@pytest.mark.parametrize('section,field,value', [
    ('model_config', 'enforce_eager', False),
    ('scheduler_config', 'max_num_seqs', 2),
    ('scheduler_config', 'max_num_batched_tokens', 1024),
    ('cache_config', 'enable_prefix_caching', True),
    ('parallel_config', 'tensor_parallel_size', 2),
    ('cache_config', 'cache_dtype', 'fp8'),
    ('cache_config', 'block_size', 16),
])
def test_unsupported_config_is_rejected(monkeypatch, section, field, value):
    from inquant_vllm.platform import CudaPlatform, InQuantPlatform
    from vllm import envs
    monkeypatch.setattr(CudaPlatform, 'check_and_update_config', classmethod(lambda cls, config: None))
    monkeypatch.setattr(envs, 'VLLM_USE_V1', True)
    config = NS(
        model_config=NS(hf_config=NS(architectures=['Qwen2ForCausalLM']),
                        dtype=torch.bfloat16, enforce_eager=True, max_model_len=4096, enable_sleep_mode=False),
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=4096,
                            long_prefill_token_threshold=0, enable_chunked_prefill=True),
        cache_config=NS(enable_prefix_caching=False, cache_dtype='auto',
                        block_size=64, cpu_offload_gb=0, sliding_window=None),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           data_parallel_size=1, worker_cls='vllm.v1.worker.gpu_worker.Worker'),
        speculative_config=None, kv_transfer_config=None)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match='InQuant vLLM 0.1 requires'):
        InQuantPlatform.check_and_update_config(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('direct_write', ['0', '1'])
def test_prefill_decode_page_boundary_request_reset_and_stale_tail(monkeypatch, direct_write):
    monkeypatch.setenv('INQUANT_DIRECT_WRITE', direct_write)
    monkeypatch.setenv('INQUANT_DONOR_BACKEND', 'triton')
    import inquant_vllm.backend as backend
    from inquant.triton_attention import SharedDecodeWorkspace
    from inquant_vllm.layout import pack_pages, unpack_pages_reference
    monkeypatch.setattr(backend, 'WORKSPACE', SharedDecodeWorkspace())
    monkeypatch.setattr(backend, 'get_current_vllm_config', lambda: NS(
        cache_config=NS(block_size=64), model_config=NS(max_model_len=512)))
    impl = backend.InQuantImpl(28, 128, 128 ** -0.5, 4, None, None, 'auto')
    pool = torch.full((12, impl.layout.page_bytes), 255, dtype=torch.uint8, device='cuda')
    blocks = torch.tensor([7, 2, 9, 3, 5, 8, 6, 1], dtype=torch.int32, device='cuda')
    torch.manual_seed(99)
    for request, initial in [('one-token', 1), ('boundary', 63), ('reused-pages', 257)]:
        k, v = torch.randn(2, initial + 3, 4, 128, dtype=torch.bfloat16, device='cuda')
        q = torch.randn(initial + 3, 28, 128, dtype=torch.bfloat16, device='cuda')
        starts = torch.tensor([0, initial], dtype=torch.int32, device='cuda')
        meta = backend.InQuantMetadata(initial, initial, 0, starts, blocks, request)
        actual = impl.forward(None, q[:initial], k[:initial], v[:initial], pool, meta,
                              torch.empty_like(q[:initial]))
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[:initial].transpose(0, 1).unsqueeze(0), k[:initial].transpose(0, 1).unsqueeze(0),
            v[:initial].transpose(0, 1).unsqueeze(0), is_causal=True, enable_gqa=True)[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        for step in range(3):
            index = initial + step
            old_pages = index // 64
            kr, vr = k[:index + 1].transpose(0, 1).clone(), v[:index + 1].transpose(0, 1).clone()
            if old_pages:
                kp = kr[:, :old_pages * 64].reshape(4, old_pages, 64, 128).transpose(0, 1)
                vp = vr[:, :old_pages * 64].reshape(4, old_pages, 64, 128).transpose(0, 1)
                kd, vd = unpack_pages_reference(pack_pages(kp, vp, impl.layout), impl.layout)
                kr[:, :old_pages * 64] = kd.transpose(0, 1).reshape(4, -1, 128)
                vr[:, :old_pages * 64] = vd.transpose(0, 1).reshape(4, -1, 128)
                kr[:, :4], vr[:, :4] = k[:4].transpose(0, 1), v[:4].transpose(0, 1)
            starts = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
            meta = backend.InQuantMetadata(1, index + 1, index, starts, blocks, request)
            actual = impl.forward(None, q[index:index + 1], k[index:index + 1], v[index:index + 1],
                                  pool, meta, torch.empty_like(q[:1]))
            expected = torch.nn.functional.scaled_dot_product_attention(
                q[index:index + 1].transpose(0, 1).unsqueeze(0), kr.unsqueeze(0), vr.unsqueeze(0),
                enable_gqa=True)[0].transpose(0, 1)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    meta.request_id = 'wrong-request'
    with pytest.raises(RuntimeError, match='state mismatch'):
        impl.forward(None, q[:1], k[:1], v[:1], pool, meta, torch.empty_like(q[:1]))
