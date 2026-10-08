"""Incremental metadata must preserve outputs through growth and reject stale prefixes."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_growing_state_matches_fresh_state_and_rejects_replacement():
    from inquant.codec import quantize, CodecConfig
    from inquant.value_codec import quantize_values
    from inquant.triton_attention import PackedDecodeState
    torch.manual_seed(831)
    keys, values = [], []
    state = PackedDecodeState([], [])
    q = torch.randn(1, 8, 1, 128, device='cuda', dtype=torch.bfloat16)
    for i in range(11):
        k = torch.randn(1, 2, 32, 128, device='cuda', dtype=torch.bfloat16)
        # Exercise source/donor indices at the signed-byte boundary.
        k[..., 127] *= 20
        keys.append(quantize(k, CodecConfig(donor_policy='min_error')))
        values.append(quantize_values(torch.randn_like(k)))
        state.extend(keys, values)
        fresh = PackedDecodeState(keys, values)
        torch.testing.assert_close(state.decode(q), fresh.decode(q), atol=0, rtol=0)
        assert state._key_map.dtype == torch.int8
        torch.testing.assert_close(state._key_map[:i+1], fresh._key_map[:i+1], atol=0, rtol=0)
        assert state.nbytes == sum(x.untyped_storage().nbytes() for x in
            (state._tables, state._key_map, state._value_map, state._workspace) if x is not None)
    with pytest.raises(ValueError, match='append-only'):
        state.extend(keys[:-1], values[:-1])
    with pytest.raises(ValueError, match='append-only'):
        state.extend(list(reversed(keys)), values)
    state.extend(keys, values)
    torch.testing.assert_close(state.decode(q), fresh.decode(q), atol=0, rtol=0)
