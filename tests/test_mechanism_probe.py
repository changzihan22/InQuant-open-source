"""Validate a diagnostic intervention before attributing output changes to donors."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_gsm8k_mechanism_probe import DonorInterventionCache
from inquant.cache import CacheConfig, InQuantCache
from inquant.codec import CodecConfig


def test_donor_intervention_changes_only_selected_sealed_key_entries():
    torch.manual_seed(725)
    config = CacheConfig(codec=CodecConfig(donor_policy='min_error'),
                         block_size=8, residual_length=2, sink_tokens=2)
    reference = InQuantCache(config)
    oracle = DonorInterventionCache(config, restore_donors=True)
    disabled = DonorInterventionCache(config, restore_donors=False)
    keys, values = torch.randn(1, 2, 35, 16), torch.randn(1, 2, 35, 16)
    for start, end in [(0, 19)]+[(i, i+1) for i in range(19, 35)]:
        normal = reference.update(keys[:, :, start:end], values[:, :, start:end], 0)
        unchanged = disabled.update(keys[:, :, start:end], values[:, :, start:end], 0)
        oracle.update(keys[:, :, start:end], values[:, :, start:end], 0)
        for a, b in zip(normal, unchanged):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        restored_k, restored_v = oracle.materialize()
        ref_k, ref_v = reference.materialize()
        torch.testing.assert_close(restored_v, ref_v, atol=0, rtol=0)
        expected = ref_k.clone()
        offset = config.sink_tokens
        for block in oracle._layers[0].key_blocks:
            count = block.shape[2]
            donor = block.donor_channels.long().unsqueeze(-2).expand(-1, -1, count, -1)
            expected[:, :, offset:offset+count].scatter_(
                -1, donor, keys[:, :, offset:offset+count].gather(-1, donor))
            offset += count
        torch.testing.assert_close(restored_k, expected, atol=0, rtol=0)
        assert not torch.equal(restored_k, ref_k)
