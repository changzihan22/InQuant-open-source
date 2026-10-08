"""Catch incomplete local model caches before spending GPU time on loading."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location('check_local_model', Path(__file__).parents[1] / 'scripts/check_local_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_shard_and_complete_header(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'mistral', 'sliding_window': None}))
    (tmp_path / 'tokenizer.json').write_text('{}')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'part.safetensors'}}))
    with pytest.raises(ValueError, match='Missing or empty weight shard'):
        module.inspect_checkpoint(tmp_path)
    save_file({'weight': torch.zeros(2, 2)}, tmp_path / 'part.safetensors')
    assert module.inspect_checkpoint(tmp_path)['weight_bytes'] > 0
    raw = (tmp_path / 'part.safetensors').read_bytes()
    (tmp_path / 'part.safetensors').write_bytes(raw[:-1])
    with pytest.raises(ValueError, match='Invalid safetensors shard'):
        module.inspect_checkpoint(tmp_path)


def test_config_only_and_windowed_mistral(tmp_path):
    config = {'model_type': 'mistral', 'sliding_window': None}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Tokenizer vocabulary'):
        module.inspect_checkpoint(tmp_path)
    config['sliding_window'] = 4096
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Sliding-window'):
        module.inspect_checkpoint(tmp_path)
