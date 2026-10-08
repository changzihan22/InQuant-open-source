#!/usr/bin/env python3
"""Check that a local Qwen2/Mistral checkpoint is complete without loading its weights."""
import argparse
import json
from pathlib import Path


def inspect_checkpoint(directory):
    root = Path(directory).expanduser().resolve()
    if not (root / 'config.json').is_file():
        raise ValueError(f'No config.json in {root}')
    config = json.loads((root / 'config.json').read_text())
    family = config.get('model_type')
    if family not in ('qwen2', 'mistral'):
        raise ValueError(f'Expected dense Qwen2 or Mistral, found {family!r}')
    if family == 'mistral' and config.get('sliding_window') is not None:
        raise ValueError('Sliding-window Mistral is not supported by the fused adapter; keep its native configuration')
    if not any((root / name).is_file() for name in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('Tokenizer vocabulary is missing: need tokenizer.json or tokenizer.model')
    indexes = [root / name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
               if (root / name).is_file()]
    if indexes:
        names = sorted(set(json.loads(indexes[0].read_text())['weight_map'].values()))
    else:
        names = [name for name in ('model.safetensors', 'pytorch_model.bin') if (root / name).is_file()]
    if not names:
        raise ValueError('No complete weight file or shard index found; a config-only cache cannot run inference')
    files = []
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty weight shard: {name}')
        if path.suffix == '.safetensors':
            from safetensors import SafetensorError, safe_open
            try:
                with safe_open(path, framework='pt', device='cpu') as tensors:
                    if not tensors.keys():
                        raise ValueError(f'No tensors in shard: {name}')
            except SafetensorError as error:
                raise ValueError(f'Invalid safetensors shard {name}: {error}') from error
        files.append({'file': name, 'bytes': path.stat().st_size})
    return {'path': str(root), 'model_type': family, 'architecture': config.get('architectures'),
            'max_position_embeddings': config.get('max_position_embeddings'),
            'sliding_window': config.get('sliding_window'), 'weight_files': files,
            'weight_bytes': sum(entry['bytes'] for entry in files),
            'validation': 'local file completeness and safetensors headers; not an inference or quality test'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.model)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Checkpoint check failed: {error}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
