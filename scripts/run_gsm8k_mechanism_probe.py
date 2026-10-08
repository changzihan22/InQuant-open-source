#!/usr/bin/env python3
"""Post-hoc causal probes on selected gains/losses; NOT a benchmark or optimized cache.

The donor oracle stores original K in extra memory, restoring donor entries only.
Compare it to the same SDPA implementation, not directly to a different kernel.
"""
import argparse
import contextlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from inquant.cache import CacheConfig, InQuantCache
from inquant.codec import CodecConfig
from inquant.evaluation import fingerprint, grade, read_jsonl
from inquant.qwen_fused import enable_fused_qwen2

# Frozen before running this probe; selected from the already inspected outputs.
# Six recorded gains and six recorded losses, excluding the identified parser bug.
CASES = ['28', '70', '46', '98', '92', '369', '172', '1217', '580', '1263', '1314', '1312']
METHODS = ['bf16', 'latest_fused', 'latest_sdpa', 'donor_exact_sdpa']


class DonorInterventionCache(InQuantCache):
    """Diagnostic sidecar; restore actual pre-quantization K donor values only."""
    def __init__(self, config, *, restore_donors):
        super().__init__(config)
        self.restore_donors = restore_donors
        self.original_keys = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        old = self.original_keys.get(layer_idx)
        self.original_keys[layer_idx] = key_states.clone() if old is None else torch.cat((old, key_states), dim=2)
        return result

    def materialize(self, layer_idx=0):
        key, value = super().materialize(layer_idx)
        if not self.restore_donors:
            return key, value
        layer = self._layers[layer_idx]
        offset = 0 if layer.sink_key is None else layer.sink_key.shape[2]
        original = self.original_keys[layer_idx]
        assert original.shape[2] == key.shape[2]
        for block in layer.key_blocks:
            length = block.shape[2]
            indices = block.donor_channels.long().unsqueeze(-2).expand(-1, -1, length, -1)
            exact = original[:, :, offset:offset+length].gather(-1, indices)
            key[:, :, offset:offset+length].scatter_(-1, indices, exact)
            offset += length
        return key, value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int, choices=[0, 1], required=True)
    p.add_argument('--output-root', default='outputs/gsm8k_mechanism_probe')
    p.add_argument('--model', default='models/Qwen2.5-7B-Instruct')
    args = p.parse_args()
    root = (ROOT / args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'shard{args.shard}.jsonl'
    if target.exists():
        raise FileExistsError(target)
    status_path = root / f'shard{args.shard}_status.json'
    config_path = ROOT / 'configs/qwen2.5_7b_latest_k4v2.json'
    config = json.loads(config_path.read_text())
    status = {'state': 'loading', 'started_at': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cases': CASES[args.shard::2], 'methods': METHODS, 'completed': 0,
              'scope': 'Post-hoc selected examples, no generalization or performance claim.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    def save():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(status_path)
    save()
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa').to('cuda').eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    base_config = CacheConfig(codec=CodecConfig(sample_stride=config['sample_stride'],
                                               salient_fraction=config['salient_fraction'],
                                               donor_policy=config['donor_policy']),
                              **{key: config[key] for key in ('block_size', 'residual_length', 'sink_tokens',
                                   'shared_workspace', 'value_bits', 'value_group_size', 'track_peak_bytes',
                                   'validate_positions')})
    data = {r['id']: r for r in read_jsonl(ROOT / 'data/gsm8k.jsonl')}
    archived = {
        'bf16': {r['id']: r for r in read_jsonl(ROOT / 'outputs/formal_baselines/bf16_gsm8k.jsonl')},
        'latest_fused': {r['id']: r for r in read_jsonl(ROOT / 'outputs/formal_inquant_latest/gsm8k/inquant_fused_gsm8k.jsonl')},
    }
    status['state'] = 'running';save()
    try:
        with target.open('x') as stream, torch.inference_mode():
            for case in status['cases']:
                row = data[case]
                prompt = row['question'] + '\nSolve step by step. Put your final answer in \\boxed{...}.'
                ids = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
                assert fingerprint(ids) == archived['bf16'][case]['prompt_sha256']
                effective = base_config
                short = config['short_context']
                if len(ids)+config['max_new_tokens'] <= short['max_context_tokens']:
                    effective = replace(base_config, block_size=short['block_size'], residual_length=short['residual_length'])
                baseline_tokens, baseline_logits = [], []
                for method in METHODS:
                    started = time.perf_counter()
                    if method == 'bf16':
                        cache = DynamicCache()
                    elif method == 'latest_fused':
                        cache = InQuantCache(effective)
                    else:
                        cache = DonorInterventionCache(effective, restore_donors=(method == 'donor_exact_sdpa'))
                    tokens, first_difference = [], None
                    context = enable_fused_qwen2(model) if method == 'latest_fused' else contextlib.nullcontext()
                    with context:
                        for step in range(config['max_new_tokens']):
                            if step == 0:
                                out = model(input_ids=torch.tensor([ids], device='cuda'),
                                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                            else:
                                if tokens[-1] in eos:
                                    break
                                position = torch.tensor([len(ids)+step-1], device='cuda')
                                out = model(input_ids=token, past_key_values=cache, use_cache=True,
                                            position_ids=position.unsqueeze(0), cache_position=position, logits_to_keep=1)
                            logits = out.logits[0, -1]
                            token = logits.argmax().reshape(1, 1)
                            chosen = int(token.item())
                            tokens.append(chosen)
                            if method == 'bf16':
                                baseline_tokens.append(chosen)
                                baseline_logits.append(logits.detach().cpu())
                            elif first_difference is None and step < len(baseline_tokens) and chosen != baseline_tokens[step]:
                                expected = baseline_tokens[step]
                                base = baseline_logits[step].float()
                                current = logits.float()
                                first_difference = {
                                    'generated_token_index_zero_based': step,
                                    'common_prefix_generated_tokens': step,
                                    'bf16_token_id': expected, 'probe_token_id': chosen,
                                    'bf16_token': tokenizer.decode([expected]), 'probe_token': tokenizer.decode([chosen]),
                                    'bf16_margin_bf16_minus_probe': float(base[expected]-base[chosen]),
                                    'probe_margin_bf16_minus_probe': float((current[expected]-current[chosen]).item()),
                                    'layer0_sealed_blocks_after_step': len(cache._layers[0].key_blocks),
                                }
                            del out, logits
                    prediction = tokenizer.decode(tokens, skip_special_tokens=True)
                    record = {'id': case, 'method': method, 'question': row['question'], 'answer': row['answer'],
                              'correct': grade('gsm8k', prediction, row['answer']), 'prediction': prediction,
                              'token_ids': tokens, 'generated_tokens': len(tokens), 'input_tokens': len(ids),
                              'prompt_sha256': fingerprint(ids), 'config': asdict(effective),
                              'first_difference_from_bf16': first_difference,
                              'diagnostic_wall_s_not_benchmark': time.perf_counter()-started}
                    if method in archived:
                        record['matches_archived_prediction'] = prediction == archived[method][case]['prediction']
                        record['matches_archived_correctness'] = record['correct'] == archived[method][case]['correct']
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n');stream.flush()
                    status['completed'] += 1
                    status['last'] = {'id': case, 'method': method, 'correct': record['correct']}
                    save()
                    print(json.dumps(status['last']), flush=True)
                    del cache
                del baseline_logits
        status['state'] = 'complete';save()
    except Exception as error:
        status['state'] = 'failed';status['error'] = repr(error);save();raise


if __name__ == '__main__':
    main()
