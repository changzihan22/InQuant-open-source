#!/usr/bin/env python3
"""Paired, seeded math subset (or full sets) for the vLLM extension.

Run each method in a fresh process. All raw answers and token IDs are retained.
This does not overwrite frozen HF experiments or assert full-suite acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inquant.evaluation import fingerprint, grade, gsm_number, last_boxed, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['bf16', 'inquant'], required=True)
    parser.add_argument('--limit', type=int, default=32, help='Per dataset; 0 means complete test sets')
    parser.add_argument('--selection-seed', type=int, default=20260919)
    parser.add_argument('--model', default='models/Qwen2.5-7B-Instruct')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/vllm_extension_math')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / f'{args.method}.jsonl'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}; use a new output root')
    os.environ.update(INQUANT_VLLM='1' if args.method == 'inquant' else '0',
                      VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    if importlib.metadata.version('vllm') != '0.9.1':
        raise RuntimeError('Use the vLLM 0.9.1 extension environment')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    selected = []
    dataset_hashes = {}
    for dataset in ['gsm8k', 'math500']:
        path = ROOT / f'data/{dataset}.jsonl'
        data = read_jsonl(path)
        dataset_hashes[dataset] = hashlib.sha256(path.read_bytes()).hexdigest()
        indices = list(range(len(data)))
        if args.limit:
            indices = sorted(random.Random(args.selection_seed).sample(indices, min(args.limit, len(data))))
        for index in indices:
            row = data[index]
            prompt = row['question'] + '\nSolve step by step. Put your final answer in \\boxed{...}.'
            ids = tokenizer.apply_chat_template([
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
            selected.append((row, ids))
    max_len = max(4096, max(len(ids) for _, ids in selected) + 2048)
    config = dict(model=args.model, dtype='bfloat16', kv_cache_dtype='auto',
                  enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False,
                  max_num_seqs=1, max_model_len=max_len, max_num_batched_tokens=max_len,
                  block_size=64, num_gpu_blocks_override=math.ceil(max_len / 64) + 1,
                  gpu_memory_utilization=0.8, seed=42, tensor_parallel_size=1)
    protocol = {'engine': config, 'selection_seed': args.selection_seed, 'limit_per_dataset': args.limit,
                'dataset_sha256': dataset_hashes, 'selected_ids': [(r['dataset'], r['id']) for r, _ in selected],
                'max_new_tokens': 2048, 'temperature': 0, 'ignore_eos': False,
                'grading': 'GSM8K: final boxed answer first, else anchored answer; MATH500: math-verify 0.8.0',
                'scope': 'full test sets' if args.limit == 0 else 'seeded integration subset; not full accuracy acceptance'}
    source_paths = [Path(__file__), ROOT / 'src/inquant/codec.py', ROOT / 'src/inquant/value_codec.py',
                    ROOT / 'src/inquant/triton_attention.py', ROOT / 'src/inquant/evaluation.py']
    source_paths += sorted((ROOT / 'extensions/vllm/src/inquant_vllm').glob('*.py'))
    protocol['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol['model_manifest'] = json.loads((Path(args.model) / 'inquant_download_manifest.json').read_text())
    freeze = args.output_root / f'{args.method}_freeze.json'
    freeze.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    status = {'state': 'loading', 'method': args.method, 'completed': 0, 'total': len(selected),
              'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}
    status_path = args.output_root / f'{args.method}_status.json'
    def save():
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(status_path)
    save()
    llm = None
    try:
        llm = LLM(**config)
        # Fixed independent warmup, no test answers used for tuning.
        warmup = tokenizer.apply_chat_template([{'role':'user', 'content':'Explain the steps of long division.'}],
                                               tokenize=True, add_generation_prompt=True)
        for _ in range(2):
            llm.generate([{'prompt_token_ids': warmup}], SamplingParams(temperature=0, max_tokens=512,
                                                                       ignore_eos=True, seed=42), use_tqdm=False)
        status['state'] = 'running'; save()
        with target.open('x') as stream:
            for row, ids in selected:
                start = time.perf_counter()
                output = llm.generate([{'prompt_token_ids': ids}],
                                     SamplingParams(temperature=0, max_tokens=2048, seed=42), use_tqdm=False)[0].outputs[0]
                elapsed = time.perf_counter() - start
                raw_grade = grade(row['dataset'], output.text, row['answer'])
                correct = raw_grade
                if row['dataset'] == 'gsm8k' and last_boxed(output.text) is not None:
                    actual = gsm_number(last_boxed(output.text), gold=True)
                    correct = actual is not None and actual == gsm_number(row['answer'], gold=True)
                result = {'method': args.method, 'dataset': row['dataset'], 'id': row['id'],
                          'answer': row['answer'], 'prediction': output.text, 'correct': bool(correct),
                          'legacy_grader_correct': bool(raw_grade), 'input_tokens': len(ids),
                          'output_tokens': len(output.token_ids), 'output_token_ids': list(output.token_ids),
                          'prompt_sha256': fingerprint(ids), 'wall_seconds': elapsed,
                          'finish_reason': output.finish_reason, 'protocol_sha256': fingerprint(protocol)}
                stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
                status['completed'] += 1
                status['last_case'] = [row['dataset'], row['id']]
                save()
        memory = llm.collective_rpc('inquant_memory_report') if args.method == 'inquant' else None
        rows = read_jsonl(target)
        summary = {'method': args.method, 'protocol': protocol, 'worker_memory_reports': memory,
                   'environment': {p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers', 'math-verify']},
                   'datasets': {}}
        for dataset in ['gsm8k', 'math500']:
            part = [r for r in rows if r['dataset'] == dataset]
            summary['datasets'][dataset] = {'correct': sum(r['correct'] for r in part), 'total': len(part),
                'absolute_accuracy': sum(r['correct'] for r in part) / len(part),
                'total_wall_seconds': sum(r['wall_seconds'] for r in part),
                'output_tokens': sum(r['output_tokens'] for r in part)}
        (args.output_root / f'{args.method}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        status['state'] = 'complete'; save()
    except Exception as exc:
        status.update(state='failed', error=repr(exc)); save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
