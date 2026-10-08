#!/usr/bin/env python3
"""Run RULER first, then AIME, with one isolated method process per GPU.

Progress is flushed per sample by run_eval.py. Re-running this command resumes
matching records. A lock prevents two campaigns writing the same output folder.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inquant.evaluation import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--ruler-data', type=Path, required=True)
    p.add_argument('--aime-data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', default=['0', '1'])
    p.add_argument('--methods', nargs='+', default=['bf16', 'inquant_fused', 'zipcache', 'snapkv', 'knorm'],
                   choices=['bf16', 'inquant_fused', 'snapkv', 'knorm', 'zipcache'])
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.methods)) != len(args.methods):
        p.error('GPU IDs and methods must be unique')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = args.output_dir / 'results'
    logs = args.output_dir / 'logs'
    results.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    lock = (args.output_dir / '.campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running', 'jobs': []}
    active = {}

    def save():
        for job in state['jobs']:
            output = Path(job['output'])
            if output.exists():
                # Count complete lines only while a process may be appending.
                job['completed'] = sum(line.endswith(b'\n') for line in output.open('rb'))
        temporary = args.output_dir / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.output_dir / 'status.json')

    try:
        for name, data, config in [('ruler', args.ruler_data, 'qwen2.5_7b_ruler.json'),
                                    ('aime2026', args.aime_data, 'qwen2.5_7b_aime2026.json')]:
            data = data.resolve()
            expected = len(read_jsonl(data))
            queue = deque(args.methods)
            while queue or active:
                for gpu in args.gpus:
                    if gpu in active or not queue:
                        continue
                    method = queue.popleft()
                    output = results / f'{name}_{method}.jsonl'
                    command = [sys.executable, str(ROOT / 'scripts/run_eval.py'),
                               '--config', str(ROOT / 'configs' / config), '--data', str(data),
                               '--method', method, '--model', args.model, '--device', 'cuda:0',
                               '--output', str(output), '--resume']
                    job = {'benchmark': name, 'method': method, 'gpu': gpu, 'expected': expected,
                           'completed': 0, 'output': str(output), 'command': command,
                           'started_at': time.time(), 'status': 'running'}
                    log = (logs / f'{name}_{method}.log').open('a')
                    process = subprocess.Popen(command, cwd=ROOT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'TOKENIZERS_PARALLELISM': 'false',
                             'PYTHONUNBUFFERED': '1'}, stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = (process, log, job)
                    state['jobs'].append(job)
                    print(f'Started {name}/{method} on GPU {gpu}, pid {process.pid}', flush=True)
                for gpu, (process, log, job) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        job.update(status='complete' if code == 0 else 'failed', exit_code=code, finished_at=time.time())
                        log.close()
                        del active[gpu]
                        print(f'{job["status"]}: {job["benchmark"]}/{job["method"]}', flush=True)
                save()
                if active:
                    time.sleep(3)
        state['status'] = 'complete' if all(j['status'] == 'complete' and j['completed'] == j['expected'] for j in state['jobs']) else 'failed'
        state['finished_at'] = time.time()
        save()
        if state['status'] == 'complete':
            subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_benchmarks.py'),
                            '--data', str(args.ruler_data.resolve()), str(args.aime_data.resolve()),
                            '--results-dir', str(results), '--output', str(args.output_dir / 'summary.json')], check=True)
    finally:
        for process, log, job in active.values():
            process.terminate()
            process.wait()
            log.close()
            job['status'] = 'interrupted'
        if active:
            state['status'] = 'interrupted'
            save()
        lock.close()
    if state['status'] != 'complete':
        raise SystemExit('Some jobs failed; inspect logs and resume after fixing the cause')


if __name__ == '__main__':
    main()
