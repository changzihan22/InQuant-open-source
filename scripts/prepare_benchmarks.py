#!/usr/bin/env python3
"""Prepare pinned RULER v1 workloads and held-out AIME 2026 for run_eval.py.

Data is downloaded separately and is not covered by the source-code MIT license.
RULER calls the unmodified upstream generators with the model's native chat
wrapper and official answer prefix. Sequence budgets include input and output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inquant.benchmarks import AIME_REVISION, RULER_REVISION, RULER_TASKS, tokenizer_fingerprint, ruler_prompt
from inquant.evaluation import fingerprint, read_jsonl


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    import requests
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(20, 90))
            response.raise_for_status()
            if not response.content:
                raise ValueError(f'Empty download: {url}')
            temporary = path.with_suffix(path.suffix + '.part')
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return
        except requests.RequestException:
            if attempt == 2:
                raise


def verify_upstream(path):
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != RULER_REVISION:
        raise ValueError(f'Expected NVIDIA/RULER {RULER_REVISION}; got {revision}')
    changes = subprocess.check_output(['git', '-C', str(path), 'diff', '--name-only', 'HEAD'], text=True)
    for changed in changes.splitlines():
        if changed == 'scripts/data/synthetic/json/english_words.json':
            pointer = subprocess.check_output(['git', '-C', str(path), 'show', 'HEAD:' + changed], text=True)
            digest = pointer.split('oid sha256:', 1)[1].splitlines()[0]
            if sha256(path / changed) == digest:
                continue
        raise ValueError(f'Use an unmodified RULER checkout: {changed}')


def fetch(args):
    import html2text
    from bs4 import BeautifulSoup
    import zipfile
    verify_upstream(args.ruler_repo)
    corpus = args.ruler_repo / 'scripts/data/synthetic/json'
    words = corpus / 'english_words.json'
    if words.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
        pointer = words.read_text()
        expected = pointer.split('oid sha256:', 1)[1].splitlines()[0]
        asset = args.cache_dir / 'english_words.json'
        download(f'https://media.githubusercontent.com/media/NVIDIA/RULER/{RULER_REVISION}/scripts/data/synthetic/json/english_words.json', asset)
        if sha256(asset) != expected:
            raise ValueError('RULER Git LFS word-list checksum mismatch')
        words.write_bytes(asset.read_bytes())
    json.loads(words.read_text())
    raw = args.cache_dir / 'essays' 
    urls = (corpus / 'PaulGrahamEssays_URLs.txt').read_text().splitlines()

    def fetch_essay(url):
        # Some hosts now require HTTPS. The source list and article identity
        # remain those in the pinned RULER revision.
        resolved = url.replace('http://www.paulgraham.com/', 'https://www.paulgraham.com/')
        if 'github.com/gkamradt/' in resolved:
            resolved = resolved.replace('https://github.com/', 'https://raw.githubusercontent.com/').replace('/raw/main/', '/main/')
        name = resolved.rsplit('/', 1)[-1]
        path = raw / name
        download(resolved, path)
        if name.endswith('.html'):
            h = html2text.HTML2Text()
            h.ignore_images = h.ignore_tables = h.escape_all = True
            h.reference_links = h.mark_code = False
            # Match the upstream downloader's decoding and text conversion.
            soup = BeautifulSoup(path.read_bytes().decode('unicode_escape'), 'html.parser')
            font = soup.find('font')
            if font is None:
                raise ValueError(f'Missing essay content: {resolved}')
            content = h.handle(str(font))
        else:
            content = path.read_text()
        return {'url': url, 'resolved_url': resolved, 'sha256': sha256(path),
                'name': name, 'content': content}

    with ThreadPoolExecutor(max_workers=8) as pool:
        essays = list(pool.map(fetch_essay, urls))
    essays.sort(key=lambda e: (e['name'].endswith('.html'), e['name']))
    (corpus / 'PaulGrahamEssays.json').write_text(json.dumps({'text': ''.join(e.pop('content') for e in essays)}))
    sources = {
        'squad.json': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
        'hotpotqa.json': 'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json',
    }
    for name, url in sources.items():
        download(url, corpus / name)
        json.loads((corpus / name).read_text())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Manual installation of a pinned, plain-text tokenizer resource. No
    # pickle models, downloader index, or downloader security overrides.
    nltk_revision = '550b6625bcef1f2abff2ff770a5a0d272c9c6b2a'
    nltk_url = f'https://raw.githubusercontent.com/nltk/nltk_data/{nltk_revision}/packages/tokenizers/punkt_tab.zip'
    punkt_zip = args.cache_dir / 'punkt_tab.zip'
    download(nltk_url, punkt_zip)
    target = args.cache_dir / 'nltk_data/tokenizers'
    with zipfile.ZipFile(punkt_zip) as archive:
        for member in archive.infolist():
            if member.filename.startswith('punkt_tab/english/') and not member.is_dir():
                relative = Path(member.filename)
                if '..' in relative.parts or relative.is_absolute() or relative.suffix not in ('.tab', '.txt'):
                    raise ValueError('Unexpected tokenizer resource member')
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    manifest = {'ruler_revision': RULER_REVISION, 'nltk_data_revision': nltk_revision, 'punkt_tab_sha256': sha256(punkt_zip), 'essays': essays, 'qa_sources': sources,
                'corpus_sha256': {p.name: sha256(p) for p in sorted(corpus.glob('*.json'))},
                'dependencies': {name: version(name) for name in ('wonderwords', 'nltk', 'numpy', 'scipy', 'html2text', 'transformers')}}
    (args.cache_dir / 'corpus_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'ready', 'essays': len(essays), 'cache_dir': str(args.cache_dir)}))


def write_dataset(rows, output, manifest):
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'Refusing to replace prepared data: {output}')
    digest = fingerprint({'manifest': manifest, 'rows': rows})
    for row in rows:
        row['dataset_fingerprint'] = digest
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest.update({'dataset_fingerprint': digest, 'jsonl_sha256': sha256(output), 'n_samples': len(rows)})
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'samples': len(rows), 'sha256': sha256(output)}), flush=True)


def ruler(args):
    import yaml
    from transformers import AutoTokenizer
    verify_upstream(args.ruler_repo)
    corpus_manifest = json.loads((args.cache_dir / 'corpus_manifest.json').read_text())
    for name, digest in corpus_manifest['corpus_sha256'].items():
        if sha256(args.ruler_repo / 'scripts/data/synthetic/json' / name) != digest:
            raise ValueError(f'Corpus changed: {name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok_hash = tokenizer_fingerprint(tokenizer)
    base = runpy.run_path(str(args.ruler_repo / 'scripts/data/synthetic/constants.py'))['TASKS']
    tasks = yaml.safe_load((args.ruler_repo / 'scripts/synthetic.yaml').read_text())
    raw_dir = args.output.parent / (args.output.stem + '_upstream')
    env = {**os.environ, 'NLTK_DATA': str(args.cache_dir / 'nltk_data'), 'PYTHONHASHSEED': '0',
           'TOKENIZERS_PARALLELISM': 'false'}
    def prepare_one(pair):
        length, task = pair
        job_rows = []
        config = tasks[task]
        spec = base[config['task']]
        template = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': spec['template']}],
            tokenize=False, add_generation_prompt=True) + spec.get('answer_prefix', '')
        folder = raw_dir / str(length)
        command = [sys.executable, str(args.ruler_repo / 'scripts/data/synthetic' / (config['task'] + '.py')),
                   '--save_dir', str(folder), '--save_name', task, '--subset', 'validation',
                   '--tokenizer_path', args.model, '--tokenizer_type', 'hf',
                   '--max_seq_length', str(length), '--tokens_to_generate', str(spec['tokens_to_generate']),
                   '--num_samples', str(args.samples_per_task), '--random_seed', str(args.seed), '--template', template]
        for key, value in config['args'].items():
            command += ['--' + key, str(value)]
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / (task + '.log')).open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw_rows = read_jsonl(folder / task / 'validation.jsonl')
        if len(raw_rows) != args.samples_per_task:
            raise ValueError(f'Wrong sample count for {length}/{task}')
        for raw in raw_rows:
            tokens = tokenizer.encode(ruler_prompt(raw), add_special_tokens=False)
            if len(tokens) + spec['tokens_to_generate'] > length:
                raise ValueError(f'Upstream generator exceeded the context budget: {length}/{task}/{raw["index"]}')
            job_rows.append({'dataset': 'ruler_v1', 'id': f'{length}/{task}/{raw["index"]}',
                             'split': 'synthetic', 'task': task, 'context_budget': length,
                             'input_ids': tokens, 'answer': raw['outputs'],
                             'max_new_tokens': spec['tokens_to_generate'],
                             'tokenizer_model': args.model, 'tokenizer_fingerprint': tok_hash,
                             'benchmark_source': {'repo': 'NVIDIA/RULER', 'revision': RULER_REVISION},
                             'upstream_metadata': {k: v for k, v in raw.items() if k not in ('input', 'outputs')}})
        print(f'Prepared {length}/{task}: {len(raw_rows)} samples', flush=True)
        return job_rows

    pairs = [(length, task) for length in args.lengths for task in args.tasks]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        prepared = [row for group in pool.map(prepare_one, pairs) for row in group]
    manifest = {'benchmark': 'RULER v1', 'revision': RULER_REVISION, 'tasks': args.tasks,
                'context_budgets': args.lengths, 'samples_per_task': args.samples_per_task, 'seed': args.seed,
                'prompt_protocol': 'native chat user + official assistant answer prefix',
                'length_protocol': 'input tokens + official per-task output budget <= context budget; no truncation',
                'tokenizer_fingerprint': tok_hash, 'corpus_manifest': corpus_manifest,
                'scope': 'pilot' if args.samples_per_task < 500 else '500 samples per task and length'}
    write_dataset(prepared, args.output, manifest)


def aime(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    path = args.cache_dir / ('aime2026-' + AIME_REVISION + '.parquet')
    url = f'https://huggingface.co/datasets/MathArena/aime_2026/resolve/{AIME_REVISION}/data/train-00000-of-00001.parquet'
    download(url, path)
    records = pq.read_table(path).to_pylist()
    if len(records) != 30 or len({r['problem_idx'] for r in records}) != 30:
        raise ValueError('Expected the complete 30-question AIME 2026 dataset')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok_hash = tokenizer_fingerprint(tokenizer)
    rows = []
    for record in records:
        question = record['problem'] + '\nSolve step by step. Put your final answer, an integer from 0 to 999, in \\boxed{...}.'
        tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True, add_generation_prompt=True)
        rows.append({'dataset': 'aime2026', 'id': str(record['problem_idx']), 'split': 'held_out_competition',
                     'input_ids': tokens, 'answer': str(record['answer']), 'max_new_tokens': args.max_new_tokens,
                     'tokenizer_model': args.model, 'tokenizer_fingerprint': tok_hash,
                     'benchmark_source': {'repo': 'MathArena/aime_2026', 'revision': AIME_REVISION, 'hub_split': 'train'}})
    write_dataset(rows, args.output, {'benchmark': 'AIME 2026', 'revision': AIME_REVISION,
                  'source_url': url, 'source_sha256': sha256(path), 'hub_split': 'train',
                  'evaluation_role': 'held-out competition questions; no calibration or tuning',
                  'license': 'CC-BY-NC-SA-4.0', 'max_new_tokens': args.max_new_tokens,
                  'tokenizer_fingerprint': tok_hash, 'scope': 'all 30 questions; greedy pass@1'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name, function in [('fetch-ruler', fetch), ('ruler', ruler), ('aime2026', aime)]:
        sub = commands.add_parser(name)
        sub.set_defaults(function=function)
        sub.add_argument('--cache-dir', type=Path, default=Path('data/benchmark_cache'))
        if name != 'aime2026':
            sub.add_argument('--ruler-repo', type=Path, required=True)
        if name != 'fetch-ruler':
            sub.add_argument('--model', required=True)
            sub.add_argument('--revision', default='a09a35458c702b33eeacc393d103063234e8bc28')
            sub.add_argument('--output', type=Path, required=True)
        if name == 'ruler':
            sub.add_argument('--lengths', nargs='+', type=int, default=[8192, 16384, 32768])
            sub.add_argument('--tasks', nargs='+', choices=RULER_TASKS, default=list(RULER_TASKS))
            sub.add_argument('--samples-per-task', type=int, default=5)
            sub.add_argument('--seed', type=int, default=42)
            sub.add_argument('--workers', type=int, default=2)
        if name == 'aime2026':
            sub.add_argument('--max-new-tokens', type=int, default=8192)
    args = p.parse_args()
    if hasattr(args, 'samples_per_task') and (args.samples_per_task < 1 or args.workers < 1 or any(n <= 128 for n in args.lengths)):
        p.error('Sample count must be positive and context budgets must exceed 128 tokens')
    if hasattr(args, 'max_new_tokens') and args.max_new_tokens < 1:
        p.error('Generation budget must be positive')
    for name in ('cache_dir', 'ruler_repo', 'output'):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    if hasattr(args, 'output') and (args.output.exists() or args.output.with_suffix('.manifest.json').exists()):
        p.error('Prepared output already exists; choose a new filename')
    args.function(args)


if __name__ == '__main__':
    main()
