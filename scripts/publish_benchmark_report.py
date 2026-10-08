#!/usr/bin/env python3
"""Publish a complete five-method benchmark report into README and a source ZIP.

Use --wait to finalize a running local campaign. Incomplete or failed runs
never replace the published result table. Raw benchmark questions stay local.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BEGIN = '<!-- BEGIN RULER AIME RESULTS -->'
END = '<!-- END RULER AIME RESULTS -->'
METHODS = [('bf16', 'BF16'), ('inquant_fused', 'InQuant K4/V2'),
           ('snapkv', 'SnapKV'), ('knorm', 'Knorm'), ('zipcache', 'ZipCache')]


def validate_report(report):
    if report['status'] != 'complete' or set(report['methods']) != {m for m, _ in METHODS}:
        raise ValueError('Publication requires a complete five-method comparison')
    groups = {}
    sample_counts = set()
    for method, _ in METHODS:
        items = report['methods'][method]
        if items['completed'] != items['expected']:
            raise ValueError(f'Incomplete method: {method}')
        for group in items['groups']:
            key = (group['dataset'], group['context_budget'])
            if key in groups.get(method, {}):
                raise ValueError('Duplicate benchmark group')
            if not group['complete'] or not group['paired_with_bf16'] or group['score_pct'] is None:
                raise ValueError('Only complete, paired groups can be published')
            if group['dataset'] == 'ruler_v1':
                if len(group['task_scores_pct']) != 13 or group['n'] % 13:
                    raise ValueError('RULER requires all 13 tasks with balanced coverage')
                sample_counts.add(group['n'] // 13)
            elif group['dataset'] == 'aime2026' and group['n'] != 30:
                raise ValueError('AIME publication requires all 30 questions')
            groups.setdefault(method, {})[key] = group
        if set(groups[method]) != {('ruler_v1', 8192), ('ruler_v1', 16384), ('ruler_v1', 32768), ('aime2026', None)}:
            raise ValueError('Expected 8K/16K/32K RULER and AIME 2026')
    if len(sample_counts) != 1:
        raise ValueError('RULER sample counts differ between methods or lengths')
    return groups, sample_counts.pop()


def render(report):
    groups, n = validate_report(report)
    scope = 'pilot' if n < 500 else 'evaluation'
    lines = [BEGIN, '### RULER v1 and AIME 2026', '',
             f'This **RULER v1 {scope}** covers all 13 classic tasks with **{n} samples per task at each context length**. '
             'AIME 2026 uses all **30 questions** with greedy decoding and an **8192-token output cap**. '
             'All methods use Qwen2.5-7B-Instruct with BF16 weights on an A100 40GB, '
             'with the Hugging Face backend (Transformers 4.53.3) and batch size 1.', '',
             '| Method | RULER 8K | RULER 16K | RULER 32K | AIME 2026 accuracy |',
             '|---|---:|---:|---:|---:|']
    for method, label in METHODS:
        row = groups[method]
        scores = [row[('ruler_v1', length)]['score_pct'] for length in (8192, 16384, 32768)]
        aime = row[('aime2026', None)]
        lines.append(f'| {label} | {scores[0]:.2f} | {scores[1]:.2f} | {scores[2]:.2f} | {aime["score_pct"]:.2f}% ({aime["all_targets_correct"]}/30) |')
    iq = groups['inquant_fused']
    bf = groups['bf16']
    iq32, bf32 = iq[('ruler_v1', 32768)], bf[('ruler_v1', 32768)]
    if all(iq[('ruler_v1', length)]['score_pct'] > groups[method][('ruler_v1', length)]['score_pct']
           for length in (8192, 16384, 32768) for method in ('snapkv', 'knorm', 'zipcache')):
        lines += ['', 'InQuant has the highest RULER score among the compressed methods at all three tested lengths.']
    lines += ['', f'At 32K, InQuant scores **{iq32["score_pct"]:.2f}**, compared with **{bf32["score_pct"]:.2f}** for BF16, '
              f'while reducing measured persistent KV and decode state by **{iq32["mean_paired_kv_reduction_pct"]:.2f}%** on average across paired requests. '
              f'On AIME, InQuant answers **{iq[("aime2026", None)]["all_targets_correct"]}/30** correctly, '
              f'compared with **{bf[("aime2026", None)]["all_targets_correct"]}/30** for BF16. '
              'The long-context result should therefore be read separately from competition-math accuracy.', '']
    lines += ['', 'RULER reports the mean of the 13 official task scores on a 0–100 scale; tasks with multiple targets receive fractional credit. '
              'AIME reports exact final-answer accuracy. One AIME question changes accuracy by 3.33 percentage points. '
              'The RULER sample size should be kept in mind when comparing small score differences.', '',
              '| Method | RULER 32K persistent KV and decode state | Mean paired reduction vs BF16 | AIME length-limit stops |',
              '|---|---:|---:|---:|']
    for method, label in METHODS:
        ruler = groups[method][('ruler_v1', 32768)]
        aime = groups[method][('aime2026', None)]
        lines.append(f'| {label} | {ruler["median_kv_mib"]:.2f} MiB | {ruler["mean_paired_kv_reduction_pct"]:.2f}% | {aime["length_limited"]}/30 |')
    lines += ['', 'Storage is the median of per-request persistent-cache measurements, including each method’s metadata and retained full-precision state. '
              'The reduction column averages each request’s reduction against its matching BF16 request. '
              'SnapKV and Knorm remove 50% of prefill tokens; ZipCache uses its 4/2-bit preset with a 40% unimportant-token fraction. '
              'These presets do not have identical cache budgets. No compression setting was tuned on these benchmark questions.', '',
              'See the [complete per-task report](benchmarks/ruler_aime2026.json) for task scores, output lengths, descriptive latency, and provenance. '
              'The [evaluation example](#ruler-and-aime-2026) reproduces the protocol. EOS stopping gives different output lengths across methods, '
              'so these runs are not used to claim a fixed-work speedup.', END, '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-dir', type=Path, required=True)
    p.add_argument('--ruler-data', type=Path, required=True)
    p.add_argument('--aime-data', type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    campaign = args.campaign_dir.resolve()
    lock = (campaign / '.publication.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        status = json.loads((campaign / 'status.json').read_text())
        if status['status'] in ('failed', 'interrupted'):
            raise SystemExit('Campaign did not complete; README and release were not updated')
        if status['status'] == 'complete' and (campaign / 'summary.json').exists():
            break
        if not args.wait:
            raise SystemExit('Campaign is still running; pass --wait to publish after completion')
        try:
            os.kill(status['pid'], 0)
        except ProcessLookupError:
            raise SystemExit('Campaign process exited without a complete report')
        time.sleep(15)
    # Rebuild from the actual complete predictions, never trust a stale summary.
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_benchmarks.py'),
                    '--data', str(args.ruler_data.resolve()), str(args.aime_data.resolve()),
                    '--results-dir', str(campaign / 'results'), '--output', str(campaign / 'summary.json')],
                   check=True, stdout=subprocess.DEVNULL)
    report = json.loads((campaign / 'summary.json').read_text())
    block = render(report)
    provenance = {}
    for name, path in [('ruler', args.ruler_data), ('aime2026', args.aime_data)]:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        provenance[name] = {key: manifest[key] for key in
            ('revision', 'dataset_fingerprint', 'jsonl_sha256', 'n_samples', 'tokenizer_fingerprint', 'scope')}
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['jsonl_sha256']:
            raise ValueError('Data file no longer matches its preparation manifest')
    report['provenance'] = provenance
    report['model'] = 'Qwen/Qwen2.5-7B-Instruct'
    report['model_revision'] = 'a09a35458c702b33eeacc393d103063234e8bc28'
    report['protocol'] = {'backend': 'transformers-reference', 'transformers_version': '4.53.3', 'batch_size': 1,
                          'ruler_config': 'configs/qwen2.5_7b_ruler.json',
                          'aime_config': 'configs/qwen2.5_7b_aime2026.json',
                          'compression_settings_tuned_on_benchmark': False}
    report_path = ROOT / 'benchmarks/ruler_aime2026.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    readme = ROOT / 'README.md'
    content = readme.read_text()
    if BEGIN in content:
        before, rest = content.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        content = before + block + '\n' + after.lstrip('\n')
    else:
        if '\n## Installation\n' not in content:
            raise ValueError('README results insertion marker is missing')
        content = content.replace('\n## Installation\n', '\n' + block + '\n## Installation\n', 1)
    readme.write_text(content)
    archive = ROOT.parent / 'InQuant-open-source-en.zip'
    subprocess.run([sys.executable, str(ROOT / 'scripts/package_source.py'), '--output', str(archive)], check=True)
    with zipfile.ZipFile(archive) as source:
        (ROOT / 'SOURCE_MANIFEST.json').write_bytes(source.read('InQuant/SOURCE_MANIFEST.json'))
    result = {'status': 'published_locally', 'readme': str(readme), 'report': str(report_path),
              'archive': str(archive), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (campaign / 'publication.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
