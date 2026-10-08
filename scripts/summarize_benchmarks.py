#!/usr/bin/env python3
"""Summarize paired RULER/AIME runs, retaining incomplete and failed protocols."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inquant.benchmarks import RULER_TASKS, score_row
from inquant.evaluation import PROTOCOL_KEYS, fingerprint, read_jsonl

METHODS = ('bf16', 'inquant_fused', 'snapkv', 'knorm', 'zipcache')


def summarize(data, results, methods=METHODS):
    expected = {(r['dataset'], r['id'], 0): r for r in data}
    if not expected or len(expected) != len(data):
        raise ValueError('Expected unique nonempty benchmark data')
    grouped = defaultdict(dict)
    for row in results:
        key = (row['dataset'], row['id'], row['repeat'])
        if row['method'] not in methods or key not in expected:
            raise ValueError(f'Unexpected result: {row["method"]}/{key}')
        if key in grouped[row['method']]:
            raise ValueError(f'Duplicate result: {row["method"]}/{key}')
        original = expected[key]
        if row.get('status') != 'ok' or row.get('score') != score_row(original, row['prediction']):
            raise ValueError(f'Failed status or incorrect saved score: {key}')
        if row.get('dataset_fingerprint') != original['dataset_fingerprint']:
            raise ValueError(f'Dataset mismatch: {key}')
        if row.get('prompt_sha256') != fingerprint(original['input_ids']):
            raise ValueError(f'Prompt mismatch: {key}')
        if row.get('max_new_tokens') != original['max_new_tokens']:
            raise ValueError(f'Generation budget mismatch: {key}')
        if row.get('fixed_output_length') is not False:
            raise ValueError('Accuracy benchmarks require EOS stopping')
        grouped[row['method']][key] = row
    report = {'status': 'complete' if all(set(grouped[m]) == set(expected) for m in methods) else 'incomplete',
              'expected_per_method': len(expected), 'expected_methods': list(methods),
              'scope': 'RULER pilot/full status depends on samples per task; AIME complete only at 30 questions',
              'expected_counts': {dataset: sum(r['dataset'] == dataset for r in data) for dataset in sorted({r['dataset'] for r in data})},
              'latency_note': 'Descriptive batch-1 latency with EOS stopping; output lengths differ. This is not fixed-work performance acceptance.',
              'methods': {}}
    for method in methods:
        rows = grouped[method]
        details = {'completed': len(rows), 'expected': len(expected), 'groups': []}
        subsets = defaultdict(list)
        for key, row in rows.items():
            subsets[(row['dataset'], row.get('context_budget'))].append(key)
        for (dataset, context), keys in sorted(subsets.items()):
            expected_keys = {k for k, v in expected.items() if v['dataset'] == dataset and v.get('context_budget') == context}
            records = [rows[k] for k in keys]
            tasks = defaultdict(list)
            for row in records:
                tasks[row.get('task') or dataset].append(row['score'])
            task_scores = {task: round(100 * mean(values), 2) for task, values in sorted(tasks.items())}
            complete = set(keys) == expected_keys
            task_complete = dataset != 'ruler_v1' or set(tasks) == set(RULER_TASKS)
            result = {'dataset': dataset, 'context_budget': context, 'n': len(keys),
                      'expected': len(expected_keys), 'complete': complete,
                      'task_scores_pct': task_scores,
                      'score_pct': round(mean(task_scores.values()), 2) if complete and task_complete else None,
                      'all_targets_correct': sum(r['correct'] for r in records),
                      'length_limited': sum(r['finish_reason'] == 'length' for r in records),
                      'mean_output_tokens': mean(r['generated_tokens'] for r in records),
                      'input_tokens_range': [min(r['input_tokens'] for r in records), max(r['input_tokens'] for r in records)],
                      'median_e2e_s': median(r['e2e_s'] for r in records),
                      'median_kv_mib': median(r['kv_cache_bytes'] for r in records) / 2**20,
                      'median_prefill_kv_mib': median(r['prefill_kv_bytes'] for r in records) / 2**20,
                      'paired_with_bf16': False}
            baseline = grouped['bf16']
            if complete and expected_keys <= baseline.keys():
                for key in keys:
                    for field in (*PROTOCOL_KEYS, 'prompt_sha256', 'input_tokens', 'dataset_fingerprint', 'fixed_output_length', 'implementation_fingerprint'):
                        if rows[key].get(field) != baseline[key].get(field) or field not in rows[key]:
                            raise ValueError(f'Unpaired {field}: {method}/{key}')
                result.update({'paired_with_bf16': True,
                               'score_delta_pp': 100 * mean(rows[k]['score'] - baseline[k]['score'] for k in keys),
                               'mean_paired_kv_reduction_pct': 100 * mean(1 - rows[k]['kv_cache_bytes'] / baseline[k]['kv_cache_bytes'] for k in keys),
                               'median_paired_e2e_ratio': median(rows[k]['e2e_s'] / baseline[k]['e2e_s'] for k in keys)})
            details['groups'].append(result)
        report['methods'][method] = details
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = [row for path in args.data for row in read_jsonl(path)]
    results = [row for path in sorted(args.results_dir.glob('*.jsonl')) for row in read_jsonl(path)]
    report = summarize(data, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
