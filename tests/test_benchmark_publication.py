import copy
import importlib.util
from pathlib import Path

import pytest
from inquant.benchmarks import RULER_TASKS

spec = importlib.util.spec_from_file_location('publish_benchmark_report', Path(__file__).parents[1] / 'scripts/publish_benchmark_report.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def complete_fixture():
    report = {'status': 'complete', 'methods': {}}
    for method, _ in module.METHODS:
        groups = []
        for context in (8192, 16384, 32768, None):
            ruler = context is not None
            groups.append({'dataset': 'ruler_v1' if ruler else 'aime2026',
                           'context_budget': context, 'complete': True, 'paired_with_bf16': True,
                           'n': 65 if ruler else 30, 'score_pct': 50.0,
                           'task_scores_pct': {task: 50.0 for task in RULER_TASKS} if ruler else {'aime2026': 50.0},
                           'all_targets_correct': 15, 'median_kv_mib': 400.0,
                           'mean_paired_kv_reduction_pct': 75.0, 'length_limited': 2})
        report['methods'][method] = {'completed': 225, 'expected': 225, 'groups': groups}
    return report


def test_public_table_includes_all_methods_and_small_sample_scope():
    text = module.render(complete_fixture())
    assert '5 samples per task' in text
    assert '**RULER v1 pilot**' in text
    assert '50.00% (15/30)' in text
    assert 'One AIME question changes accuracy by 3.33' in text
    for _, name in module.METHODS:
        assert name in text
    assert 'fixed-work speedup' in text


@pytest.mark.parametrize('failure', ['partial', 'missing_method', 'missing_length', 'unpaired', 'aime_subset'])
def test_refuses_incomplete_or_unpaired_publication(failure):
    report = copy.deepcopy(complete_fixture())
    if failure == 'partial':
        report['status'] = 'incomplete'
    elif failure == 'missing_method':
        del report['methods']['zipcache']
    elif failure == 'missing_length':
        report['methods']['bf16']['groups'].pop(0)
    elif failure == 'unpaired':
        report['methods']['knorm']['groups'][0]['paired_with_bf16'] = False
    elif failure == 'aime_subset':
        report['methods']['bf16']['groups'][-1]['n'] = 29
    with pytest.raises(ValueError):
        module.render(report)
