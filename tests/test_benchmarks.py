import importlib.util
from pathlib import Path

import pytest

from inquant.benchmarks import (aime_answer, generation_budget, ruler_score,
                                 score_row, validate_resume_row)
from inquant.evaluation import fingerprint


def test_ruler_fractional_credit_and_aliases():
    assert ruler_score('ALPHA and beta', ['alpha', 'beta', 'gamma', 'delta'], 'niah_multivalue') == .5
    assert ruler_score('The city is PARIS.', ['Paris', 'City of Paris'], 'qa_1') == 1
    assert ruler_score('', ['a'], 'vt') == 0
    assert ruler_score('12345', ['234'], 'niah_single_1') == 1  # official substring semantics
    assert ruler_score('a\x01b', ['a\nb'], 'cwe') == 1
    with pytest.raises(ValueError):
        ruler_score('abc', [], 'vt')
    with pytest.raises(ValueError):
        ruler_score('abc', ['abc'], 'unrecognized')


@pytest.mark.parametrize(('text', 'answer'), [(r'Work: 17. Final: \boxed{042}', 42),
    ('Final answer: 999', 999), ('000', 0), ('42', 42), ('Intermediate value is 42', None),
    (r'\boxed{1000}', None), (r'\boxed{-1}', None), (r'\boxed{1.5}', None),
    (r'\boxed{42} then \boxed{17}', 17), ('42 or 43', None)])
def test_aime_final_integer(text, answer):
    assert aime_answer(text) == answer


def test_aime_exact_score():
    row = {'dataset': 'aime2026', 'answer': '42'}
    assert score_row(row, r'\boxed{042}') == 1
    assert score_row(row, 'I computed 42 but cannot finish') == 0


def test_budget_and_resume_reject_changed_data_or_protocol():
    config = {'max_new_tokens': 128}
    row = {'dataset': 'ruler_v1', 'max_new_tokens': 30, 'input_ids': [1, 2]}
    previous = {'status': 'ok', 'dataset_fingerprint': 'original', 'max_new_tokens': 30,
                'grading': 'ruler-v1-substring-all-or-part-v1', 'prompt_sha256': fingerprint([1, 2])}
    validate_resume_row(previous, row, 'original', config)
    for field, value in [('dataset_fingerprint', 'modified'), ('max_new_tokens', 128),
                          ('grading', 'other'), ('prompt_sha256', 'other'), ('status', 'error')]:
        with pytest.raises(ValueError):
            validate_resume_row({**previous, field: value}, row, 'original', config)
    for budget in (0, -1, 129, 1.5, True):
        with pytest.raises(ValueError):
            generation_budget({'max_new_tokens': budget}, config)


def test_summary_does_not_label_partial_task_coverage_full_ruler():
    path = Path(__file__).parents[1] / 'scripts/summarize_benchmarks.py'
    spec = importlib.util.spec_from_file_location('summary', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = [{'dataset': 'ruler_v1', 'id': 'a', 'task': 'vt', 'context_budget': 8192,
             'answer': ['a', 'b'], 'input_ids': [1, 2], 'dataset_fingerprint': 'x', 'max_new_tokens': 30}]
    report = mod.summarize(data, [])
    assert report['status'] == 'incomplete'
    assert report['methods']['bf16']['completed'] == 0
    row = {**data[0], 'repeat': 0, 'method': 'snapkv', 'status': 'ok', 'prediction': 'a', 'score': .5,
           'correct': False, 'prompt_sha256': fingerprint([1, 2]), 'fixed_output_length': False,
           'finish_reason': 'eos', 'generated_tokens': 3, 'input_tokens': 2,
           'e2e_s': 1, 'kv_cache_bytes': 1000, 'prefill_kv_bytes': 900}
    report = mod.summarize(data, [row])
    group = report['methods']['snapkv']['groups'][0]
    assert group['score_pct'] is None
    assert group['task_scores_pct']['vt'] == 50
    assert group['paired_with_bf16'] is False
    with pytest.raises(ValueError):
        mod.summarize(data, [row, row])
    with pytest.raises(ValueError):
        mod.summarize(data, [{**row, 'score': 1.0}])


def test_ruler_restores_assistant_answer_prefix():
    from inquant.benchmarks import ruler_prompt
    assert ruler_prompt({'input': 'question\nassistant:', 'answer_prefix': ' Answer:'}) == 'question\nassistant: Answer:'
    with pytest.raises(ValueError):
        ruler_prompt({'input': 'question'})
