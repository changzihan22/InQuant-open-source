import copy

import pytest

from inquant.evaluation import compare, grade, gsm_number, last_boxed, PROTOCOL_KEYS


def test_grading_avoids_intermediate_answer_false_positive():
    assert grade("gsm8k", r"Calculations... \boxed{1,250}", "work #### 1250")
    assert not grade("gsm8k", "1250 was the initial amount, then we lost some.", "#### 1250")
    assert not grade("gsm8k", r"\boxed{1251}", "#### 1250")
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{1") is None
    assert gsm_number("NaN", gold=True) is None
    assert grade("gsm8k", r"\boxed{\frac{3}{2}}", "#### 1.5")
    assert grade("gsm8k", r"\boxed{10\text{ dollars}}", "#### 10")
    assert not grade("passkey32k", "001234567899", "12345678")
    assert grade("passkey32k", "The passkey is 12345678.", "12345678")


def fixture_rows():
    rows = []
    for dataset, size in (("gsm8k", 1319), ("math500", 500), ("passkey32k", 3)):
        for index in range(size):
            for repeat in range(5 if dataset == "passkey32k" else 1):
                row = {key: "same" for key in PROTOCOL_KEYS}
                row.update(dataset=dataset, id=str(index), repeat=repeat, correct=True,
                           dataset_split="synthetic" if dataset == "passkey32k" else "test",
                           model_repo="Qwen/Qwen2.5-7B-Instruct",
                           method="bf16", status="ok", e2e_s=10., kv_cache_bytes=1000,
                           input_tokens=32768, generated_tokens=128, max_new_tokens=128,
                           prompt_sha256=str(index), dataset_fingerprint="data", warmup_runs=2,
                           fixed_output_length=True, device_type="cuda")
                rows.append(row)
    candidate = copy.deepcopy(rows)
    for row in candidate:
        row.update(method="inquant", e2e_s=8., kv_cache_bytes=700)
    return rows, candidate


def test_full_paired_report_and_missing_evidence():
    ref, cand = fixture_rows()
    assert compare(ref, cand)["status"] == "PASS"
    assert compare(ref[:-1], cand)["status"] == "NOT_MET"
    assert compare([], [])["status"] == "NOT_MET"
    cand[0]["prompt_sha256"] = "different"
    assert compare(ref, cand)["status"] == "NOT_MET"


def test_accuracy_per_dataset_and_absolute_boundary():
    ref, cand = fixture_rows()
    for row in cand:
        if row["dataset"] == "math500" and int(row["id"]) < 15:
            row["correct"] = False
    assert compare(ref, cand)["status"] == "NOT_MET"  # exactly 3% is not <3%


def test_default_absolute_accuracy_accepts_2_6pp_drop_from_78_8_percent():
    ref, cand = fixture_rows()
    for rows, correct_count in ((ref, 394), (cand, 381)):
        for row in rows:
            if row['dataset'] == 'math500':
                row['correct'] = int(row['id']) < correct_count
    report = compare(ref, cand)
    assert report['accuracy_mode'] == 'absolute'
    assert report['status'] == 'PASS'
    assert report['checks']['math500']['loss_percentage_points'] == pytest.approx(2.6)
    assert 'relative_loss' not in report['checks']['math500']
    assert compare(ref, cand, accuracy_mode='both')['status'] == 'NOT_MET'


def test_cpu_short_output_or_non_bf16_cannot_pass():
    ref, cand = fixture_rows()
    cand[-1]["device_type"] = "cpu"
    assert compare(ref, cand)["status"] == "NOT_MET"
    cand[-1]["device_type"] = "cuda"
    cand[-1]["generated_tokens"] = 127
    assert compare(ref, cand)["status"] == "NOT_MET"
    ref[0]["method"] = "snapkv"
    assert compare(ref, cand)["status"] == "NOT_MET"


def test_duplicates_rejected():
    ref, cand = fixture_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        compare(ref + [ref[0]], cand)


def test_fast_small_cache_with_wrong_retrieval_still_fails_acceptance():
    ref, cand = fixture_rows()
    cand[-1]["correct"] = False
    report = compare(ref, cand)
    assert report["status"] == "NOT_MET"
    assert report["checks"]["kv_savings"]["pass"]
    assert report["checks"]["e2e_latency"]["pass"]
    assert not report["checks"]["32k_retrieval"]["pass"]


def test_mixed_models_and_unfrozen_codec_cannot_pass():
    ref, cand = fixture_rows()
    for row in ref + cand:
        if row["dataset"] == "passkey32k":
            row["model"] = "tiny-random-qwen"
    assert compare(ref, cand)["status"] == "NOT_MET"
    ref, cand = fixture_rows()
    cand[-1]["config"] = {"salient_fraction": 0.2}
    assert compare(ref, cand)["status"] == "NOT_MET"


@pytest.mark.parametrize("field,value", [
    ("value_bits", 4),
    ("short_context", {"max_context_tokens":4096,"block_size":128,"residual_length":32}),
])
def test_optimized_candidate_must_freeze_value_codec_and_context_policy(field, value):
    ref, cand = fixture_rows()
    for row in cand:
        row['config'] = {"value_bits":2,"value_group_size":64,"shared_workspace":True,"track_peak_bytes":True,
                         "short_context":{"max_context_tokens":4096,"block_size":64,"residual_length":32}}
    assert compare(ref,cand)['status']=='PASS'
    cand[-1]['config'][field]=value
    report=compare(ref,cand)
    assert report['status']=='NOT_MET'
    assert any('configuration must remain fixed' in e for e in report['protocol_errors'])
