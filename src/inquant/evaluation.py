"""Answer grading and fail-closed, paired experiment acceptance checks."""
from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median

EXPECTED_SIZES = {"gsm8k": 1319, "math500": 500}
PROTOCOL_KEYS = (
    "model", "model_repo", "model_architecture", "model_revision", "backend", "dtype", "attention", "batch_size",
    "max_new_tokens", "rope_scaling", "device_name", "software", "grading",
)


def last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    return None


def gsm_number(text: str, *, gold: bool = False) -> Decimal | None:
    answer = text.rsplit("####", 1)[-1] if "####" in text else last_boxed(text)
    if answer is None:
        # Explicit final-answer anchor only; do not reward an intermediate number.
        matches = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)", text, re.I)
        answer = matches[-1] if matches else (text if gold else "")
    answer = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}\s*$", "", answer.strip())
    answer = answer.replace(r"\,", "").replace(r"\$", "").replace(r"\%", "")
    answer = answer.strip().replace(",", "").replace("$", "").rstrip(".%").strip()
    fraction = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", answer)
    if fraction:
        numerator, denominator = map(Decimal, fraction.groups())
        return numerator / denominator if denominator else None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", answer):
        return None
    try:
        return Decimal(answer)
    except InvalidOperation:
        return None


def grade(dataset: str, prediction: str, answer: str) -> bool:
    if dataset == "gsm8k":
        target, actual = gsm_number(answer, gold=True), gsm_number(prediction)
        return target is not None and actual is not None and target == actual
    if dataset in ("math500", "math_train"):
        from math_verify import LatexExtractionConfig, parse, verify
        actual = last_boxed(prediction)
        if actual is None:
            return False
        configs = [LatexExtractionConfig()]
        gold = parse("$" + answer + "$", extraction_config=configs)
        pred = parse("$" + actual + "$", extraction_config=configs)
        return bool(gold and pred and verify(gold, pred))
    if dataset in ("passkey32k", "passkey"):
        return re.search(r"(?<!\d)" + re.escape(answer) + r"(?!\d)", prediction) is not None
    raise ValueError(f"Unknown grading dataset: {dataset}")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def _index(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        key = (row["dataset"], row["id"], row.get("repeat", 0))
        if key in result:
            raise ValueError(f"Duplicate result: {key}")
        result[key] = row
    return result


def compare(reference: list[dict], candidate: list[dict], *, accuracy_mode="absolute") -> dict:
    """No PASS for incomplete suites, CPU runs, differing prompts or protocols.

    Speed is latency reduction, 1 - median(candidate)/median(reference), >=10%.
    Math accuracy is evaluated on full test sets independently, never averaged.
    By default, accuracy loss must be below 3 percentage points versus BF16.
    Long-context benchmark is separate from the short standard math tasks.
    """
    if accuracy_mode not in ("both", "absolute", "relative"):
        raise ValueError("accuracy_mode must be both, absolute or relative")
    ref, cand = _index(reference), _index(candidate)
    errors = []
    if not ref or set(ref) != set(cand):
        errors.append("Both files must contain the same nonempty sample/repeat IDs.")
    common = sorted(set(ref) & set(cand))
    for key in common:
        a, b = ref[key], cand[key]
        for field in (*PROTOCOL_KEYS, "prompt_sha256", "input_tokens", "dataset_fingerprint"):
            if field not in a or field not in b or a[field] != b[field]:
                errors.append(f"Unmatched or absent {field}: {key}")
        if a.get("method") != "bf16":
            errors.append(f"Primary reference must be bf16: {key}")
        if b.get("method") not in ("inquant", "inquant_fused"):
            errors.append(f"Acceptance candidate must implement InQuant: {key}")
        for row in (a, b):
            if row.get("status") != "ok":
                errors.append(f"Failed/missing status: {key}")
            for field in ("e2e_s", "kv_cache_bytes", "input_tokens", "generated_tokens"):
                number = row.get(field)
                if not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                    errors.append(f"Invalid {field}: {key}")
            if not isinstance(row.get("correct"), bool):
                errors.append(f"Missing boolean correctness: {key}")
            if row.get("device_type") != "cuda":
                errors.append(f"Formal evaluation requires CUDA: {key}")
            if row.get("model_repo") != "Qwen/Qwen2.5-7B-Instruct":
                errors.append(f"Formal evaluation requires the requested official 7B model: {key}")
            if row.get("dataset_split") != ("synthetic" if row["dataset"] == "passkey32k" else "test"):
                errors.append(f"Calibration/unknown split cannot be used for acceptance: {key}")
        if b.get("method") == "inquant_fused" and b.get("fused_decode_calls", 0) <= 0:
            errors.append(f"No verified fused decode calls: {key}")
    invariant_keys = ("model", "model_repo", "model_architecture", "model_revision", "backend", "dtype",
                      "device_name", "software", "grading")
    identities = {fingerprint({key: row.get(key) for key in invariant_keys}) for row in reference + candidate}
    if len(identities) != 1:
        errors.append("The complete suite must use one model/revision/framework/hardware/software/grader identity.")
    codec_keys = ("block_size", "residual_length", "sink_tokens", "sample_stride", "salient_fraction", "donor_policy", "validate_positions",
                  "value_bits", "value_group_size", "shared_workspace", "track_peak_bytes", "short_context")
    codec_configs = {fingerprint({key: row.get("config", {}).get(key) for key in codec_keys}) for row in candidate}
    if len(codec_configs) > 1 or len({row.get("method") for row in candidate}) > 1:
        errors.append("Candidate method and codec configuration must remain fixed across the full suite.")
    if errors:
        return {"status": "NOT_MET", "accuracy_mode": accuracy_mode, "checks": {},
                "protocol_errors": sorted(set(errors))}
    checks = {}
    for dataset, size in EXPECTED_SIZES.items():
        keys = [key for key in common if key[0] == dataset]
        if len(keys) != size or len({key[1] for key in keys}) != size:
            checks[dataset] = {"pass": False, "reason": f"Need {size} unique complete test examples", "n": len(keys)}
            continue
        a = mean(float(ref[k]["correct"]) for k in keys)
        b = mean(float(cand[k]["correct"]) for k in keys)
        absolute = a - b
        relative = absolute / a if a else None
        passed = a > 0 and (absolute + 1e-12 < 0.03 if accuracy_mode != "relative" else True)
        passed = passed and (relative is not None and relative + 1e-12 < 0.03 if accuracy_mode != "absolute" else True)
        checks[dataset] = {"pass": bool(passed), "n": len(keys), "bf16_accuracy": a,
                           "candidate_accuracy": b, "loss_percentage_points": absolute * 100}
        if accuracy_mode != "absolute":
            checks[dataset]["relative_loss"] = relative
    long_keys = [k for k in common if k[0] == "passkey32k"]
    groups = {}
    for key in long_keys:
        groups.setdefault(key[1], []).append(key)
    ratios, savings = [], []
    valid_long = len(groups) >= 3
    correct_retrieval = bool(long_keys)
    for group in groups.values():
        if len(group) < 5:
            valid_long = False
        for key in group:
            a, b = ref[key], cand[key]
            valid_long &= (a.get("input_tokens", 0) >= 32768 and b.get("input_tokens", 0) >= 32768
                           and a.get("warmup_runs", 0) >= 2 and b.get("warmup_runs", 0) >= 2
                           and a.get("generated_tokens") == b.get("generated_tokens") == a.get("max_new_tokens")
                           and a.get("fixed_output_length") is True and b.get("fixed_output_length") is True
                           and a.get("device_type") == b.get("device_type") == "cuda")
            correct_retrieval &= a.get("correct") is True and b.get("correct") is True
        if all(ref[k].get("e2e_s", 0) > 0 for k in group):
            ratios.append(median(cand[k]["e2e_s"] for k in group) / median(ref[k]["e2e_s"] for k in group))
        savings.extend(1 - cand[k]["kv_cache_bytes"] / ref[k]["kv_cache_bytes"]
                       for k in group if ref[k].get("kv_cache_bytes", 0) > 0)
    checks["32k_workload"] = {"pass": bool(valid_long), "cases": len(groups),
                                "reason": "Need >=3 cases, >=5 repeats, 2 warmups, >=32768 input tokens and fixed output on CUDA"}
    checks["32k_retrieval"] = {"pass": bool(valid_long and correct_retrieval), "n": len(long_keys),
                               "bf16_correct": sum(ref[k]["correct"] for k in long_keys),
                               "candidate_correct": sum(cand[k]["correct"] for k in long_keys),
                               "reason": "All measured retrieval cases must be correct; speed and memory alone cannot pass acceptance"}
    checks["kv_savings"] = {"pass": bool(valid_long and savings and min(savings) + 1e-12 >= 0.20),
                            "minimum_fraction": min(savings) if savings else None}
    checks["e2e_latency"] = {"pass": bool(valid_long and ratios and max(ratios) <= 0.90),
                             "minimum_reduction": 1 - max(ratios) if ratios else None}
    return {"status": "PASS" if not errors and all(v["pass"] for v in checks.values()) else "NOT_MET",
            "accuracy_mode": accuracy_mode, "checks": checks, "protocol_errors": sorted(set(errors))}
