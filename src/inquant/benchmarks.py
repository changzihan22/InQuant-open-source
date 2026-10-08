"""Scoring and prompt identities for RULER v1 and held-out AIME 2026."""
from __future__ import annotations

import re
from .evaluation import fingerprint, grade, last_boxed

RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
AIME_REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"
RULER_TASKS = ("niah_single_1", "niah_single_2", "niah_single_3",
               "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
               "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2")


def tokenizer_fingerprint(tokenizer):
    return fingerprint({"vocab": tokenizer.get_vocab(),
                        "special_tokens": tokenizer.special_tokens_map,
                        "chat_template": tokenizer.chat_template})


def generation_budget(row, config):
    budget = row.get("max_new_tokens", config["max_new_tokens"])
    if type(budget) is not int or not 0 < budget <= config["max_new_tokens"]:
        raise ValueError("Per-row generation budget must be positive and within the configured cap")
    return budget


def ruler_score(prediction, answers, task):
    """Official case-insensitive substring metric, including fractional credit.

    NVIDIA/RULER eval/synthetic/constants.py uses any-alias matching for QA
    and the fraction of retrieved targets for every other task. Do not add
    word boundaries or deduplicate references: both would change the metric.
    """
    if task not in RULER_TASKS:
        raise ValueError(f"Unknown RULER task: {task}")
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a for a in answers):
        raise ValueError("RULER references must be a nonempty list of nonempty strings")
    cleaned = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().lower()
    matches = [a.lower() in cleaned for a in answers]
    return float(any(matches)) if task.startswith("qa_") else sum(matches) / len(matches)


def aime_answer(prediction):
    answer = last_boxed(prediction)
    if answer is None:
        # An explicit final answer or an answer-only response; never a number
        # selected from intermediate reasoning.
        anchored = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)", prediction, re.I)
        answer = anchored[-1] if anchored else prediction.strip()
    answer = answer.strip().rstrip(".").strip()
    if re.fullmatch(r"[0-9]{1,3}", answer):
        return int(answer)
    return None


def score_row(row, prediction):
    if row["dataset"] == "ruler_v1":
        return ruler_score(prediction, row["answer"], row["task"])
    if row["dataset"] == "aime2026":
        target = str(row["answer"])
        if not re.fullmatch(r"[0-9]{1,3}", target):
            raise ValueError("AIME reference must be an integer from 0 to 999")
        return float(aime_answer(prediction) == int(target))
    return float(grade(row["dataset"], prediction, row["answer"]))


def grading_protocol(row):
    if row["dataset"] == "ruler_v1":
        return "ruler-v1-substring-all-or-part-v1"
    if row["dataset"] == "aime2026":
        return "aime-final-integer-exact-v1"
    return "explicit-boxed-gsm8k+math-verify-0.8.0"


def validate_resume_row(previous, row, data_hash, config):
    if previous.get("status") != "ok":
        raise ValueError("Cannot resume unsuccessful records")
    if previous.get("dataset_fingerprint") != (row.get("dataset_fingerprint") or data_hash):
        raise ValueError("Resume dataset fingerprint mismatch")
    if previous.get("max_new_tokens") != generation_budget(row, config):
        raise ValueError("Resume generation budget mismatch")
    if previous.get("grading") != grading_protocol(row):
        raise ValueError("Resume grading protocol mismatch")
    if "input_ids" in row and previous.get("prompt_sha256") != fingerprint(row["input_ids"]):
        raise ValueError("Resume prompt mismatch")


def ruler_prompt(raw):
    """The pinned generator stores the assistant prefill separately."""
    if not isinstance(raw.get('input'), str) or not isinstance(raw.get('answer_prefix'), str):
        raise ValueError('Expected upstream input and answer_prefix strings')
    return raw['input'] + raw['answer_prefix']
