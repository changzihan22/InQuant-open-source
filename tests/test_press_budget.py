"""Regression: short prompts must not prune inside SnapKV's observation window."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parents[1] / "scripts/run_eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_budget_guard_catches_prompts_longer_than_window_but_too_short_to_prune():
    assert runner.snapkv_skip_reason(61, 64, .5)
    assert runner.snapkv_skip_reason(100, 64, .5)
    assert runner.snapkv_skip_reason(129, 64, .5)
    assert runner.snapkv_skip_reason(130, 64, .5) is None
    assert runner.snapkv_skip_reason(100, 64, .2) is None
    assert runner.snapkv_skip_reason(32768, 64, .5) is None
