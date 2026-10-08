#!/usr/bin/env python3
"""Summarize actual JSONL runs, without converting incomplete runs into acceptance."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inquant.evaluation import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default="results/summary.json")
    args = parser.parse_args()
    summary = []
    for filename in args.files:
        groups = defaultdict(list)
        for row in read_jsonl(filename):
            if row["dataset"] in ("ruler_v1", "aime2026"):
                parser.error("Use summarize_benchmarks.py for RULER/AIME task scores and paired protocol checks")
            groups[(row["method"], row["dataset"], row.get("dataset_split", "unknown"))].append(row)
        for (method, dataset, split), rows in groups.items():
            summary.append({"file": filename, "method": method, "dataset": dataset, "split": split,
                            "n_records": len(rows), "n_unique_samples": len({row["id"] for row in rows}),
                            "accuracy": mean(row["correct"] for row in rows),
                            "median_e2e_s": median(row["e2e_s"] for row in rows),
                            "mean_generated_tokens": mean(row["generated_tokens"] for row in rows),
                            "median_kv_bytes": median(row["kv_cache_bytes"] for row in rows),
                            "max_peak_allocated_bytes": max(row.get("peak_allocated_bytes") or 0 for row in rows),
                            "min_input_tokens": min(row["input_tokens"] for row in rows),
                            "max_input_tokens": max(row["input_tokens"] for row in rows),
                            "fused_decode_calls": sum(row.get("fused_decode_calls", 0) for row in rows)})
    report = {"status": "descriptive_only_not_acceptance", "runs": summary}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
