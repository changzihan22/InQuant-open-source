#!/usr/bin/env python3
"""Compare recorded outputs; absent evidence never counts as success."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inquant.evaluation import compare, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", nargs="+", required=True, help="BF16 math and 32K JSONL files")
    p.add_argument("--candidate", nargs="+", required=True, help="Candidate math and 32K JSONL files")
    p.add_argument("--accuracy-mode", choices=["absolute", "relative", "both"], default="absolute",
                   help="Default: accuracy loss below 3 percentage points versus BF16")
    p.add_argument("--output")
    args = p.parse_args()
    reference = [row for path in args.reference for row in read_jsonl(path)]
    candidate = [row for path in args.candidate for row in read_jsonl(path)]
    report = compare(reference, candidate, accuracy_mode=args.accuracy_mode)
    report["input_files"] = {"reference": args.reference, "candidate": args.candidate}
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
