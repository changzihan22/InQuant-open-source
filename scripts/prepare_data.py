#!/usr/bin/env python3
"""Export standard test splits, or a separate exact-token 32K retrieval workload."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inquant.evaluation import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["gsm8k", "math500", "passkey32k", "passkey"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", default="main", help="Pin a Hub dataset revision for archived experiments")
    p.add_argument("--split", choices=["test", "train"], default="test", help="GSM8K train is for calibration only")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--local-files-only", action="store_true", help="Use an existing local tokenizer without downloads")
    p.add_argument("--input-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=42, help="Synthetic retrieval seed; use a new seed for held-out validation")
    p.add_argument("--recorded-test-format", action="store_true", help="Reproduce the archived test JSONL schema without an explicit split field")
    args = p.parse_args()
    if args.split != "test" and args.dataset != "gsm8k":
        p.error("Only GSM8K exposes a train split here")
    if args.recorded_test_format and (args.split != "test" or args.dataset in ("passkey32k", "passkey")):
        p.error("--recorded-test-format is only for GSM8K/MATH500 test exports")
    if args.dataset in ("passkey32k", "passkey"):
        if args.dataset == "passkey32k" and args.input_tokens < 32768:
            p.error("passkey32k requires at least 32768 input tokens")
        if args.input_tokens < 1024:
            p.error("Synthetic retrieval requires at least 1024 input tokens")
        from transformers import AutoConfig, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, local_files_only=args.local_files_only)
        model_config = AutoConfig.from_pretrained(args.model, revision=args.model_revision, local_files_only=args.local_files_only)
        rows = []
        rng = random.Random(args.seed)
        marker = "__INQUANT_CONTEXT_MARKER__"
        messages = [{"role": "user", "content": marker + "\nWhat is the secret passkey? Reply with the digits only."}]
        if model_config.model_type != "mistral":
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        shell = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix, suffix = shell.split(marker)
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        front, back = encode(prefix), encode(suffix)
        filler_unit = encode("This is an ordinary archive entry. The weather is mild and the trees are green.\n")
        for index, depth in enumerate((0.1, 0.5, 0.9)):
            answer = str(rng.randrange(10000000, 99999999))
            needle = encode("\nThe secret passkey is " + answer + ". Remember this exact passkey.\n")
            free = args.input_tokens - len(front) - len(back) - len(needle)
            if free <= 0:
                raise ValueError("Token budget too small")
            filler = (filler_unit * (free // len(filler_unit) + 1))[:free]
            position = int(free * depth)
            ids = front + filler[:position] + needle + filler[position:] + back
            assert len(ids) == args.input_tokens
            rows.append({"id": f"depth-{depth}", "dataset": args.dataset,
                         "input_ids": ids, "answer": answer, "depth": depth,
                         "seed": args.seed,
                         "tokenizer_model": args.model, "tokenizer_revision": args.model_revision})
        dataset_hash = fingerprint(rows)
    else:
        from huggingface_hub import HfApi
        import pyarrow.parquet as pq
        import requests
        repo, subset = ("openai/gsm8k", "main") if args.dataset == "gsm8k" else ("HuggingFaceH4/MATH-500", "default")
        info = HfApi().dataset_info(repo, revision=args.revision, files_metadata=True)
        candidates = [item for item in info.siblings
                      if (item.rfilename.endswith(".parquet") and f"/{args.split}-" in item.rfilename or item.rfilename == f"{args.split}.jsonl")
                      and (args.dataset != "gsm8k" or item.rfilename.startswith("main/"))]
        if not candidates:
            raise RuntimeError("Could not locate official test parquet files")
        raw_dir = Path(args.output).parent / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dataset = []
        for item in sorted(candidates, key=lambda value: value.rfilename):
            raw = raw_dir / (args.dataset + "-" + info.sha + "-" + Path(item.rfilename).name)
            if not raw.exists():
                response = requests.get(f"https://huggingface.co/datasets/{repo}/resolve/{info.sha}/{item.rfilename}", timeout=120)
                response.raise_for_status()
                if len(response.content) != item.size:
                    raise RuntimeError("Dataset download size mismatch")
                raw.write_bytes(response.content)
            if item.rfilename.endswith(".jsonl"):
                with raw.open() as source:
                    dataset.extend(json.loads(line) for line in source if line.strip())
            else:
                dataset.extend(pq.read_table(raw).to_pylist())
        rows = []
        for index, item in enumerate(dataset):
            rows.append({"id": str(index), "dataset": args.dataset,
                         "split": args.split,
                         "question": item["question"] if args.dataset == "gsm8k" else item["problem"],
                         "answer": item["answer"], "source": repo, "source_revision": info.sha})
            if args.recorded_test_format:
                rows[-1].pop("split")
        dataset_hash = fingerprint(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        for row in rows:
            row["dataset_fingerprint"] = dataset_hash
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(path), "samples": len(rows), "sha256": dataset_hash}))


if __name__ == "__main__":
    main()
