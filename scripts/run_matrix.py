#!/usr/bin/env python3
"""Run a reproducible, resumable experiment matrix on one explicitly selected GPU.

Source and configuration snapshots keep a long-running experiment independent
of subsequent working-tree edits. This script never stops other GPU processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--methods", nargs="+", choices=["bf16", "inquant_fused", "snapkv", "knorm", "zipcache"], required=True)
    p.add_argument("--datasets", nargs="+", choices=["gsm8k", "math500", "long", "calibration"], required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--limit", type=int)
    p.add_argument("--donor-policy", choices=["neighbors", "min_error"])
    p.add_argument("--salient-fraction", type=float)
    p.add_argument("--long-data", default="data/passkey32k.jsonl", help="Independent synthetic validation workload")
    p.add_argument("--math-config", default="qwen2.5_7b.json", help="Filename under configs/ for GSM8K and MATH500")
    p.add_argument("--long-config", default="qwen2.5_7b_32k.json", help="Filename under configs/ for 32K")
    p.add_argument("--acceptance-reference", nargs="+", help="After an InQuant suite completes, compare these BF16 JSONLs automatically")
    p.add_argument("--acceptance-output", help="Default: <output-root>/acceptance.json")
    args = p.parse_args()
    for name in (args.math_config, args.long_config):
        if Path(name).name != name or not (ROOT / "configs" / name).is_file():
            p.error("Custom configs must be existing filenames directly under configs/")
    if args.acceptance_reference and (args.methods != ["inquant_fused"] or set(args.datasets) != {"gsm8k", "math500", "long"}):
        p.error("Automatic acceptance requires only inquant_fused and exactly the long/gsm8k/math500 suite")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ("src", "scripts", "configs"):
            shutil.copytree(ROOT / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if "zipcache" in args.methods:
            shutil.copytree(ROOT / "third_party/ZipCache", snapshot / "third_party/ZipCache",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    digest = hashlib.sha256()
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(str(path.relative_to(snapshot)).encode())
            digest.update(path.read_bytes())
    status_path = output / "status.json"
    status = {"pid": os.getpid(), "gpu": args.gpu, "source_sha256": digest.hexdigest(),
              "started_at": datetime.now(timezone.utc).isoformat(), "state": "starting", "jobs": []}
    active = None

    def save():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    def stop(_sig, _frame):
        if active is not None and active.poll() is None:
            active.terminate()
        status["state"] = "interrupted"
        save()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    # Finish all methods for the first workload before the longer math suites.
    # Passing --datasets long gsm8k math500 gives early long-context comparisons.
    for dataset in args.datasets:
        for method in args.methods:
            config_name = args.long_config if dataset == "long" else "calibration_gsm8k.json" if dataset == "calibration" else args.math_config
            data_name = "passkey32k.jsonl" if dataset == "long" else "calibration_gsm8k32.jsonl" if dataset == "calibration" else dataset + ".jsonl"
            result = output / f"{method}_{dataset}.jsonl"
            log_path = output / f"{method}_{dataset}.log"
            job = {"method": method, "dataset": dataset, "output": str(result), "log": str(log_path)}
            status["jobs"].append(job)
            while True:
                free = int(subprocess.check_output([
                    "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
                ], text=True).strip())
                occupants = subprocess.check_output([
                    "nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
                ], text=True).strip().splitlines()
                if free >= 30 * 1024 and not occupants:
                    status.pop("free_mib", None)
                    status.pop("gpu_processes", None)
                    break
                status["state"] = "waiting_for_gpu"
                status["free_mib"] = free
                status["gpu_processes"] = occupants
                save()
                time.sleep(30)
            command = [sys.executable, str(snapshot / "scripts/run_eval.py"),
                       "--config", str(snapshot / "configs" / config_name),
                       "--model", args.model, "--data", str((ROOT / args.long_data) if dataset == "long" else (ROOT / "data" / data_name)),
                       "--method", method, "--output", str(result), "--resume"]
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            for name in ("donor_policy", "salient_fraction"):
                value = getattr(args, name)
                if value is not None:
                    command.extend(["--" + name.replace("_", "-"), str(value)])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            status["state"] = "running"
            job["command"] = command
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            with log_path.open("a") as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = active.pid
                save()
                job["exit_code"] = active.wait()
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if job["exit_code"] != 0:
                status["state"] = "failed"
                save()
                return job["exit_code"]
            save()
    if args.acceptance_reference:
        report = Path(args.acceptance_output).resolve() if args.acceptance_output else output / "acceptance.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(snapshot / "scripts/check_acceptance.py"),
                   "--reference", *(str(Path(name).resolve()) for name in args.acceptance_reference),
                   "--candidate", *(str(output / f"inquant_fused_{name}.jsonl") for name in args.datasets),
                   "--output", str(report)]
        status["state"] = "checking_acceptance"
        status["acceptance"] = {"command": command, "report": str(report)}
        save()
        with (output / "acceptance.log").open("a") as log:
            active = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status["acceptance"]["exit_code"] = active.wait()
        if not report.exists():
            status["state"] = "failed"
            save()
            return status["acceptance"]["exit_code"] or 1
        status["acceptance"]["result"] = json.loads(report.read_text())["status"]
    status["state"] = "complete"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
