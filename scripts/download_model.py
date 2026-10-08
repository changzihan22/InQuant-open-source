#!/usr/bin/env python3
"""Download the official checkpoint with streaming GET and LFS SHA256 verification.

Useful on proxies which strip the Hub HEAD metadata used by snapshot_download.
Only public model files are fetched; credentials and signed redirect URLs are not logged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from huggingface_hub import HfApi
import requests


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--revision", default="main", help="Use a commit SHA to reproduce the exact checkpoint")
    p.add_argument("--destination", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--manifest", default="results/model_manifest.json")
    p.add_argument("--aria2", action="store_true", help="Use installed aria2 for parallel ranged transfers")
    args = p.parse_args()
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    root = Path(args.destination)
    root.mkdir(parents=True, exist_ok=True)
    selected = [f for f in info.siblings if f.rfilename.endswith((".safetensors", ".json", ".txt", ".model"))]
    print(json.dumps({"repo": args.repo, "revision": info.sha, "bytes": sum(f.size for f in selected), "files": len(selected)}), flush=True)
    if args.aria2:
        if not shutil.which("aria2c"):
            raise RuntimeError("aria2c is not installed")
        entries = []
        for item in selected:
            path = root / item.rfilename
            if path.exists() and path.stat().st_size == item.size:
                continue
            urls = f"https://huggingface.co/{args.repo}/resolve/{info.sha}/{item.rfilename}?download=true"
            if item.lfs:
                urls = f"https://modelscope.cn/models/{args.repo}/resolve/master/{item.rfilename}\t" + urls
            entries.append(urls + f"\n dir={path.parent}\n out={path.name}.partial\n"
                           + (f" checksum=sha-256={item.lfs.sha256}\n" if item.lfs else ""))
        transfers = Path(args.manifest).parent / "model_download.aria2.txt"
        transfers.parent.mkdir(parents=True, exist_ok=True)
        transfers.write_text("\n".join(entries))
        if entries:
            subprocess.run(["aria2c", "--continue=true", "--max-connection-per-server=16", "--split=16",
                            "--min-split-size=16M", "--max-concurrent-downloads=4", "--file-allocation=none",
                            "--check-integrity=true", "--auto-file-renaming=false", "--allow-overwrite=false",
                            "--console-log-level=warn", "--summary-interval=30", "--download-result=hide",
                            "--input-file=" + str(transfers)], check=True)

    def download(item):
        path = root / item.rfilename
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = item.lfs.sha256 if item.lfs else None
        if path.exists() and path.stat().st_size == item.size and (not expected_hash or sha256(path) == expected_hash):
            print("verified existing " + item.rfilename, flush=True)
            return {"file": item.rfilename, "bytes": item.size, "sha256": expected_hash or sha256(path)}
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists() and partial.stat().st_size == item.size:
            digest = sha256(partial)
            if not expected_hash or digest == expected_hash:
                partial.replace(path)
                print("verified " + item.rfilename, flush=True)
                return {"file": item.rfilename, "bytes": item.size, "sha256": digest}
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                url = f"https://huggingface.co/{args.repo}/resolve/{info.sha}/{item.rfilename}?download=true"
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        offset = 0
                    else:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Unexpected resume range")
                    last_report = offset
                    with partial.open("ab" if offset else "wb") as target:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                target.write(chunk)
                                offset += len(chunk)
                                if offset - last_report >= 512 * 1024 * 1024:
                                    print(f"{item.rfilename}: {offset}/{item.size}", flush=True)
                                    last_report = offset
                if partial.stat().st_size != item.size:
                    raise ValueError("Incomplete download")
                digest = sha256(partial)
                if expected_hash and digest != expected_hash:
                    raise ValueError("Checkpoint SHA256 mismatch")
                partial.replace(path)
                print("verified " + item.rfilename, flush=True)
                return {"file": item.rfilename, "bytes": item.size, "sha256": digest}
            except (requests.RequestException, ValueError) as error:
                print(f"Retry {attempt + 1}: {item.rfilename}: {type(error).__name__}", flush=True)
                if attempt == 4:
                    raise RuntimeError(f"Cannot download/verify {item.rfilename}") from None
                time.sleep(min(2 ** attempt, 8))

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, selected))
    manifest = {"repo": args.repo, "revision": info.sha, "local_path": str(root), "files": files}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "inquant_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
