#!/usr/bin/env python3
"""Build the English source release, excluding local experiment archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {'README.md', 'ROADMAP.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
              'pyproject.toml', 'requirements-reproduce.txt', '.gitignore',
              'benchmarks/runtime_20260925.json'}
SOURCE_DIRS = {'src', 'tests', 'extensions', 'third_party'}
SCRIPTS = {
    'download_model.py', 'prepare_data.py', 'run_eval.py', 'run_matrix.py',
    'summarize_results.py', 'check_acceptance.py', 'run_vllm.py', 'run_vllm_math.py',
    'bench_attention.py', 'run_gsm8k_mechanism_probe.py', 'package_source.py',
    'bench_cache_runtime.py', 'bench_decode_pair.py', 'bench_page_packing.py',
    'check_local_model.py', 'prepare_benchmarks.py',
    'run_benchmark_campaign.py', 'summarize_benchmarks.py', 'publish_benchmark_report.py',
}
CONFIGS = {
    'qwen2.5_7b.json', 'qwen2.5_7b_32k.json',
    'qwen2.5_7b_latest_k4v2.json', 'qwen2.5_7b_32k_latest_k4v2.json',
    'zipcache_qwen2.5_7b.json', 'zipcache_qwen2.5_7b_32k.json',
    'vllm_inquant_smoke.json', 'vllm_inquant_32k.json',
    'vllm_bf16_extension_control_32k.json', 'calibration_gsm8k.json',
    'qwen2.5_7b_ruler.json', 'qwen2.5_7b_aime2026.json',
    'mistral_7b_k4v2.json', 'mistral_7b_context32k_k4v2.json',
}
ASSETS = {'figure-1.png', 'figure-3.png'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist'}


def source_files():
    selected = [ROOT / name for name in ROOT_FILES]
    report = ROOT / 'benchmarks/ruler_aime2026.json'
    if report.is_file():
        selected.append(report)
    selected += [ROOT / 'scripts' / name for name in SCRIPTS]
    selected += [ROOT / 'configs' / name for name in CONFIGS]
    selected += [ROOT / 'docs/assets' / name for name in ASSETS]
    for dirname in sorted(SOURCE_DIRS):
        for base, dirs, filenames in os.walk(ROOT / dirname, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not d.endswith('.egg-info') and not d.startswith('.venv'))
            if any((Path(base) / d).is_symlink() for d in dirs):
                raise ValueError(f'Symlink requires explicit review: {base}')
            for name in filenames:
                path = Path(base) / name
                if path.suffix not in {'.pyc', '.whl', '.zip', '.pdf'} and not name.startswith('.env'):
                    selected.append(path)
    for path in sorted(selected):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Required release file is missing or a symlink: {path}')
        yield path, path.relative_to(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'InQuant-open-source-en.zip')
    parser.add_argument('--export-dir', type=Path, help='Also create an unpacked source directory; must not exist')
    args = parser.parse_args()
    if args.export_dir is not None and args.export_dir.exists():
        parser.error('--export-dir already exists; choose a new directory')
    paths = list(source_files())
    manifest = {'format': 1, 'language': 'en',
                'scope': 'English source release. Historical reports, weights, datasets and raw runs are not bundled.',
                'files': {str(rel): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'bytes': path.stat().st_size} for path, rel in paths}}
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, rel in paths:
            archive.write(path, str(Path('InQuant') / rel))
        archive.writestr('InQuant/SOURCE_MANIFEST.json', manifest_text)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP CRC verification failed')
    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True)
        for path, rel in paths:
            destination = args.export_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        (args.export_dir / 'SOURCE_MANIFEST.json').write_text(manifest_text)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(f'{digest}  {args.output.name}\n')
    print(json.dumps({'archive': str(args.output.resolve()), 'files': len(paths) + 1,
                      'bytes': args.output.stat().st_size, 'sha256': digest,
                      'export_dir': str(args.export_dir.resolve()) if args.export_dir else None}, indent=2))


if __name__ == '__main__':
    main()
