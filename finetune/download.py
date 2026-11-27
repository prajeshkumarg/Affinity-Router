"""Download the base model and all datasets once, to $AR_DATA, so GPU jobs run offline.

Usage (inside the env, after `source env/paths.sh`):
    python -m finetune.download            # model + datasets
    python -m finetune.download --skip-model
"""

from __future__ import annotations

import argparse
import json
import time

from datasets import load_dataset
from huggingface_hub import snapshot_download

from finetune import paths
from finetune.sources import DATASETS, MODEL_ALLOW_PATTERNS, MODEL_IGNORE_PATTERNS


def download_model() -> None:
    target = paths.model_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"[model] {paths.model_id()} -> {target}")
    t0 = time.time()
    snapshot_download(
        repo_id=paths.model_id(),
        local_dir=target,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        ignore_patterns=MODEL_IGNORE_PATTERNS,
    )
    size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
    print(f"[model] done in {time.time() - t0:.0f}s, {size_gb:.1f} GB")


def download_datasets() -> None:
    root = paths.datasets_dir()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo_id, config) in DATASETS.items():
        out = root / name
        if out.exists():
            print(f"[data] {name}: already at {out}, skipping")
        else:
            print(f"[data] {name}: {repo_id} -> {out}")
            ds = load_dataset(repo_id, config)
            ds.save_to_disk(str(out))
        manifest[name] = {"repo_id": repo_id, "config": config, "path": str(out)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    args = parser.parse_args()
    if not args.skip_model:
        download_model()
    if not args.skip_datasets:
        download_datasets()


if __name__ == "__main__":
    main()
