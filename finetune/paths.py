"""Artifact locations, read from the environment set by env/paths.sh."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Run `source env/paths.sh` first.")
    return Path(value)


def model_id() -> str:
    return os.environ.get("AR_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")


def model_dir() -> Path:
    return _env_path("AR_MODEL_DIR")


def datasets_dir() -> Path:
    return _env_path("AR_DATASETS_DIR")


def prepared_dir() -> Path:
    return _env_path("AR_PREPARED_DIR")


def adapters_dir() -> Path:
    return _env_path("AR_ADAPTERS_DIR")
