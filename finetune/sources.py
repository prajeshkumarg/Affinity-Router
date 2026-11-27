"""Hugging Face sources for the two adapters. Changing this list changes the experiment."""

from __future__ import annotations

# name -> (hub repo id, config name or None)
DATASETS: dict[str, tuple[str, str | None]] = {
    # Finance: instruction-formatted sentiment (FPB / FiQA-style) and ConvFinQA numeric QA.
    "fingpt_sentiment": ("FinGPT/fingpt-sentiment-train", None),
    "fingpt_convfinqa": ("FinGPT/fingpt-convfinqa", None),
    # Coding: OSS-Instruct problems with solutions (MIT license).
    "magicoder_oss": ("ise-uiuc/Magicoder-OSS-Instruct-75K", None),
    # Coding eval (used in Phase 1 eval, downloaded now so GPU nodes need no network).
    "humaneval": ("openai/openai_humaneval", None),
}

# Model files to fetch. Skips Meta's original/ consolidated checkpoint (~16 GB, not needed).
MODEL_ALLOW_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model"]
MODEL_IGNORE_PATTERNS = ["original/*"]
