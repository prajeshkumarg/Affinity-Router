"""Turn the downloaded datasets into prompt/completion chat JSONL for one domain.

Output: $AR_PREPARED_DIR/<domain>/{train,eval}.jsonl and stats.json.
Each line: {"prompt": [{"role": "user", ...}], "completion": [{"role": "assistant", ...}],
            "source": "<dataset name>"}

Usage:
    python -m finetune.prepare_data --config configs/finetune/finance.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Iterable
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

from finetune import paths


def to_example(user: str, assistant: str, source: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": user.strip()}],
        "completion": [{"role": "assistant", "content": assistant.strip()}],
        "source": source,
    }


def instruction_rows(rows: Iterable[dict], source: str) -> list[dict]:
    """FinGPT-style rows with instruction / input / output columns."""
    out = []
    for r in rows:
        instruction = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        user = f"{instruction}\n\n{inp}" if inp else instruction
        if user and (r.get("output") or "").strip():
            out.append(to_example(user, str(r["output"]), source))
    return out


def sample(rows: list, n: int, rng: random.Random) -> list:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n] if n < len(rows) else rows


def build_finance(cfg: dict, rng: random.Random) -> tuple[list[dict], list[dict]]:
    root = paths.datasets_dir()
    sent = instruction_rows(load_from_disk(str(root / "fingpt_sentiment"))["train"], "sentiment")
    sent = sample(sent, cfg["sentiment_train"] + cfg["sentiment_eval"], rng)
    sent_eval, sent_train = sent[: cfg["sentiment_eval"]], sent[cfg["sentiment_eval"] :]

    cfqa = load_from_disk(str(root / "fingpt_convfinqa"))
    cfqa_train = sample(instruction_rows(cfqa["train"], "convfinqa"), cfg["convfinqa_train"], rng)
    cfqa_eval = sample(instruction_rows(cfqa["test"], "convfinqa"), cfg["convfinqa_eval"], rng)
    return sent_train + cfqa_train, sent_eval + cfqa_eval


def build_coding(cfg: dict, rng: random.Random) -> tuple[list[dict], list[dict]]:
    ds = load_from_disk(str(paths.datasets_dir() / "magicoder_oss"))["train"]
    langs = set(cfg.get("languages", ["python"]))
    rows = [
        to_example(r["problem"], r["solution"], f"magicoder_{r['lang']}")
        for r in ds
        if r["lang"] in langs and r["problem"] and r["solution"]
    ]
    rows = sample(rows, cfg["train"] + cfg["eval"], rng)
    return rows[cfg["eval"] :], rows[: cfg["eval"]]


def length_filter(rows: list[dict], tokenizer, max_length: int) -> tuple[list[dict], list[int]]:
    """Drop examples longer than max_length so training never truncates a completion."""
    kept, lengths = [], []
    for ex in rows:
        ids = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=True)
        if not isinstance(ids, list):  # newer transformers may return a BatchEncoding
            ids = ids["input_ids"]
        if len(ids) <= max_length:
            kept.append(ex)
            lengths.append(len(ids))
    return kept, lengths


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def describe(rows: list[dict], lengths: list[int]) -> dict:
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    return {
        "n": len(rows),
        "by_source": by_source,
        "tokens_mean": round(statistics.mean(lengths), 1) if lengths else 0,
        "tokens_p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
        "tokens_max": max(lengths) if lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    domain, data_cfg = cfg["domain"], cfg["data"]
    max_length = cfg["train"]["max_length"]

    rng = random.Random(cfg["seed"])
    builder = {"finance": build_finance, "coding": build_coding}[domain]
    train, evals = builder(data_cfg, rng)

    tokenizer = AutoTokenizer.from_pretrained(str(paths.model_dir()))
    train, train_len = length_filter(train, tokenizer, max_length)
    evals, eval_len = length_filter(evals, tokenizer, max_length)
    rng.shuffle(train)

    out_dir = paths.prepared_dir() / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", evals)
    stats = {
        "domain": domain,
        "seed": cfg["seed"],
        "max_length": max_length,
        "train": describe(train, train_len),
        "eval": describe(evals, eval_len),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
