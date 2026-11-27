"""QLoRA fine-tuning of one domain adapter on the 4-bit base model.

Reads $AR_PREPARED_DIR/<domain>/{train,eval}.jsonl and writes:
    $AR_DATA/checkpoints/<domain>/<run>/     intermediate checkpoints (resumable)
    $AR_ADAPTERS_DIR/<domain>/<run>/         final adapter + train_meta.json + samples.jsonl
    $AR_ADAPTERS_DIR/<domain>/current        symlink to the latest finished run

Usage:
    python -m finetune.train_lora --config configs/finetune/finance.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import peft
import torch
import transformers
import trl
import yaml
from datasets import load_dataset
from packaging.version import Version
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from finetune import paths

N_SAMPLES = 3


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.environ.get("AR_REPO", "."), text=True
        ).strip()
    except Exception:
        return None


def load_tokenizer(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        # Llama 3.1 ships a dedicated pad token; fall back to EOS for other models.
        pad = "<|finetune_right_pad_id|>"
        tok.pad_token = pad if pad in tok.get_vocab() else tok.eos_token
    return tok


def load_base_4bit(model_dir: Path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb,
        attn_implementation="sdpa",
        device_map={"": 0},
        **{dtype_kw: torch.bfloat16},
    )
    model.config.use_cache = False
    return model


def load_split(domain: str, split: str):
    path = paths.prepared_dir() / domain / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run finetune.prepare_data first.")
    ds = load_dataset("json", data_files=str(path), split="train")
    return ds.select_columns(["prompt", "completion"])


@torch.no_grad()
def write_samples(model, tokenizer, eval_path: Path, out_path: Path) -> None:
    """Greedy-decode a few eval prompts as a quick sanity check of the adapter."""
    model.eval()
    model.config.use_cache = True
    rows = [json.loads(line) for line in eval_path.open()][:N_SAMPLES]
    with out_path.open("w") as f:
        for r in rows:
            enc = tokenizer.apply_chat_template(
                r["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            out = model.generate(**enc, max_new_tokens=256, do_sample=False)
            text = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            f.write(
                json.dumps(
                    {
                        "source": r.get("source"),
                        "prompt_tail": r["prompt"][-1]["content"][-300:],
                        "reference": r["completion"][0]["content"][:500],
                        "generated": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None, help="default: <domain>-r<rank>-s<steps>")
    parser.add_argument("--overwrite", action="store_true", help="replace existing adapter")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    domain, lcfg, tcfg = cfg["domain"], cfg["lora"], cfg["train"]
    run = args.run_name or f"{domain}-r{lcfg['r']}-s{tcfg['max_steps']}"
    set_seed(cfg["seed"])

    ckpt_dir = Path(os.environ["AR_DATA"]) / "checkpoints" / domain / run
    final_dir = paths.adapters_dir() / domain / run
    if final_dir.exists() and not args.overwrite:
        raise SystemExit(f"{final_dir} already exists. Use --overwrite or a new --run-name.")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(paths.model_dir())
    model = load_base_4bit(paths.model_dir())
    train_ds, eval_ds = load_split(domain, "train"), load_split(domain, "eval")

    sft_args = SFTConfig(
        output_dir=str(ckpt_dir),
        max_length=tcfg["max_length"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_steps=max(1, int(tcfg["warmup_frac"] * tcfg["max_steps"])),
        max_steps=tcfg["max_steps"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        optim=tcfg["optim"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=cfg["seed"],
    )
    lora = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()

    resume = get_last_checkpoint(str(ckpt_dir))
    if resume:
        print(f"Resuming from {resume}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    wall = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    eval_metrics = trainer.evaluate()

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    stats_path = paths.prepared_dir() / domain / "stats.json"
    meta = {
        "domain": domain,
        "run": run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "base_model_id": paths.model_id(),
        "base_model_dir": str(paths.model_dir()),
        "config": cfg,
        "data_stats": json.loads(stats_path.read_text()) if stats_path.exists() else None,
        "wall_time_s_this_session": round(wall, 1),
        "resumed_from": resume,
        "peak_gpu_mem_gb": round(peak_gb, 2),
        "final_train_loss": losses[-1] if losses else None,
        "eval": eval_metrics,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
        },
    }
    (final_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (final_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    write_samples(
        trainer.model,
        tokenizer,
        paths.prepared_dir() / domain / "eval.jsonl",
        final_dir / "samples.jsonl",
    )

    current = paths.adapters_dir() / domain / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(final_dir.name)
    print(f"Adapter saved to {final_dir} (current -> {final_dir.name})")
    print(json.dumps({k: meta[k] for k in ("peak_gpu_mem_gb", "final_train_loss")}, indent=2))


if __name__ == "__main__":
    main()
