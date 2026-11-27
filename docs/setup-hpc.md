# HPC Setup and Adapter Training Runbook

How to go from a fresh Cloud Bursting account to two trained adapters on disk.
Every step runs on NYU HPC Cloud Bursting (through Open OnDemand), never on a laptop.

## Where things live

| What | Path (defaults in `env/paths.sh`) | In git? |
|---|---|---|
| Repo | `/scratch/$USER/affinity-router/` (`$AR_REPO`, detected automatically) | yes |
| Everything large | `$AR_REPO/artifacts/` (`$AR_DATA`) | no (git-ignored) |
| Conda overlay | `artifacts/env/overlay.ext3` | no |
| Base model | `artifacts/models/Llama-3.1-8B-Instruct/` (~16 GB) | no |
| Raw datasets | `artifacts/datasets/<name>/` | no |
| Prepared JSONL | `artifacts/prepared/<domain>/{train,eval}.jsonl`, `stats.json` | no |
| Training checkpoints | `artifacts/checkpoints/<domain>/<run>/` | no |
| **Final adapters** | `artifacts/adapters/<domain>/<run>/`, with `current` → latest run | no |
| Slurm logs | `logs/` | no |

Everything stays inside one folder on scratch. Use the OOD file browser to upload or download files when needed.

## 0. One-time setup

From an OOD terminal (login or interactive CPU session):

```bash
# 1. Clone the repo
cd /scratch/$USER && git clone <repo-url> affinity-router && cd affinity-router

# 2. Pick a Singularity image: CUDA 12.x on Ubuntu 22.04/24.04.
ls /share/apps/images/
#    then set SIF in env/paths.sh to that full path.

# 3. Load paths and create the data dir
source env/paths.sh && mkdir -p "$AR_DATA/env"

# 4. Copy an overlay template (pick >= 25 GB and >= 500K files; conda creates many files)
ls /share/apps/overlay-fs-ext3/
cp /share/apps/overlay-fs-ext3/<template>.ext3.gz "$AR_DATA/env/"
gunzip "$AR_DATA/env/<template>.ext3.gz" && mv "$AR_DATA/env/<template>.ext3" "$OVERLAY"
```

Build the environment. Do this inside a short interactive job, not on the login node: the install is heavy.

```bash
singularity exec --overlay "$OVERLAY":rw "$SIF" /bin/bash
bash env/setup_overlay.sh       # installs Miniforge + packages, writes env/requirements.lock.txt
exit
```

Commit `env/requirements.lock.txt`. Whoever builds the environment second gets the exact same versions from it.

**Hugging Face token** (needs Llama 3.1 access granted on your account):

```bash
singularity exec --overlay "$OVERLAY":ro "$SIF" /bin/bash -c \
  "source /ext3/env.sh && source env/paths.sh && hf auth login"
```

The token is stored under `$HF_HOME` (inside `$AR_DATA`), so batch jobs can use it. Never put a token in the repo.

## 1. Download + prepare data (interactive)

Run this in an OOD terminal on an interactive CPU session, from the repo root:

```bash
source env/paths.sh
singularity exec --overlay "$OVERLAY":ro "$SIF" /bin/bash -c \
  "source /ext3/env.sh && source env/paths.sh && \
   python -m finetune.download && \
   python -m finetune.prepare_data --config configs/finetune/finance.yaml && \
   python -m finetune.prepare_data --config configs/finetune/coding.yaml"
```

`finetune.download` skips datasets that are already on disk, so rerunning it is safe. Use `--skip-model` or `--skip-datasets` to fetch only part.

Check the result:
- `artifacts/models/Llama-3.1-8B-Instruct/` has `config.json`, 4 `*.safetensors` shards and the tokenizer files.
- `artifacts/prepared/finance/stats.json` and `.../coding/stats.json` show the row counts and token lengths.

## 2. Train adapters (GPU jobs, 1x A100 each)

```bash
sbatch slurm/train.sbatch finance
sbatch slurm/train.sbatch coding
squeue -u $USER
```

- Each job is 750 steps at an effective batch of 16 with QLoRA rank 8. Expect roughly 1–2 GPU-hours each.
- If a job dies (time limit, preemption, idle shutdown), resubmit the same command and it resumes from the last checkpoint (saved every 250 steps).
- To retrain on purpose, use a new run name: add `--run-name finance-v2` to the command in `train.sbatch`, or pass `--overwrite`.

## 3. Check the adapters

For each domain, `$AR_DATA/adapters/<domain>/current/` should contain:

| File | What to look at |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | The adapter itself (tens of MB) |
| `train_meta.json` | Peak GPU memory, final train loss, eval loss, versions, git commit |
| `log_history.json` | Loss curve |
| `samples.jsonl` | 3 greedy generations on held-out prompts. Read them. |

Back up the adapters off the cluster: download `artifacts/adapters/` with the OOD file browser. Each one is only tens of MB, and scratch is not backed up.

## 4. What these adapters feed into

The vLLM replicas load both adapters with `--enable-lora --max-lora-rank 8` and `--lora-modules finance=artifacts/adapters/finance/current coding=artifacts/adapters/coding/current`. That's the serving setup for Phase 0's smoke test and every grid run.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `403` when downloading the model | Llama access not granted yet, or `hf auth login` was not run with `env/paths.sh` sourced |
| `ERROR: set SIF in env/paths.sh` | Step 0.2 |
| CUDA out of memory | Lower `per_device_batch_size` to 2 and raise `grad_accum` to 8 (same effective batch) |
| `sbatch: invalid partition` / gres error | Check `sinfo`; the partition names come from the course allocation |
| Job killed for low GPU utilization | Use batch jobs, not idle interactive sessions; resubmit to resume |
