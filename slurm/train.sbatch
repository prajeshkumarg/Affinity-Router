#!/bin/bash
#SBATCH --job-name=ar-train
#SBATCH --account=csci_ga_3033_085-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# QLoRA-trains one domain adapter on 1x A100 40GB. Resumes from the last checkpoint if rerun.
# Submit from the repo root:
#   sbatch slurm/train.sbatch finance
#   sbatch slurm/train.sbatch coding

DOMAIN="${1:?usage: sbatch slurm/train.sbatch <finance|coding>}"
source slurm/common.sh

nvidia-smi --query-gpu=name,memory.total --format=csv
# Model and data were downloaded beforehand (docs/setup-hpc.md step 1); GPU jobs run offline.
run_in_env "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m finetune.train_lora --config configs/finetune/${DOMAIN}.yaml"

echo "done $(date -Is)"
