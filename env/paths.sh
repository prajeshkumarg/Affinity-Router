# Source this file on the cluster (login node, Slurm jobs, inside or outside Singularity).
# Override any variable by exporting it before sourcing.

# The repo lives on scratch, e.g. /scratch/$USER/affinity-router.
# Detected from this file's location, so no editing needed.
_AR_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AR_REPO="${AR_REPO:-$_AR_HERE}"

# Large artifacts (overlay, model weights, datasets, adapters) live inside the repo
# under artifacts/, which is git-ignored. Move files on/off the cluster with OOD.
export AR_DATA="${AR_DATA:-$AR_REPO/artifacts}"

# Singularity image and conda overlay. Pick the image with `ls /share/apps/images`
# (a CUDA 12.x + Ubuntu 22.04/24.04 image).
export SIF="${SIF:-/share/apps/images/SET_ME.sif}"
export OVERLAY="${OVERLAY:-$AR_DATA/env/overlay.ext3}"

# Hugging Face cache and our fixed artifact locations.
export HF_HOME="${HF_HOME:-$AR_DATA/hf_cache}"
export AR_MODEL_ID="${AR_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
export AR_MODEL_DIR="${AR_MODEL_DIR:-$AR_DATA/models/Llama-3.1-8B-Instruct}"
export AR_DATASETS_DIR="${AR_DATASETS_DIR:-$AR_DATA/datasets}"
export AR_PREPARED_DIR="${AR_PREPARED_DIR:-$AR_DATA/prepared}"
export AR_ADAPTERS_DIR="${AR_ADAPTERS_DIR:-$AR_DATA/adapters}"

# Slurm account for every job.
export AR_SLURM_ACCOUNT="csci_ga_3033_085-2026fa"
