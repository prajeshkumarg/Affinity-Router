# Shared helpers for Slurm jobs. Sourced by the *.sbatch files.
set -euo pipefail

# Submit jobs from the repo root (sbatch slurm/xyz.sbatch); that directory is the repo.
export AR_REPO="${AR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "$AR_REPO/env/paths.sh"

if [[ "$SIF" == *SET_ME* ]]; then
  echo "ERROR: set SIF in env/paths.sh (see docs/setup-hpc.md)" >&2
  exit 1
fi

mkdir -p "$AR_DATA" "$HF_HOME"

# Run a command inside the Singularity image with the overlay mounted read-only.
# --nv exposes the GPU (harmless on CPU nodes).
run_in_env() {
  singularity exec --nv \
    --overlay "$OVERLAY":ro \
    --bind "$AR_DATA" \
    "$SIF" /bin/bash -c "source /ext3/env.sh && source $AR_REPO/env/paths.sh && cd $AR_REPO && $*"
}

echo "job=${SLURM_JOB_NAME:-local} id=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
