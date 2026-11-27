#!/usr/bin/env bash
# One-time environment build. Run INSIDE Singularity with the overlay mounted read-write:
#
#   singularity exec --overlay "$OVERLAY":rw "$SIF" /bin/bash
#   bash env/setup_overlay.sh
#
# Installs Miniforge into /ext3, installs the project packages, and writes
# env/requirements.lock.txt so both teammates get identical versions.
set -euo pipefail

if [[ ! -d /ext3 ]]; then
  echo "ERROR: /ext3 not found. Run this inside Singularity with --overlay <file>:rw" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO_DIR/env/requirements.lock.txt"

if [[ ! -x /ext3/miniforge3/bin/python ]]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p /ext3/miniforge3
  rm /tmp/miniforge.sh
fi

cat > /ext3/env.sh <<'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
if [[ -n "${AR_REPO:-}" ]]; then export PYTHONPATH="$AR_REPO:${PYTHONPATH:-}"; fi
EOF

# shellcheck disable=SC1091
source /ext3/env.sh
python -m pip install --upgrade pip

if [[ -f "$LOCK" ]]; then
  echo "Installing from lock file: $LOCK"
  pip install -r "$LOCK"
else
  echo "No lock file yet: installing latest compatible versions, then locking."
  # vLLM first: it pins torch, and everything else resolves around it.
  pip install vllm
  pip install transformers peft trl bitsandbytes accelerate datasets huggingface_hub \
              pyyaml pandas httpx ruff pytest
  pip freeze > "$LOCK"
  echo "Wrote $LOCK. Commit it so the other environment matches."
fi

python - <<'EOF'
import torch, transformers, peft, trl, vllm
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| trl", trl.__version__, "| vllm", vllm.__version__)
EOF
