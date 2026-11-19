#!/bin/bash
# One-time setup. Run on the LOGIN node (needs internet), from the repo root:
#   bash slurm/setup.sh
# Compute nodes are often offline, so everything (checkpoints, CIFAR-10, ViT weights)
# is downloaded/cached here first; job.sh then runs with HF_HUB_OFFLINE=1.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
source slurm/env.sh

# --- create venv (no conda). Needs Python >= 3.9.
if [ ! -f "$VENV/bin/activate" ]; then
  PY=${PYTHON:-python3}
  if command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
    "$PY" -m venv "$VENV"
  elif command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
    # system python too old/missing: uv downloads a standalone Python 3.10 into $HOME (no root needed)
    UV=$(command -v uv || echo "$HOME/.local/bin/uv")
    "$UV" venv --python 3.10 --seed "$VENV"
  else
    echo "Need Python >= 3.9. Options:"
    echo "  - module avail python   then   MODULES=python/3.10 bash slurm/setup.sh"
    echo "  - install uv (no root): curl -LsSf https://astral.sh/uv/install.sh | sh   then rerun"
    exit 1
  fi
fi
activate_env
python -m pip install --upgrade pip
# Pick the CUDA wheel matching the node driver (check `nvidia-smi` on a GPU node; cu121 needs driver >= 530).
pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_WHEEL:-cu121}
# transformers 5.x needs torch>=2.6 (not on the cu121 index) -> stay on 4.x
pip install "transformers>=4.40,<5" timm statsmodels scipy "numpy<2.3" gdown
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

# 1) iDDPM 50M CIFAR-10 diffusion checkpoint
[ -f cifar10/cifar10_uncond_50M_500K.pt ] || wget -O cifar10/cifar10_uncond_50M_500K.pt \
  https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt

# 2) Cohen et al. architectures
if [ ! -d cifar10/archs ]; then
  TMP=$(mktemp -d); git clone --depth 1 https://github.com/locuslab/smoothing.git "$TMP/cohen"
  cp -r "$TMP/cohen/code/archs" cifar10/archs; rm -rf "$TMP"
fi

# 3) Cohen et al. noise-trained models (models.tar, Google Drive link in locuslab/smoothing README).
#    Verify the file id against that README; override with COHEN_GDRIVE_ID=... if it differs.
GID=${COHEN_GDRIVE_ID:-1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4}
if [ ! -f models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar ]; then
  gdown "$GID" -O models.tar || { echo "gdown failed: download models.tar manually into $ROOT"; exit 1; }
  tar -xf models.tar   # -> models/cifar10/resnet110/noise_{0.12,0.25,0.50,1.00}/checkpoint.pth.tar
fi

# 4) CIFAR-10 + ViT weights into local caches
mkdir -p data out logs
python - <<'EOF'
from torchvision import datasets
datasets.CIFAR10("data/cifar10", train=False, download=True)
datasets.CIFAR10("data/cifar10", train=True, download=True)
from transformers import AutoModelForImageClassification
AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
print("datasets + ViT cached")
EOF

# 5) Reference Carlini results reproduced by Dual RS (for the reproduction check)
[ -d ../Dual-Randomized-Smoothing ] || \
  git clone --depth 1 https://github.com/xiasong0501/Dual-Randomized-Smoothing.git ../Dual-Randomized-Smoothing \
  || echo "WARN: could not clone Dual RS repo; set REF_DIR manually in submit_all.sh"

# 6) Sanity check: imports + checkpoint loading on CPU
PYTHONPATH=cifar10 python - <<'EOF'
import core  # fails here if binom_test fix is missing
from DRM_cascade import load_cohen_resnet110
for s in ["0.25", "0.50", "1.00"]:
    load_cohen_resnet110(f"models/cifar10/resnet110/noise_{s}/checkpoint.pth.tar")
print("setup OK")
EOF
