#!/usr/bin/env bash
set -e

echo "========================================"
echo " DaLGM environment setup (CUDA 13.0)"
echo "========================================"

# -------- CONFIG --------
CUDA_VERSION=${1:-13.0}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
TORCHAUDIO_VERSION="2.9.1"
XFORMERS_VERSION="0.0.33.post2"

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="https://github.com/Lhhiep-maxcode/DaLGM.git"
REPO_BRANCH="adaptive-LGM-with-depth"
MODEL_URL="https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors"
# ------------------------

echo "[1/8] Install system dependencies"
apt-get update -qq
apt-get install -y \
    git unzip zip tmux wget curl build-essential ninja-build \
    libgl1 libglib2.0-0 ffmpeg \
    libopengl0  # required for pymeshlab plugins

echo "[2/8] Python environment"
if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
    echo "Activated /venv/main"
else
    echo "No /venv/main found; using current Python."
fi
python -V && which python
pip install --upgrade pip setuptools wheel

echo "[3/8] Install PyTorch stack (pinned, CUDA 13.0)"
pip uninstall -y torch torchvision torchaudio xformers || true
pip install --no-cache-dir \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"
pip install --no-cache-dir \
    "xformers==$XFORMERS_VERSION" \
    --index-url "$TORCH_INDEX_URL" \
    --no-deps

echo "[4/8] Clone DaLGM repository"
# Uncomment if not already cloned:
# git clone --branch $REPO_BRANCH $REPO_URL
echo "Assuming repository already exists, skipping clone"

echo "[5/8] Install diff-gaussian-rasterization"
if [ ! -d "$WORKSPACE/diff-gaussian-rasterization/.git" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization "$WORKSPACE/diff-gaussian-rasterization"
else
    echo "diff-gaussian-rasterization already exists, skipping clone"
fi
pip install "$WORKSPACE/diff-gaussian-rasterization" --no-build-isolation

echo "[6/8] Install nvdiffrast from source"
# Build from source for CUDA 13.0 compatibility (wheel may not support cu130)
pip uninstall nvdiffrast -y || true
pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

echo "[7/8] Install Python requirements"
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir torchmetrics 


echo "[8/8] Verify installation"
python - <<'PY'
import torch, torchvision, torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
print("Torch version:   ", torch.__version__)
print("CUDA version:    ", torch.version.cuda)
print("CUDA available:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:             ", torch.cuda.get_device_name(0))
print("torchvision:     ", torchvision.__version__)
print("torchmetrics:    ", torchmetrics.__version__)

import pymeshlab
ms = pymeshlab.MeshSet()
assert hasattr(ms, 'meshing_isotropic_explicit_remeshing'), \
    "pymeshlab missing meshing_isotropic_explicit_remeshing — check libopengl0"
print("pymeshlab:        OK")

from core.model import LGM
print("LGM import:       OK")
PY

echo "========================================"
echo " ✅ Setup completed successfully!"
echo "========================================"