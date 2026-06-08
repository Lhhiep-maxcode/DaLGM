#!/usr/bin/env bash
set -e  # stop immediately if any command fails

echo "========================================"
echo " DaLGM eval environment setup"
echo "========================================"

# -------- CONFIG --------
CUDA_VERSION=${1:-13.0}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"
REPO_URL="https://github.com/Lhhiep-maxcode/DaLGM.git"
REPO_BRANCH="adaptive-LGM-with-depth"
MODEL_URL="https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors"
# ------------------------

echo "[1/8] Install system dependencies"
apt-get update -qq
apt-get install -y \
    git build-essential ninja-build \
    libgl1 libglib2.0-0 \
    libopengl0  # required for pymeshlab plugins (libfilter_meshing.so etc.)

echo "[2/8] Install PyTorch + TorchVision"
pip install torch torchvision --index-url $TORCH_INDEX_URL

echo "[3/8] Clone DaLGM repository"
# Uncomment if not already cloned:
# git clone --branch $REPO_BRANCH $REPO_URL
echo "Assuming repository already exists, skipping clone"

echo "[4/8] Install xFormers"
pip install xformers --index-url $TORCH_INDEX_URL

echo "[5/8] Clone & install diff-gaussian-rasterization"
if [ ! -d "diff-gaussian-rasterization" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
else
    echo "diff-gaussian-rasterization already exists, skipping clone"
fi
pip install ./diff-gaussian-rasterization --no-build-isolation

echo "[6/8] Install nvdiffrast from source"
pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

echo "[7/8] Install Python requirements"
pip install -r requirements.txt

echo "[8/8] Verify installation"
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA version:  ", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Verify pymeshlab loads plugins correctly (needs libopengl0)
import pymeshlab
ms = pymeshlab.MeshSet()
assert hasattr(ms, 'meshing_isotropic_explicit_remeshing'), \
    "pymeshlab missing meshing_isotropic_explicit_remeshing — check libopengl0 install"
print("pymeshlab: OK (meshing_isotropic_explicit_remeshing available)")
EOF

echo "========================================"
echo " ✅ Setup completed successfully!"
echo "========================================"