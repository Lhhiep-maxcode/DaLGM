#!/usr/bin/env bash
set -e  # stop immediately if any command fails

echo "========================================"
echo " DaLGM environment setup"
echo "========================================"

# -------- CONFIG --------
CUDA_VERSION=${1:-13.0}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"
REPO_URL="https://github.com/Lhhiep-maxcode/DaLGM.git"
REPO_BRANCH="adaptive-LGM-with-depth"
MODEL_URL="https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors"
# ------------------------

echo "[1/7] Install PyTorch + TorchVision"
pip install torch torchvision --index-url $TORCH_INDEX_URL

echo "[2/7] Clone DaLGM repository"

echo "Repository already exists, skipping clone"


echo "[3/7] Install xFormers"
pip install xformers --index-url $TORCH_INDEX_URL

echo "[4/7] Clone diff-gaussian-rasterization"
if [ ! -d "diff-gaussian-rasterization" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
else
    echo "diff-gaussian-rasterization already exists, skipping clone"
fi

echo "[5/7] Install diff-gaussian-rasterization"
pip install ./diff-gaussian-rasterization --no-build-isolation

echo "[6/7] Install nvdiffrast wheel"
pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

echo "[7/7] Install Python requirements"
pip install -r requirements.txt

echo "========================================"
echo " ✅ Setup completed successfully!"
echo "========================================"
