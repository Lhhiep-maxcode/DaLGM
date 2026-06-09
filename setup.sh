#!/usr/bin/env bash
set -e

echo "========================================"
echo " DaLGM environment setup"
echo "========================================"

# -------- CONFIG --------
CUDA_VERSION=${1:-13.0}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"
REPO_URL="https://github.com/Lhhiep-maxcode/DaLGM.git"
MODEL_URL="https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors"
# ------------------------

echo "[1/9] Install system dependencies"
apt-get update -qq
apt-get install -y \
    git unzip zip tmux wget curl build-essential ninja-build \
    libgl1 libglib2.0-0 ffmpeg \
    libopengl0

echo "[2/9] Python environment"
pip install --upgrade pip setuptools wheel

echo "[3/9] Install PyTorch + TorchVision"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

echo "[4/9] Install xFormers"
pip install xformers --index-url "$TORCH_INDEX_URL"

echo "[5/9] Install diff-gaussian-rasterization"
if [ ! -d "diff-gaussian-rasterization" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
else
    echo "diff-gaussian-rasterization already exists, skipping clone"
fi
pip install ./diff-gaussian-rasterization --no-build-isolation

echo "[6/9] Install nvdiffrast from source"
pip uninstall nvdiffrast -y || true
pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

echo "[7/9] Install Python requirements"
pip install -r requirements.txt

# Patch nerfacc for CUDA 13.0 / C++20 compatibility
NERFACC_MATH=$(python -c "import nerfacc; import os; print(os.path.join(os.path.dirname(nerfacc.__file__), 'cuda/csrc/include/utils_math.cuh'))")
if [ -f "$NERFACC_MATH" ]; then
    sed -i 's/inline __device__ __host__ float lerp(float a, float b, float t)/inline __device__ __host__ float nerfacc_lerp(float a, float b, float t)/g' "$NERFACC_MATH"
    echo "Patched nerfacc utils_math.cuh"
else
    echo "[WARN] nerfacc utils_math.cuh not found at $NERFACC_MATH"
fi
rm -rf /root/.cache/torch_extensions/*/nerfacc_cuda

echo "[8/9] Download pretrained model"
mkdir -p pretrained && cd pretrained
if [ ! -f "model_fp16_fixrot.safetensors" ]; then
    wget $MODEL_URL
else
    echo "Pretrained model already exists, skipping download"
fi
cd ..

mkdir -p best_phase1 && cd best_phase1
pip install gdown
gdown 1t1HkFyPrvCdMgmQi__cIEx1pnf4hbWtf
cd ..

echo "[9/9] Download data from Kaggle"
pip install kaggle
mkdir -p 10k-dataset-9-views
kaggle datasets download laihoanghiep/10k-dataset-9-views-depth
unzip 10k-dataset-9-views-depth.zip -d 10k-dataset-9-views
rm 10k-dataset-9-views-depth.zip

kaggle datasets download laihoanghiep/10k-dataset-9-views
unzip 10k-dataset-9-views.zip -d 10k-dataset-9-views
rm 10k-dataset-9-views.zip

echo "========================================"
echo " Verify installation"
echo "========================================"
python - <<'PY'
import torch, torchvision, torchmetrics
print("Torch version:   ", torch.__version__)
print("CUDA available:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:             ", torch.cuda.get_device_name(0))
print("torchvision:     ", torchvision.__version__)
print("torchmetrics:    ", torchmetrics.__version__)
PY

echo "========================================"
echo " ✅ Setup completed successfully!"
echo "========================================"