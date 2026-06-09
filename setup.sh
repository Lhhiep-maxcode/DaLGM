#!/usr/bin/env bash
set -e

echo "========================================"
echo " DaLGM environment setup"
echo "========================================"

# -------- CONFIG --------
CUDA_VERSION=${1:-13.0}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION/./}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="https://github.com/Lhhiep-maxcode/DaLGM.git"
REPO_BRANCH="adaptive-LGM-with-depth"
MODEL_URL="https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors"
# ------------------------

echo "[1/9] Install system dependencies"
apt-get update -qq
apt-get install -y \
    git unzip zip tmux wget curl build-essential ninja-build \
    libgl1 libglib2.0-0 ffmpeg \
    libopengl0

echo "[2/9] Python environment"
if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
    echo "Activated /venv/main"
else
    echo "No /venv/main found; using current Python."
fi
python -V && which python
pip install --upgrade pip setuptools wheel

echo "[3/9] Install PyTorch + TorchVision"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

echo "[4/9] Install xFormers"
pip install xformers --index-url "$TORCH_INDEX_URL"

echo "[5/9] Install diff-gaussian-rasterization"
if [ ! -d "$WORKSPACE/diff-gaussian-rasterization/.git" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization "$WORKSPACE/diff-gaussian-rasterization"
else
    echo "diff-gaussian-rasterization already exists, skipping clone"
fi
pip install "$WORKSPACE/diff-gaussian-rasterization" --no-build-isolation

echo "[6/9] Install nvdiffrast from source"
pip uninstall nvdiffrast -y || true
pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

echo "[7/9] Install Python requirements"
pip install -r requirements.txt
pip install torchmetrics

echo "[8/9] Download pretrained model"
mkdir -p pretrained && cd pretrained
if [ ! -f "model_fp16_fixrot.safetensors" ]; then
    wget "$MODEL_URL"
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
kaggle datasets download laihoanghiep/10k-dataset-9-views
unzip 10k-dataset-9-views-depth.zip -d 10k-dataset-9-views
unzip 10k-dataset-9-views.zip -d 10k-dataset-9-views

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