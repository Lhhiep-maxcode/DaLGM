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

echo "[1/9] Install PyTorch + TorchVision"
pip install torch torchvision --index-url $TORCH_INDEX_URL

echo "[2/9] Clone DaLGM repository"

echo "Repository already exists, skipping clone"


echo "[3/9] Install xFormers"
pip install xformers --index-url $TORCH_INDEX_URL

echo "[4/9] Clone diff-gaussian-rasterization"
if [ ! -d "diff-gaussian-rasterization" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
else
    echo "diff-gaussian-rasterization already exists, skipping clone"
fi

echo "[5/9] Install diff-gaussian-rasterization"
pip install ./diff-gaussian-rasterization --no-build-isolation

echo "[6/9] Install nvdiffrast wheel"
pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

echo "[7/9] Install Python requirements"
pip install -r requirements.txt

echo "[8/9] Download pretrained model"
mkdir -p pretrained
cd pretrained

if [ ! -f "model_fp16_fixrot.safetensors" ]; then
    wget $MODEL_URL
else
    echo "Pretrained model already exists, skipping download"
fi

cd ..

mkdir -p best_phase1
cd best_phase1
pip install gdown
gdown 1t1HkFyPrvCdMgmQi__cIEx1pnf4hbWtf
cd ..

cd ..

echo "[9/9] Download Data"
pip install kaggle

echo "Download depth data from Kaggle"
mkdir 10k-dataset-9-views
kaggle datasets download laihoanghiep/10k-dataset-9-views-depth
kaggle datasets download laihoanghiep/10k-dataset-9-views

unzip 10k-dataset-9-views-depth.zip -d 10k-dataset-9-views
unzip 10k-dataset-9-views.zip -d 10k-dataset-9-views


python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "========================================"
echo " ✅ Setup completed successfully!"
echo "========================================"
