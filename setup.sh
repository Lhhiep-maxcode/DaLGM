#!/usr/bin/env bash
set -e  # stop immediately if any command fails

echo "========================================"
echo " LGM-from-sratch environment setup"
echo "========================================"

# -------- CONFIG --------
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
REPO_URL=https://github.com/Lhhiep-maxcode/LGM-from-sratch.git
REPO_BRANCH=adaptive-LGM-with-depth
MODEL_URL=https://huggingface.co/Hiepppp/LGM/resolve/main/model_fp16_fixrot.safetensors
# ------------------------

echo "[1/9] Install PyTorch + TorchVision + TorchAudio"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url $TORCH_INDEX_URL

echo "[2/9] Clone LGM-from-sratch repository"
if [ ! -d "LGM-from-sratch" ]; then
    git clone --branch $REPO_BRANCH $REPO_URL
else
    echo "Repository already exists, skipping clone"
fi

cd LGM-from-sratch

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
pip install ./wheels/nvdiffrast-0.3.3-py3-none-any.whl

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

echo "[9/9] Verify installation"
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "========================================"
echo " ✅ Setup completed successfully!"
echo " Next steps:"
echo "   cd LGM-from-sratch"
echo "   (activate your env if needed)"
echo "   run your training script"
echo "========================================"
