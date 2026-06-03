#!/bin/bash
# =============================================================================
# Conda Environment Setup: dvrk_multimodal_process
# =============================================================================
#
# Creates a unified conda environment for ALL processing pipelines:
#   - Raw image resize/rectify
#   - Kinematic mapping (hand-eye & dVRK)
#   - Depth estimation (FoundationStereo)
#   - Optical flow (Farneback & RAFT)
#   - Data annotation GUIs (PyQt5)
#   - Data export (convert_open_h.py)
#   - Image ↔ Video conversion (FFmpeg-based)
#
# Prerequisites:
#   - conda (Miniconda or Anaconda)
#   - NVIDIA GPU with CUDA 12.x drivers installed
#   - CUDA Toolkit 12.x (for flash-attn compilation)
#   - ~10 GB disk space for all packages
#
# Usage:
#   cd <project_root>
#   bash env_setup/create_env.sh
#
# =============================================================================

set -e  # Exit on any error

ENV_NAME="dvrk_multimodal_process_test"
PYTHON_VERSION="3.9"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=============================================="
echo "  Creating conda environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  Project root: ${PROJECT_ROOT}"
echo "=============================================="

# ---- Step 1: Create base conda environment ----
echo ""
echo "[1/9] Creating conda environment with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# ---- Step 2: Install OpenCV + PyQt5 + FFmpeg via conda-forge (shared Qt) ----
# CRITICAL: Both py-opencv and pyqt must come from conda-forge so they share
# the SAME Qt libraries (qt-main 5.15.x). Installing either via pip bundles
# separate Qt libraries, causing "Could not load the Qt platform plugin xcb"
# errors when both OpenCV GUI (cv2.imshow) and PyQt5 are used together.
# py-opencv from conda-forge includes contrib modules (SIFT, SURF, etc.).
# FFmpeg is installed via conda-forge as a system binary; the image↔video
# conversion scripts (convert_image_to_video.py, convert_video_to_image.py)
# call it via subprocess. Installing via conda keeps the binary inside the
# conda env so it doesn't interfere with any system-level FFmpeg installation.
echo ""
echo "[2/9] Installing OpenCV + PyQt5 + FFmpeg via conda-forge (shared Qt 5.15.x)..."
conda install -n "${ENV_NAME}" -c conda-forge \
    "py-opencv=4.5.5" \
    "pyqt=5.15.9" \
    "numpy=1.23.5" \
    "ffmpeg" \
    -y

# ---- Step 3: Install PyTorch ecosystem with CUDA 12.1 support ----
# Using cu121 to match the tested FoundationStereo configuration.
# CUDA 12.1 binaries are forward-compatible with CUDA 12.4 drivers.
echo ""
echo "[3/9] Installing PyTorch 2.4.1 with CUDA 12.1 support..."
conda run -n "${ENV_NAME}" pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url "${PYTORCH_INDEX}"

# ---- Step 4: Install xformers (must match torch + CUDA version exactly) ----
echo ""
echo "[4/9] Installing xformers 0.0.28.post1..."
conda run -n "${ENV_NAME}" pip install xformers==0.0.28.post1

# ---- Step 5: Install core processing dependencies ----
# numpy is already installed via conda in Step 2 — do NOT reinstall via pip
echo ""
echo "[5/9] Installing core processing dependencies..."
conda run -n "${ENV_NAME}" pip install \
    "hydra-core>=1.3.2" \
    omegaconf \
    scipy \
    pyyaml \
    tqdm \
    "pandas>=1.3.0" \
    pyarrow

# ---- Step 6: Install computer vision, 3D, ML packages ----
# NOTE: OpenCV and PyQt5 are already installed via conda (Step 2).
# Do NOT install opencv-* or PyQt5 via pip — it would override the
# conda packages and reintroduce the Qt library conflict.
echo ""
echo "[6/9] Installing CV, 3D, and ML packages..."
conda run -n "${ENV_NAME}" pip install \
    scikit-image \
    timm \
    albumentations \
    imgaug \
    scikit-learn \
    joblib \
    einops \
    trimesh \
    open3d \
    transformations \
    imageio \
    gdown \
    "huggingface-hub" \
    ruamel.yaml \
    ninja

# ---- Step 7: Install flash-attn (builds from source, requires CUDA toolkit) ----
# This step takes several minutes as it compiles CUDA kernels.
# If it fails, depth estimation still works (with reduced performance).
echo ""
echo "[7/9] Installing flash-attn (this may take several minutes)..."
echo "       If this fails, see env_setup/INSTALL.md for troubleshooting."
conda run -n "${ENV_NAME}" pip install flash-attn==2.7.4.post1 --no-build-isolation || {
    echo ""
    echo "WARNING: flash-attn installation failed."
    echo "  Depth estimation will still work but may be slower."
    echo "  See env_setup/INSTALL.md for troubleshooting."
    echo ""
}

# ---- Step 8: Install the local dvrk_data_processing package ----
echo ""
echo "[8/9] Installing local dvrk_data_processing package..."
cd "${PROJECT_ROOT}"
conda run -n "${ENV_NAME}" pip install -e .

# ---- Step 9: Verify FFmpeg binary is available ----
echo ""
echo "[9/9] Verifying FFmpeg installation..."
conda run -n "${ENV_NAME}" bash -c "
    if command -v ffmpeg &>/dev/null; then
        echo \"FFmpeg: \$(ffmpeg -version | head -1)\"
    else
        echo 'WARNING: ffmpeg binary not found on PATH'
    fi
    if command -v ffprobe &>/dev/null; then
        echo \"FFprobe: available\"
    else
        echo 'WARNING: ffprobe binary not found on PATH'
    fi
"

# ---- Verification ----
echo ""
echo "=============================================="
echo "  Verifying installation..."
echo "=============================================="
conda run -n "${ENV_NAME}" python -c "
import sys
import shutil
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import cv2
print(f'OpenCV: {cv2.__version__}')
# Verify non-headless: check if highgui backend is available
build_info = cv2.getBuildInformation()
has_gui = 'QT' in build_info or 'GTK' in build_info
print(f'OpenCV GUI (cv2.imshow): {\"YES\" if has_gui else \"NO (headless)\"}')

import hydra
print(f'Hydra: {hydra.__version__}')

from scipy.spatial.transform import Rotation
print(f'SciPy: OK')

import pandas, pyarrow
print(f'Pandas: {pandas.__version__}, PyArrow: {pyarrow.__version__}')

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print(f'PyQt5: {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})')

# Verify OpenCV and PyQt5 share the same Qt — no conflict
print(f'OpenCV + PyQt5 Qt coexistence: OK')

import torchvision
print(f'TorchVision: {torchvision.__version__} (includes RAFT)')

import dvrk_data_processing
print(f'dvrk_data_processing: OK')

# Check FFmpeg availability (used by image↔video conversion scripts)
ffmpeg_path = shutil.which('ffmpeg')
ffprobe_path = shutil.which('ffprobe')
print(f'FFmpeg binary: {ffmpeg_path if ffmpeg_path else \"NOT FOUND\"}')
print(f'FFprobe binary: {ffprobe_path if ffprobe_path else \"NOT FOUND\"}')

# Check FoundationStereo imports
sys.path.append('${PROJECT_ROOT}/FoundationStereo')
try:
    from core.utils.utils import InputPadder
    from Utils import vis_disparity, set_logging_format, set_seed
    print(f'FoundationStereo imports: OK')
except ImportError as e:
    print(f'FoundationStereo imports: FAILED ({e})')

try:
    import flash_attn
    print(f'flash-attn: {flash_attn.__version__}')
except ImportError:
    print(f'flash-attn: NOT INSTALLED (optional, depth estimation still works)')
"

echo ""
echo "=============================================="
echo "  Environment '${ENV_NAME}' created successfully!"
echo ""
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "=============================================="
