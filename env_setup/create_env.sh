#!/bin/bash
# =============================================================================
# Conda Environment Setup: dvrk_multimodal_process
# =============================================================================
#
# Creates a unified conda environment for ALL processing pipelines:
#   - Raw image rectify / resize
#   - Kinematic mapping (hand-eye & dVRK)
#   - Depth estimation (FoundationStereo)
#   - Optical flow (Farneback & RAFT)
#   - Data annotation GUIs (PyQt5)
#   - Data export (convert_open_h.py)
#   - Image ↔ Video conversion (FFmpeg subprocess)
#
# Prerequisites:
#   - conda (Miniconda or Anaconda)
#   - NVIDIA GPU with CUDA 12.x driver (nvidia-smi must show CUDA 12.x)
#   - CUDA Toolkit 12.x (only required if you want flash-attn to compile)
#   - ~10 GB free disk space
#   - Internet (PyPI + conda-forge + PyTorch index)
#
# Usage:
#   cd <project_root>
#   bash env_setup/create_env.sh
#
# Expected runtime: ~10-15 min (flash-attn compile dominates).
#
# What the script does, at a glance:
#   1. Make sure libmamba is the active conda solver — without it, the
#      ffmpeg conda-forge step can stall for many minutes. See INSTALL.md
#      § "Why libmamba?" for the full rationale.
#   2. Create the env at Python 3.9.
#   3. Install ffmpeg from conda-forge (binary the conversion scripts
#      call via subprocess).
#   4. Install everything else from pip:
#        - PyTorch 2.4.1 + CUDA 12.1 from PyTorch's wheel index
#        - xformers, flash-attn (must match torch + CUDA exactly)
#        - numpy 1.23.5 pinned BEFORE OpenCV (opencv-contrib-python-headless
#          was built against the numpy-1.x ABI and crashes on numpy 2.x)
#        - OpenCV (headless variant — no bundled Qt; cv2.imshow does NOT work,
#          annotation GUIs use PyQt5 windows directly)
#        - PyQt5 (ships its own Qt 5.15.18 inside the wheel)
#        - everything else (hydra, pandas, pyarrow, scikit-image, etc.)
#   5. Editable-install the local `dvrk_data_processing` package.
#   6. Verify imports + GPU + FFmpeg.
#
# =============================================================================

set -e  # Exit immediately on any failure.

ENV_NAME="dvrk_multimodal_process"
PYTHON_VERSION="3.9"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# -----------------------------------------------------------------------------
# Portability pre-flight checks — run these BEFORE doing any work, so a fresh
# Ubuntu 20.04 box with a half-set-up environment gets a friendly error
# instead of a cryptic failure ten minutes into the install.
#
# Checked:
#   - `conda` is on PATH and callable
#   - `nvidia-smi` is available (warn-only — CPU-only use cases still work)
#   - `nvcc` is on PATH (warn-only — flash-attn will be skipped if absent)
#
# Anything that is a HARD requirement aborts the script; soft requirements
# print a warning and continue.
# -----------------------------------------------------------------------------
echo ""
echo "[-1/11] Pre-flight environment checks..."

# (a) HARD: conda must be on PATH. On a fresh Miniconda install the user
# needs to either source `conda.sh` or run `conda init bash && exec bash`
# before this script will work. We can't do that for them (modifying their
# .bashrc is too invasive), but we can fail loudly with the fix.
if ! command -v conda >/dev/null 2>&1; then
    cat <<'ERR' >&2
ERROR: `conda` not found on PATH.

Possible fixes:
  - If you have not installed conda yet:
        # Miniconda quick install:
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash
        # then open a new terminal and re-run this script.

  - If conda IS installed but not initialized in this shell:
        source ~/miniconda3/etc/profile.d/conda.sh      # adjust path
        # (or run `conda init bash` once, then open a new terminal)
ERR
    exit 1
fi
echo "       conda    : $(command -v conda) ($(conda --version 2>/dev/null))"

# (b) SOFT: nvidia-smi presence. If absent, the GPU-dependent pipelines
# (depth estimation, RAFT optical flow) will not work — but annotation
# GUIs, raw image processing, kinematic mapping, and data conversion all
# do, so we don't abort.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "       nvidia-smi: $(command -v nvidia-smi)  (${GPU_INFO})"
else
    cat <<'WARN' >&2
WARNING: `nvidia-smi` not found. Assuming no NVIDIA GPU on this machine.
         - PyTorch will install (CPU + CUDA wheels are bundled together)
           but torch.cuda.is_available() will be False.
         - flash-attn compile will be skipped automatically.
         - Depth estimation (FoundationStereo) and RAFT optical flow
           require a GPU and will not run.
         - Annotation GUIs, raw image processing, kinematic mapping,
           and data conversion DO work CPU-only.
WARN
fi

# (c) SOFT: nvcc presence. flash-attn compiles CUDA kernels from source
# and needs nvcc. If missing, we let the user know now (rather than 8
# minutes into the flash-attn compile when it dies with a cryptic
# "CUDA_HOME not set" / "Cannot find nvcc" error).
if command -v nvcc >/dev/null 2>&1; then
    echo "       nvcc     : $(command -v nvcc) ($(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1))"
else
    cat <<'WARN' >&2
WARNING: `nvcc` not found on PATH.
         flash-attn compile WILL FAIL (depth estimation still works,
         just slower). To fix BEFORE running this script:

             sudo apt install -y nvidia-cuda-toolkit          # quick path
             # OR for the latest CUDA 12.x:
             #   https://developer.nvidia.com/cuda-downloads
             #   then add /usr/local/cuda/bin to PATH:
             #     export PATH=/usr/local/cuda/bin:$PATH

WARN
fi

# -----------------------------------------------------------------------------
# Pip constraints file — pins numpy at 1.23.5 across every pip-install step.
#
# Why this is needed: pip's resolver re-evaluates the dependency graph on
# every `pip install` call. Without constraints, a later install (e.g.
# scikit-image, pandas, scipy) that depends on `numpy` can silently UPGRADE
# numpy to 2.x, even though we explicitly `pip install numpy==1.23.5` earlier.
# When that happens, the OpenCV 4.5 binary (compiled against the numpy-1.x
# ABI) imports with:
#     "A module compiled using NumPy 1.x cannot be run in NumPy 2.0.2 …"
#     "ImportError: numpy.core.multiarray failed to import"
#
# A pip constraints file (`-c <file>`) tells pip: "If you install any version
# of these packages, pin them to these versions." Pip will then backtrack
# through alternative versions of the *other* packages to find a combination
# that satisfies the constraint, instead of bumping the constrained one.
#
# We create the file once in /tmp and pass it to every `pip install` below.
# The `trap` cleans it up on exit, even if the script fails partway through.
#
# If you add a new pip step below, REMEMBER to include `-c "${PIP_CONSTRAINTS}"`
# or the next person to install the env will be back to debugging numpy 2.x.
# -----------------------------------------------------------------------------
# mktemp portable form — works on GNU coreutils (Ubuntu 20.04+) and BSD
# alike. Don't use `-t <template>`: that's the legacy form on GNU mktemp
# and emits "deprecated" warnings on newer Ubuntu releases.
PIP_CONSTRAINTS="$(mktemp /tmp/surgsync_pip_constraints.XXXXXX.txt)"
trap 'rm -f "${PIP_CONSTRAINTS}"' EXIT
cat > "${PIP_CONSTRAINTS}" <<'CONSTRAINTS'
# Hard-pinned for OpenCV 4.5.3 ABI compatibility. Bumping past 1.x breaks cv2.
numpy==1.23.5
CONSTRAINTS

echo "=============================================="
echo "  Creating conda environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  Project root: ${PROJECT_ROOT}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 0 — Pre-flight: ensure libmamba is the active conda solver.
#
# The single conda-install step below (ffmpeg from conda-forge) pulls in a
# large transitive dep graph (codec libraries, etc.) which is exactly the kind
# of solve where conda's legacy "classic" Python solver hangs for 10-30 min
# on "examining package conflicts" with no visible progress. libmamba (C++,
# libsolv-backed) reduces this to seconds and is the official default for
# conda 23.10+. We install + enable it once here; it's a base-env-level
# setting that persists.
# -----------------------------------------------------------------------------
echo ""
echo "[0/11] Pre-flight: ensuring libmamba solver is active..."
if conda config --show solver 2>/dev/null | grep -q libmamba; then
    echo "       libmamba solver: ACTIVE (skipping install)"
else
    echo "       libmamba solver not active — installing into base env..."
    conda install -n base -c conda-forge conda-libmamba-solver -y
    conda config --set solver libmamba
    echo "       libmamba solver: ACTIVE"
fi

# -----------------------------------------------------------------------------
# Step 1 — Create the env at Python 3.9.
#
# Python 3.9 is required by FoundationStereo (and we've validated the full
# stack on it). 3.10+ is untested for this combination.
# -----------------------------------------------------------------------------
echo ""
echo "[1/11] Creating conda environment with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# -----------------------------------------------------------------------------
# Step 2 — Install ffmpeg from conda-forge.
#
# ffmpeg is the ONE thing we install via conda. The conversion scripts
# (convert_image_to_video.py, convert_video_to_image.py) call `ffmpeg` and
# `ffprobe` via subprocess; putting the binary inside the env avoids depending
# on whatever the system has.
#
# We deliberately do NOT install py-opencv, pyqt, qt-main from conda-forge:
# the well-known "shared Qt with PyQt5" trick is a pain to keep working when
# conda-forge bumps qt-main, and the current setup (pip OpenCV headless +
# pip PyQt5 with bundled Qt) is simpler and demonstrably works.
# -----------------------------------------------------------------------------
echo ""
echo "[2/11] Installing FFmpeg from conda-forge (system binary)..."
conda install -n "${ENV_NAME}" -c conda-forge ffmpeg -y

# -----------------------------------------------------------------------------
# Step 3 — PyTorch 2.4.1 with CUDA 12.1 from the official wheel index.
#
# CUDA 12.1 binaries run fine on CUDA 12.4 drivers (forward-compatible);
# pinning 2.4.1 matches the FoundationStereo + xformers + flash-attn versions
# tested below. Bumping torch breaks at least one of those.
# -----------------------------------------------------------------------------
echo ""
echo "[3/11] Installing PyTorch 2.4.1 (CUDA 12.1)..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url "${PYTORCH_INDEX}"

# -----------------------------------------------------------------------------
# Step 4 — xformers (must match torch 2.4.1 + cu121 exactly).
# -----------------------------------------------------------------------------
echo ""
echo "[4/11] Installing xformers 0.0.28.post1..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    xformers==0.0.28.post1

# -----------------------------------------------------------------------------
# Step 5 — Pin numpy=1.23.5 BEFORE OpenCV.
#
# Two reasons:
#   (a) opencv-contrib-python-headless 4.5.3.56 was compiled against the
#       numpy-1.x ABI. Loading it with numpy 2.x raises:
#         "A module compiled using NumPy 1.x cannot be run in NumPy 2.x"
#       (segfault / ImportError, depending on which symbol triggers).
#   (b) Several downstream pip installs (scipy, pandas, etc.) would otherwise
#       pull in the latest numpy as a transitive dep. Installing 1.23.5 first
#       forces pip's resolver to keep it.
# -----------------------------------------------------------------------------
echo ""
echo "[5/11] Pinning numpy 1.23.5 (OpenCV 4.5 ABI requirement)..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    "numpy==1.23.5"

# -----------------------------------------------------------------------------
# Step 6 — OpenCV (headless variant).
#
# We deliberately use the *headless* package (`opencv-contrib-python-headless`),
# NOT the full `opencv-contrib-python`. Headless means cv2 has no bundled Qt
# (or any GUI backend) — so:
#   - cv2.imshow() does NOT work in this env. Don't use it.
#   - There's no Qt-version conflict with PyQt5 (which bundles its own Qt
#     in the next step), because cv2 has nothing to conflict with.
#
# All annotation GUIs (data_annotate.py, meta_data_annotate.py) are PyQt5
# windows; they never call cv2.imshow.
#
# The `-contrib` package gives us the OpenCV contrib modules (SIFT, SURF,
# tracking, ximgproc, etc.) which several preprocessing scripts depend on.
# -----------------------------------------------------------------------------
echo ""
echo "[6/11] Installing OpenCV (headless, with contrib modules)..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    "opencv-contrib-python-headless==4.5.3.56"

# -----------------------------------------------------------------------------
# Step 7 — PyQt5 (provides its own Qt 5.15.18 via the PyQt5-Qt5 wheel).
#
# PyQt5-Qt5 is the binary Qt distribution wheel — installing PyQt5 from pip
# automatically pulls it in. There's no conflict with OpenCV because OpenCV
# is headless (no Qt of its own).
# -----------------------------------------------------------------------------
echo ""
echo "[7/11] Installing PyQt5 (bundled Qt 5.15.18)..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    "PyQt5==5.15.11" \
    "PyQt5-sip==12.17.1"

# -----------------------------------------------------------------------------
# Step 8 — Core processing dependencies.
# -----------------------------------------------------------------------------
echo ""
echo "[8/11] Installing core processing dependencies..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    "hydra-core>=1.3.2" \
    omegaconf \
    scipy \
    pyyaml \
    tqdm \
    "pandas>=1.3.0" \
    pyarrow

# -----------------------------------------------------------------------------
# Step 9 — Computer vision, 3D, ML packages.
#
# IMPORTANT: do NOT add `opencv-python` or `opencv-contrib-python` (non-headless
# variants) anywhere in here, even transitively. Several packages (`albumentations`,
# `imgaug`) used to pull non-headless OpenCV; modern versions either don't or have
# extras you can opt into. If a pip install accidentally pulls in non-headless
# OpenCV, the Qt-version conflict with PyQt5 returns and the annotation GUIs
# crash with "Could not load the Qt platform plugin 'xcb'". Check after install:
#   pip list | grep opencv     # should ONLY show opencv-contrib-python-headless
# -----------------------------------------------------------------------------
echo ""
echo "[9/11] Installing CV, 3D, and ML packages..."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
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

# -----------------------------------------------------------------------------
# Step 9b — Normalize OpenCV (single variant: contrib-headless 4.5.3.56).
#
# Why this is necessary: several packages from step 9 declare opencv as a
# dependency under different distribution names, and pip happily satisfies
# each independently:
#   - imgaug==0.4.0          requires `opencv-python`            (non-headless)
#   - albumentations         requires `opencv-python-headless`   (headless)
#   - scikit-image           may add `opencv-python-headless`    indirectly
#
# All four opencv distributions (`opencv-python`, `opencv-contrib-python`,
# `opencv-python-headless`, `opencv-contrib-python-headless`) install into
# the SAME `site-packages/cv2/` directory. When multiple variants coexist,
# their files overlap and cv2 imports break with one of:
#     "partially initialized module 'cv2' has no attribute '_registerMatType'"
#     "AttributeError: module 'cv2' has no attribute ..."
#     "ImportError: numpy.core.multiarray failed to import"   (different bug,
#                                                              fixed earlier)
#
# pip constraints (`-c <file>`) can pin *versions* but not *exclude packages*,
# so we can't tell pip "don't install opencv-python" via the constraints file.
# Instead, we let pip install whatever the deps want, then normalize:
#   1) uninstall every opencv variant pip pulled in
#   2) force-reinstall the one we actually want, with --no-deps so its
#      numpy dep doesn't re-trigger the resolver
#
# Result: site-packages/cv2/ contains exactly one variant
# (opencv-contrib-python-headless 4.5.3.56), no overlap, cv2 imports cleanly.
# -----------------------------------------------------------------------------
echo ""
echo "[9b/11] Normalizing OpenCV to single variant (contrib-headless 4.5.3.56)..."
conda run -n "${ENV_NAME}" --no-capture-output pip uninstall -y \
    opencv-python \
    opencv-contrib-python \
    opencv-python-headless \
    opencv-contrib-python-headless 2>&1 | grep -vE "^WARNING: Skipping" || true
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    --no-deps --force-reinstall \
    "opencv-contrib-python-headless==4.5.3.56"

# -----------------------------------------------------------------------------
# Step 10 — flash-attn (optional, compile-from-source, ~5 min).
#
# flash-attn 2.7.4.post1 is pinned for compatibility with GLIBC 2.31 (Ubuntu
# 20.04 default) and with torch 2.4.1 + CUDA 12.1. If the compile fails (CUDA
# toolkit missing, GLIBC mismatch, etc.), depth estimation still works — just
# slower, because FoundationStereo falls back to standard attention.
# -----------------------------------------------------------------------------
echo ""
echo "[10/11] Installing flash-attn (compile-from-source — may take 5+ min)..."
echo "        If this fails, depth estimation still works (slower)."
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    flash-attn==2.7.4.post1 --no-build-isolation || {
    echo ""
    echo "WARNING: flash-attn installation failed."
    echo "  Depth estimation will still work but will be slower."
    echo "  See env_setup/INSTALL.md § Troubleshooting for fix steps."
    echo ""
}

# -----------------------------------------------------------------------------
# Step 11 — Editable-install the local dvrk_data_processing package.
# -----------------------------------------------------------------------------
echo ""
echo "[11/11] Installing local dvrk_data_processing package (editable)..."
cd "${PROJECT_ROOT}"
conda run -n "${ENV_NAME}" --no-capture-output pip install \
    -c "${PIP_CONSTRAINTS}" \
    -e .

# -----------------------------------------------------------------------------
# Verification — comprehensive import + version check.
#
# Reports honest values, including:
#   - cv2 has no GUI backend (correct: headless install)
#   - Python / PyTorch / CUDA / OpenCV / PyQt5 / FFmpeg versions
#   - FoundationStereo importability (sys.path-dependent; warns rather than aborts)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Verifying installation..."
echo "=============================================="
conda run -n "${ENV_NAME}" --no-capture-output python << PY
import sys, shutil
print(f'Python      : {sys.version.split()[0]}')

# Numpy ABI guard: opencv-contrib-python-headless 4.5.3.56 was compiled
# against the numpy-1.x ABI; importing cv2 against numpy 2.x crashes with
# "A module compiled using NumPy 1.x cannot be run in NumPy 2.x".
# Fail loudly here with an actionable error instead of letting cv2 raise
# the cryptic "numpy.core.multiarray failed to import" backtrace.
import numpy as np
print(f'NumPy       : {np.__version__}')
if not np.__version__.startswith('1.'):
    sys.exit(
        f"FATAL: numpy is {np.__version__} but OpenCV 4.5 requires numpy 1.x. "
        "One of the pip-install steps lost the constraints-file pin. Re-run "
        "the script after `pip install -c <constraints> numpy==1.23.5 --force-reinstall`."
    )

# OpenCV variant guard: only one of the four opencv distributions
# (opencv-python, opencv-contrib-python, opencv-python-headless,
# opencv-contrib-python-headless) may be installed at a time. Multiple
# variants overlap in site-packages/cv2/ and break cv2 imports with
# `_registerMatType` circular-import or similar errors.
import importlib.metadata as _md
_opencv_variants = sorted(
    d.metadata['Name'] for d in _md.distributions()
    if d.metadata['Name'] and d.metadata['Name'].lower().startswith('opencv-')
)
if len(_opencv_variants) != 1 or _opencv_variants[0] != 'opencv-contrib-python-headless':
    sys.exit(
        f"FATAL: expected exactly one opencv variant (opencv-contrib-python-headless), "
        f"found {_opencv_variants!r}. Run: pip uninstall {' '.join(_opencv_variants)} -y && "
        "pip install --no-deps --force-reinstall opencv-contrib-python-headless==4.5.3.56"
    )

import torch
print(f'PyTorch     : {torch.__version__}')
print(f'CUDA build  : {torch.version.cuda}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')

import cv2
print(f'OpenCV      : {cv2.__version__}  (headless — cv2.imshow does NOT work; use PyQt5 windows)')

import torchvision
print(f'TorchVision : {torchvision.__version__}  (includes RAFT)')

import hydra
print(f'Hydra       : {hydra.__version__}')

from scipy.spatial.transform import Rotation  # noqa: F401
import pandas, pyarrow
print(f'Pandas      : {pandas.__version__}  /  PyArrow: {pyarrow.__version__}')

from PyQt5.QtWidgets import QApplication  # noqa: F401
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print(f'PyQt5       : {PYQT_VERSION_STR}  (Qt {QT_VERSION_STR})')

import dvrk_data_processing
print(f'dvrk_data_processing: importable (editable install OK)')

ffmpeg_path  = shutil.which('ffmpeg')
ffprobe_path = shutil.which('ffprobe')
print(f'ffmpeg      : {ffmpeg_path or "NOT FOUND"}')
print(f'ffprobe     : {ffprobe_path or "NOT FOUND"}')

# FoundationStereo lives under the repo root, not site-packages; importing it
# requires sys.path manipulation. Try here so the user sees an early warning
# if the submodule wasn't initialized.
sys.path.append('${PROJECT_ROOT}/FoundationStereo')
try:
    from core.utils.utils import InputPadder  # noqa: F401
    from Utils import vis_disparity, set_logging_format, set_seed  # noqa: F401
    print(f'FoundationStereo: importable')
except ImportError as e:
    print(f'FoundationStereo: NOT importable ({e}) — did you clone with --recursive?')

# flash-attn is optional; report whether it landed.
try:
    import flash_attn
    print(f'flash-attn  : {flash_attn.__version__}')
except ImportError:
    print(f'flash-attn  : NOT INSTALLED (optional; depth estimation slower without it)')
PY

echo ""
echo "=============================================="
echo "  Environment '${ENV_NAME}' created successfully!"
echo ""
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "=============================================="
