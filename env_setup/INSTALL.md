# Environment Setup: `dvrk_multimodal_process`

Unified conda environment for all dVRK multi-modal data processing pipelines.

## What This Environment Supports

| Pipeline | Scripts | Key Dependencies |
|----------|---------|-----------------|
| Raw Image Processing | `gen_resize_rectify.py` | OpenCV, NumPy |
| Kinematic Mapping | `gen_kinematic_heatmap_handeye.py`, `gen_kinematic_heatmap_dVRK.py` | OpenCV, SciPy, Hydra |
| Depth Estimation | `gen_depth_estimate.py` | FoundationStereo, PyTorch+CUDA, flash-attn |
| Optical Flow (Farneback) | `gen_optical_flow.py` | OpenCV |
| Optical Flow (RAFT) | `gen_optical_flow_raft.py` | TorchVision (RAFT models), PyTorch+CUDA |
| Data Annotation GUIs | `data_annotate.py`, `meta_data_annotate.py` | PyQt5, OpenCV |
| Data Export | `convert_open_h.py` | Pandas, PyArrow |
| Image → Video | `convert_image_to_video.py` | FFmpeg (subprocess), OpenCV (fallback) |
| Video → Image | `convert_video_to_image.py` | FFmpeg (subprocess), OpenCV (fallback) |

## Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04)
- **Conda**: Miniconda or Anaconda installed
- **NVIDIA GPU**: With CUDA 12.x compatible drivers (`nvidia-smi` should work)
- **CUDA Toolkit**: 12.x installed at `/usr/local/cuda` (needed for flash-attn compilation)
- **Disk Space**: ~10 GB for all packages
- **Git Submodules**: Repository cloned with `--recursive` (FoundationStereo submodule required)

Check prerequisites:

```bash
# Verify GPU driver
nvidia-smi

# Verify CUDA toolkit
nvcc --version

# Verify submodule
ls FoundationStereo/readme.md
# If missing: git submodule update --init --recursive
```

## Quick Install (Automated)

```bash
cd <project_root>
bash env_setup/create_env.sh
```

The script takes ~10-15 minutes (mostly flash-attn compilation). It will:
1. Create conda env `dvrk_multimodal_process` with Python 3.9
2. Install OpenCV + PyQt5 + FFmpeg via conda-forge (shared Qt, no conflicts)
3. Install PyTorch 2.4.1 with CUDA 12.1 support
4. Install xformers, flash-attn, and all processing dependencies
5. Install `pip install -e .` for the local package
6. Verify FFmpeg binary availability
7. Run verification checks

## Manual Install (Step-by-Step)

If the automated script fails or you need more control:

### Step 1: Create conda environment

```bash
conda create -n dvrk_multimodal_process python=3.9 -y
conda activate dvrk_multimodal_process
```

### Step 2: Install OpenCV + PyQt5 + FFmpeg via conda-forge (shared Qt)

**CRITICAL**: OpenCV and PyQt5 must **both** come from conda-forge so they share the
same Qt libraries (`qt-main 5.15.x`). Installing either via pip bundles separate Qt
libraries that conflict with each other, causing the `"Could not load the Qt platform
plugin xcb"` error when `cv2.imshow()` and PyQt5 are used in the same process.

FFmpeg is installed via conda-forge as a system binary. The image↔video conversion
scripts (`convert_image_to_video.py`, `convert_video_to_image.py`) call `ffmpeg` and
`ffprobe` via subprocess. Installing via conda keeps the binaries inside the conda env
so they don't interfere with any system-level FFmpeg installation.

```bash
# py-opencv from conda-forge includes contrib modules (SIFT, SURF, etc.)
# and is compiled with Qt5 highgui backend (cv2.imshow works)
# ffmpeg provides the ffmpeg and ffprobe binaries for video conversion
conda install -c conda-forge py-opencv=4.5.5 pyqt=5.15.9 numpy=1.23.5 ffmpeg -y
```

Verify:
```bash
python -c "
import cv2
import shutil
print(f'OpenCV: {cv2.__version__}')
# Check that highgui has Qt backend
build_info = cv2.getBuildInformation()
has_gui = 'QT' in build_info or 'GTK' in build_info
print(f'GUI backend: {\"YES\" if has_gui else \"NO\"}')

from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print(f'PyQt5: {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})')

# If this line runs without error, Qt libraries are compatible
from PyQt5.QtWidgets import QApplication
print('OpenCV + PyQt5 coexistence: OK')

# Check FFmpeg binary is available
ffmpeg_path = shutil.which('ffmpeg')
ffprobe_path = shutil.which('ffprobe')
print(f'FFmpeg: {ffmpeg_path if ffmpeg_path else \"NOT FOUND\"}')
print(f'FFprobe: {ffprobe_path if ffprobe_path else \"NOT FOUND\"}')
"
```

### Step 3: Install PyTorch with CUDA support

```bash
# PyTorch 2.4.1 with CUDA 12.1 (forward-compatible with CUDA 12.4 drivers)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.1+cu121 True
```

### Step 4: Install xformers

```bash
# Must match the torch + CUDA version exactly
pip install xformers==0.0.28.post1
```

### Step 5: Install core processing dependencies

```bash
# numpy is already installed via conda in Step 2 — do NOT reinstall via pip
pip install \
    "hydra-core>=1.3.2" \
    omegaconf \
    scipy \
    pyyaml \
    tqdm \
    "pandas>=1.3.0" \
    pyarrow
```

### Step 6: Install computer vision, 3D, and ML packages

**IMPORTANT**: Do NOT install `opencv-*` or `PyQt5` via pip here — they are
already installed via conda (Step 2). Pip versions would override the conda
packages and reintroduce the Qt library conflict.

```bash
pip install \
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
    huggingface-hub \
    ruamel.yaml \
    ninja
```

### Step 7: Install flash-attn (optional but recommended)

```bash
# Version pinned for GLIBC 2.31 compatibility (Ubuntu 20.04)
# Use a cached wheel if available, otherwise compiles CUDA kernels (~5 min)
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Step 8: Install the local package

```bash
cd <project_root>
pip install -e .
```

### Step 9: Verify installation

```bash
python -c "
import shutil
import torch, cv2, hydra, scipy, pandas, pyarrow, tqdm
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
import torchvision
import dvrk_data_processing
print('All core imports OK')
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'OpenCV {cv2.__version__}')
print(f'PyQt5: {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})')
print(f'TorchVision {torchvision.__version__}')
ffmpeg_path = shutil.which('ffmpeg')
print(f'FFmpeg: {ffmpeg_path if ffmpeg_path else \"NOT FOUND\"}')
"
```

## Verification: FoundationStereo

FoundationStereo requires pretrained model weights. To verify depth estimation end-to-end:

1. Download pretrained models from:
   https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf

2. Place in `FoundationStereo/pretrained_models/`:
   ```
   FoundationStereo/pretrained_models/
   ├── 23-51-11/    # ViT-large (best quality)
   │   ├── model.pth
   │   └── cfg.yaml
   └── 11-33-40/    # ViT-small (faster inference)
       ├── model.pth
       └── cfg.yaml
   ```

3. Test import chain:
   ```bash
   conda activate dvrk_multimodal_process
   python -c "
   import sys; sys.path.append('FoundationStereo')
   from core.foundation_stereo import FoundationStereo
   from core.utils.utils import InputPadder
   from Utils import vis_disparity
   print('FoundationStereo: all imports OK')
   "
   ```

## Running Processing Scripts

After activation, all scripts are ready to use:

```bash
conda activate dvrk_multimodal_process

# 1. Resize and rectify (run first)
cd src/dvrk_data_processing/raw_image_processing
python gen_resize_rectify.py

# 2. Kinematic mapping
cd ../kinematic_mapping
python gen_kinematic_heatmap_handeye.py

# 3. Depth estimation (requires GPU + pretrained models)
cd ../depth_estimation
python gen_depth_estimate.py

# 4. Optical flow
cd ../optical_flow
python gen_optical_flow_raft.py     # RAFT (GPU, recommended)
python gen_optical_flow.py          # Farneback (CPU fallback)

# 5. Data annotation GUIs
cd ../data_annotation
python data_annotate.py
python meta_data_annotate.py

# 6. Image ↔ Video conversion (uses FFmpeg with OpenCV fallback)
cd ../../scripts
python convert_image_to_video.py   # image sequence → MP4
python convert_video_to_image.py   # MP4 → image sequence
```

**Important**: Before running any script, edit the `@hydra.main(config_name=...)` decorator
in the script to point to your personal config file. See `config/` for examples.

## Package Version Summary

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| Python | 3.9 | conda | FoundationStereo requirement |
| py-opencv | 4.5.5 | conda-forge | Image processing, calibration, `cv2.imshow()` GUI |
| pyqt | 5.15.9 | conda-forge | Annotation GUI framework |
| qt-main | 5.15.x | conda-forge | Shared Qt libraries (used by both OpenCV and PyQt5) |
| numpy | 1.23.5 | conda-forge | Array computation (pinned for OpenCV compatibility) |
| torch | 2.4.1+cu121 | pip (PyTorch) | GPU computation, RAFT, FoundationStereo |
| torchvision | 0.19.1 | pip (PyTorch) | RAFT optical flow models |
| xformers | 0.0.28.post1 | pip | Efficient attention (FoundationStereo) |
| flash-attn | 2.7.4.post1 | pip | Flash attention (FoundationStereo, pinned for GLIBC 2.31) |
| ffmpeg | latest | conda-forge | Video encoding/decoding (image↔video conversion scripts) |
| hydra-core | >=1.3.2 | pip | Configuration management |
| scipy | latest | pip | Rotation transforms, interpolation |
| pandas + pyarrow | latest | pip | Data export (Parquet format) |
| open3d | latest | pip | 3D point cloud (FoundationStereo) |
| timm | latest | pip | Vision transformer models |

### Why conda-forge for OpenCV + PyQt5?

When installed via pip, `opencv-contrib-python` (non-headless) bundles its own Qt 5.15.0
libraries, while `PyQt5` bundles Qt 5.15.18. Loading two different Qt builds in the same
Python process causes crashes:

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in ".../cv2/qt/plugins"
```

The pip **headless** OpenCV variant avoids this by not bundling Qt at all, but then
`cv2.imshow()` doesn't work. Installing both via conda-forge solves this: conda's
dependency solver ensures a single `qt-main` package is used by both, so they share
the exact same Qt shared libraries with no conflicts. This gives you both `cv2.imshow()`
and PyQt5 working together.

## Troubleshooting

### flash-attn fails to install

```
ERROR: Failed building wheel for flash-attn
```

**Cause**: Missing CUDA toolkit or version mismatch.

**Fix**:
```bash
# Check CUDA toolkit is installed
nvcc --version
# Should show CUDA 12.x

# If nvcc not found, ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Retry installation
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

**Skip**: Depth estimation works without flash-attn (slower performance).

### PyQt5 GUI shows blank window or crashes

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

**Fix**:
```bash
# Install system Qt/xcb dependencies
sudo apt-get install -y libxcb-xinerama0 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xfixes0 libxcb-shape0

# If running over SSH, ensure X11 forwarding is enabled:
ssh -X user@host
```

### OpenCV / PyQt5 Qt conflict

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```
or
```
ImportError: ... libQt5 ... symbol not found
```

**Cause**: OpenCV and PyQt5 were installed from different sources (e.g., one from pip,
one from conda), so they bundle different Qt versions that conflict.

**Fix**: Reinstall both from conda-forge so they share the same Qt:
```bash
# Remove any pip-installed versions first
pip uninstall opencv-python opencv-python-headless opencv-contrib-python \
    opencv-contrib-python-headless PyQt5 PyQt5-sip PyQt5-Qt5 -y 2>/dev/null

# Install both from conda-forge (shared Qt)
conda install -c conda-forge py-opencv=4.5.5 pyqt=5.15.9 -y
```

### pip accidentally reinstalls OpenCV or PyQt5

If a pip package pulls in `opencv-python` or `PyQt5` as a dependency, it can
override the conda-forge versions and reintroduce the Qt conflict.

**Fix**: Check and remove the pip version, then verify conda version is active:
```bash
# Check which opencv is installed
python -c "import cv2; print(cv2.__file__)"
# Should point to conda env, NOT to site-packages/cv2 from pip

# If pip version overrode conda:
pip uninstall opencv-python opencv-contrib-python -y
conda install -c conda-forge py-opencv=4.5.5 --force-reinstall -y
```

### `import dvrk_data_processing` fails

```
ModuleNotFoundError: No module named 'dvrk_data_processing'
```

**Fix**: The local package must be installed in the active environment:
```bash
conda activate dvrk_multimodal_process
cd <project_root>
pip install -e .
```

### CUDA out of memory during depth estimation

**Fix**: Use image scaling or hierarchical inference in the Hydra config:
```yaml
preprocess:
  scale: 0.5                    # Reduce resolution
  hierarchical_inference: true   # Process in tiles
```

### FoundationStereo import error

```
ModuleNotFoundError: No module named 'core'
```

**Fix**: FoundationStereo must be added to `sys.path`. The `gen_depth_estimate.py` script
handles this automatically. If running standalone:
```bash
export PYTHONPATH="${PYTHONPATH}:<project_root>/FoundationStereo"
```

## Removing the Environment

```bash
conda deactivate
conda env remove -n dvrk_multimodal_process
```
