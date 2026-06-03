# Environment Setup: `dvrk_multimodal_process`

Unified conda environment for all dVRK multi-modal data processing pipelines.
This document describes what the environment actually contains; the
`create_env.sh` script automates the steps below.

## What this environment supports

| Pipeline                | Scripts                                                              | Key dependencies                                  |
|-------------------------|----------------------------------------------------------------------|--------------------------------------------------|
| Raw image processing    | `gen_rectify_resize.py` (legacy: `gen_resize_rectify.py`)            | OpenCV, NumPy                                    |
| Kinematic mapping       | `gen_kinematic_heatmap_handeye.py`, `gen_kinematic_heatmap_dVRK.py`  | OpenCV, SciPy, Hydra                             |
| Depth estimation        | `gen_depth_estimate.py`                                              | FoundationStereo, PyTorch+CUDA, flash-attn (opt) |
| Optical flow (Farneback)| `gen_optical_flow.py`                                                | OpenCV                                           |
| Optical flow (RAFT)     | `gen_optical_flow_raft.py`                                           | TorchVision (RAFT models), PyTorch+CUDA          |
| Data annotation GUIs    | `data_annotate.py`, `meta_data_annotate.py`                          | PyQt5 (cv2.imshow is NOT used — see notes below) |
| Data export             | `convert_open_h.py`                                                  | Pandas, PyArrow                                  |
| Image → video           | `convert_image_to_video.py`                                          | FFmpeg (subprocess), OpenCV (fallback)           |
| Video → image           | `convert_video_to_image.py`                                          | FFmpeg (subprocess), OpenCV (fallback)           |

## Prerequisites

- **OS**: Linux. Tested on Ubuntu 20.04 (GLIBC 2.31); should also work on
  22.04 and 24.04 but the flash-attn / xformers / torch wheels were pinned
  against 20.04 binaries, so newer Ubuntu users may have to bump versions
  if the wheels don't load.
- **Disk space**: ~10 GB for the env + ~10 GB for the FoundationStereo
  pretrained weights.
- **NVIDIA GPU**: required for depth estimation (FoundationStereo) and
  RAFT optical flow. The other pipelines (raw image processing, kinematic
  mapping, annotation GUIs, data conversion) run CPU-only. The script
  detects GPU presence and warns rather than aborting if missing.

### One-time setup on a fresh Ubuntu 20.04 machine

If you're starting from a clean Ubuntu install, run these once before
the env-setup script:

```bash
# (1) System packages: build tools, git, X11/Qt runtime libs.
#     build-essential covers gcc/g++/make for any compile-from-source pip
#     package (flash-attn is the big one). The libxcb-* set is what PyQt5
#     loads at runtime when you open the annotation GUI windows; without
#     them you get "Could not load the Qt platform plugin 'xcb'".
sudo apt update
sudo apt install -y \
    build-essential git wget curl ca-certificates \
    libxcb-xinerama0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 \
    libxcb-shape0 libgl1 libglib2.0-0

# (2) NVIDIA driver + CUDA toolkit (skip if no GPU).
#     The driver is what `nvidia-smi` queries; the toolkit (with nvcc) is
#     only needed if you want flash-attn to compile.
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
# Reboot so the driver loads:
sudo reboot

# (3) Miniconda. The env-setup script needs `conda` on PATH; if it's not,
#     the script's pre-flight will print the install hints below.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
# Open a new terminal so `conda init` takes effect, then continue.
```

### Repository checkout

The toolkit uses FoundationStereo as a git submodule; clone with
`--recursive` so it comes along:

```bash
git clone --recursive https://github.com/<owner>/dvrk_multimodal_data_collection.git
cd dvrk_multimodal_data_collection
# If you already cloned without --recursive, run:
git submodule update --init --recursive
```

### Sanity checks before running the script

```bash
conda --version                 # conda is on PATH
nvidia-smi                      # GPU driver visible (skip if no GPU)
nvcc --version                  # CUDA toolkit (skip if no flash-attn)
ls FoundationStereo/readme.md   # submodule initialised
```

The env-setup script's `[-1/11]` pre-flight repeats these checks and
fails loudly with fix-suggestion messages if anything is missing — so
you can skip the manual sanity-checking if you'd rather just run the
script and see what it complains about.

## Quick install (automated)

```bash
cd <project_root>
bash env_setup/create_env.sh
```

Runtime: ~10-15 min, dominated by flash-attn's CUDA-kernel compile. The script
will:

1. Make sure libmamba is the active conda solver (installs it into base
   if needed — see *Why libmamba?* below).
2. Create the env `dvrk_multimodal_process` at Python 3.9.
3. Install FFmpeg from conda-forge (the only conda-install step).
4. Install PyTorch 2.4.1 (CUDA 12.1), xformers, flash-attn from pip.
5. Pin `numpy==1.23.5` BEFORE OpenCV (ABI requirement; see step 5 below).
6. Install OpenCV (**headless**) + PyQt5 + everything else from pip.
7. `pip install -e .` for the local package.
8. Verify imports + GPU + FFmpeg.

## Why libmamba?

Conda's default solver pre-23.10 — the **classic** Python solver — does
SAT-style dependency resolution by backtracking in pure Python. For large
multi-channel dependency graphs (e.g. installing FFmpeg from conda-forge,
which transitively brings in 30+ codec libraries), classic can take **10-30
minutes** and frequently stalls on hard-to-solve cases with the unhelpful
`"examining package conflicts"` progress bar.

**libmamba** is a C++ reimplementation of the solver built on libsolv (the
same SAT engine Fedora and openSUSE use for RPM dependency resolution). It is:

- **~50-100× faster** than classic on typical conda-forge solves (seconds
  instead of minutes).
- **More reliable** on conflict-heavy graphs — won't hang indefinitely.
- The **official default** in conda 23.10+, but legacy installations may
  still be on classic.

The `create_env.sh` script does a single conda-install step (FFmpeg from
conda-forge). On classic, that's a 10+ minute wait at step 2; on libmamba,
it's a few seconds. The script's preflight block (step 0) ensures libmamba
is installed and active before anything else runs:

```bash
# Equivalent to what the script does at step 0:
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba
```

This is a base-env-level setting that persists across all subsequent conda
invocations, so you only pay this cost once.

## Manual install (step-by-step)

Use these if the automated script fails or you need more control.

### Step 0: Enable libmamba (one-time, persistent)

```bash
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba
```

### Step 1: Create the env

```bash
conda create -n dvrk_multimodal_process python=3.9 -y
conda activate dvrk_multimodal_process
```

### Step 2: Install FFmpeg from conda-forge

FFmpeg is the **one** conda-install in the whole workflow. The conversion
scripts call `ffmpeg` and `ffprobe` via subprocess; installing into the conda
env keeps the binaries self-contained and avoids depending on whatever the
host system has.

```bash
conda install -c conda-forge ffmpeg -y
```

Verify:
```bash
which ffmpeg && ffmpeg -version | head -1
```

### Step 3: PyTorch 2.4.1 + CUDA 12.1

```bash
pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

CUDA 12.1 binaries are forward-compatible with CUDA 12.4 drivers; pinning
torch 2.4.1 matches the FoundationStereo + xformers + flash-attn versions
tested below. Bumping torch breaks at least one of those.

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.1+cu121 True
```

### Step 4: xformers

```bash
pip install xformers==0.0.28.post1
```

Must match the torch + CUDA version exactly. Don't bump.

### Step 5: Pin `numpy==1.23.5` and use a pip constraints file from here on

`opencv-contrib-python-headless==4.5.3.56` was compiled against the
numpy-1.x ABI. Loading it with numpy 2.x raises:

```
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
ImportError: numpy.core.multiarray failed to import
```

Installing `numpy==1.23.5` once is **not enough** by itself, because pip's
resolver re-evaluates the dependency graph on every subsequent `pip install`
call. A later install (e.g. `scikit-image`, `pandas`, `scipy`) that depends on
`numpy` can silently UPGRADE numpy to 2.x even though you pinned it earlier.

The fix: a pip **constraints file** that pip respects across every install.
Save the following as `/tmp/constraints.txt`:

```
# Hard-pinned for OpenCV 4.5.3 ABI compatibility. Bumping past 1.x breaks cv2.
numpy==1.23.5
```

Then install numpy with the constraint:

```bash
pip install -c /tmp/constraints.txt "numpy==1.23.5"
```

And **pass `-c /tmp/constraints.txt` to every pip install command in steps
6–11 below.** Pip will then backtrack through alternative versions of the
*other* packages (e.g. scipy 1.13 instead of 1.15) to find a combination
that keeps numpy at 1.23.5, instead of bumping numpy.

If you forget the constraint on a later step and import cv2 fails with the
ABI message, fix with:

```bash
pip install -c /tmp/constraints.txt "numpy==1.23.5" --force-reinstall
```

### Step 6: Install OpenCV (**headless** variant)

```bash
pip install "opencv-contrib-python-headless==4.5.3.56"
```

> ⚠ **Headless** means cv2 has **no bundled Qt** and **no GUI backend**.
> `cv2.imshow()` will silently fail or raise — **don't use it in any script.**
> All UI in this project is PyQt5 (the annotation GUIs are PyQt5 windows;
> none call cv2.imshow).
>
> This is a deliberate choice — see *Why headless OpenCV + bundled-Qt
> PyQt5?* below.

The `-contrib-` package includes the OpenCV contrib modules (SIFT, SURF,
tracking, ximgproc, etc.) that several preprocessing scripts depend on.

### Step 7: PyQt5 (bundled Qt 5.15.18)

```bash
pip install "PyQt5==5.15.11" "PyQt5-sip==12.17.1"
```

Pip's PyQt5 wheel automatically pulls in `PyQt5-Qt5==5.15.18`, which is the
binary Qt distribution. PyQt5 uses its own bundled Qt; nothing else in the
env shares it.

### Step 8: Core processing dependencies

```bash
pip install \
    "hydra-core>=1.3.2" \
    omegaconf \
    scipy \
    pyyaml \
    tqdm \
    "pandas>=1.3.0" \
    pyarrow
```

### Step 9: Computer vision / 3D / ML packages

> **IMPORTANT — `imgaug` and `albumentations` will pull in additional
> OpenCV variants behind your back.** As of 2026:
> - `imgaug==0.4.0` declares `opencv-python` (non-headless) as a hard dep,
> - `albumentations` declares `opencv-python-headless` (headless, different name),
> - `scikit-image` may pull `opencv-python-headless` indirectly.
>
> All four opencv distributions (`opencv-python`, `opencv-contrib-python`,
> `opencv-python-headless`, `opencv-contrib-python-headless`) install into
> the SAME `site-packages/cv2/` directory. When multiple variants coexist,
> cv2 imports break with cryptic errors:
> ```
> AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType'
>     (most likely due to a circular import)
> ```
>
> A pip constraints file can pin *versions* but not *exclude packages by name*,
> so the only reliable fix is to **normalize after the install**: uninstall
> every opencv variant pip dragged in, then reinstall only the one we want.
> See step 9b below.

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

### Step 9b: Normalize OpenCV (single variant)

After step 9 finishes, check what opencv variants pip ended up with:

```bash
pip list | grep -i opencv
```

If you see anything OTHER than just `opencv-contrib-python-headless 4.5.3.56` —
e.g. you also see `opencv-python` or `opencv-python-headless` — uninstall
every variant and reinstall only the one we want:

```bash
pip uninstall -y \
    opencv-python \
    opencv-contrib-python \
    opencv-python-headless \
    opencv-contrib-python-headless

pip install --no-deps --force-reinstall \
    "opencv-contrib-python-headless==4.5.3.56"
```

`--no-deps` is important — without it, pip's resolver would re-run the
dependency graph and possibly pull a different opencv variant back in.
`--force-reinstall` clears any half-deleted cv2 files from the
uninstall step.

Verify:

```bash
pip list | grep -i opencv
# Expected: ONLY `opencv-contrib-python-headless 4.5.3.56`
python -c "import cv2; print(cv2.__version__)"
# Expected: 4.5.3
```

### Step 10: flash-attn (optional, compile-from-source)

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Pinned 2.7.4.post1 for GLIBC 2.31 (Ubuntu 20.04) + torch 2.4.1 + CUDA 12.1.
Compile takes ~5 min. If it fails, depth estimation still works — just
slower (FoundationStereo falls back to standard attention).

### Step 11: Editable-install the local package

```bash
cd <project_root>
pip install -e .
```

### Step 12: Verify

```bash
python << 'PY'
import sys, shutil
import torch, cv2, hydra, scipy, pandas, pyarrow, tqdm
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
import torchvision
import dvrk_data_processing

print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}  CUDA: {torch.cuda.is_available()}')
print(f'OpenCV      : {cv2.__version__}  (headless — cv2.imshow does NOT work)')
print(f'PyQt5       : {PYQT_VERSION_STR}  (Qt {QT_VERSION_STR})')
print(f'TorchVision : {torchvision.__version__}')
print(f'FFmpeg      : {shutil.which("ffmpeg") or "NOT FOUND"}')
PY
```

## Verifying FoundationStereo end-to-end

The FoundationStereo Python imports work as soon as the submodule is checked
out, but actually running depth estimation requires the pretrained weights:

1. Download from
   <https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf>
2. Place under `FoundationStereo/pretrained_models/`:
   ```
   FoundationStereo/pretrained_models/
   ├── 23-51-11/    # ViT-large (best quality)
   │   ├── model.pth
   │   └── cfg.yaml
   └── 11-33-40/    # ViT-small (faster inference)
       ├── model.pth
       └── cfg.yaml
   ```
3. Smoke-test the imports:
   ```bash
   conda activate dvrk_multimodal_process
   python -c "
   import sys; sys.path.append('FoundationStereo')
   from core.foundation_stereo import FoundationStereo
   from core.utils.utils import InputPadder
   from Utils import vis_disparity
   print('FoundationStereo imports OK')
   "
   ```

## Running processing scripts

After activation, all scripts are ready to use:

```bash
conda activate dvrk_multimodal_process

# 1. Rectify and resize (must run first — produces intermediate_dir)
cd src/dvrk_data_processing/raw_image_processing && python gen_rectify_resize.py

# 2. Kinematic mapping (hand-eye is the default)
cd ../kinematic_mapping && python gen_kinematic_heatmap_handeye.py

# 3. Depth estimation (requires GPU + FoundationStereo weights)
cd ../depth_estimation && python gen_depth_estimate.py

# 4. Optical flow
cd ../optical_flow
python gen_optical_flow_raft.py     # RAFT (GPU, recommended)
python gen_optical_flow.py          # Farneback (CPU fallback)

# 5. Annotation GUIs (PyQt5 windows; X11 forwarding works over SSH)
cd ../data_annotation
python data_annotate.py
python meta_data_annotate.py

# 6. Image ↔ video conversion (uses FFmpeg via subprocess, OpenCV fallback)
cd ../../scripts
python convert_image_to_video.py
python convert_video_to_image.py
```

Before running, edit the `@hydra.main(config_name=...)` decorator at the bottom
of each script to point at your personal config under `config/`.

## Package version summary

| Package                              | Version           | Source        | Notes                                                              |
|--------------------------------------|-------------------|---------------|--------------------------------------------------------------------|
| Python                               | 3.9               | conda         | FoundationStereo requirement                                       |
| ffmpeg                               | 8.x               | conda-forge   | Binary called via subprocess (image↔video conversion)              |
| numpy                                | **1.23.5** (pin)  | pip           | Pinned to numpy-1.x ABI for OpenCV 4.5 compat                      |
| opencv-contrib-python-headless       | **4.5.3.56** (pin)| pip           | Headless — no Qt, no cv2.imshow                                    |
| PyQt5                                | 5.15.11           | pip           | Annotation GUI framework                                           |
| PyQt5-Qt5                            | 5.15.18           | pip (transitive) | Bundled Qt binary distribution                                  |
| PyQt5-sip                            | 12.17.1           | pip           |                                                                    |
| torch                                | **2.4.1+cu121**   | pip (PyTorch) | Pinned for FoundationStereo + xformers + flash-attn compatibility  |
| torchvision                          | 0.19.1+cu121      | pip (PyTorch) | RAFT models                                                        |
| torchaudio                           | 2.4.1+cu121       | pip (PyTorch) |                                                                    |
| xformers                             | **0.0.28.post1**  | pip           | Must match torch exactly                                           |
| flash-attn                           | **2.7.4.post1**   | pip (compile) | Optional; GLIBC 2.31 + torch 2.4.1 + CUDA 12.1 only                |
| hydra-core                           | >=1.3.2           | pip           | Configuration management                                           |
| scipy / pandas / pyarrow             | latest            | pip           |                                                                    |
| scikit-image / timm / albumentations | latest            | pip           | CV / model utilities                                               |
| open3d / trimesh                     | latest            | pip           | 3D point cloud (depth viz)                                         |

## Why headless OpenCV + bundled-Qt PyQt5?

A previous version of this env tried to install both OpenCV and PyQt5 from
conda-forge so they would share a single `qt-main` package. That works in
principle (it's the "canonical" anti-conflict story), but in practice
conda-forge's solver was fragile here — `py-opencv` releases drift relative
to `qt-main` and `pyqt`, and bumping any one of them could break the
others. Whoever set up this env historically pip-installed both packages
out from under the conda-forge versions to get the build unstuck.

The result, by accident, is the **most robust** configuration of the three:

| Configuration                                          | OpenCV's Qt    | PyQt5's Qt     | Conflict? |
|--------------------------------------------------------|---------------|---------------|-----------|
| pip `opencv-python` (non-headless) + pip `PyQt5`       | bundled 5.15  | bundled 5.15  | ⚠ two different Qt's in one process — crashes  |
| conda-forge `py-opencv` + conda-forge `pyqt`           | shared `qt-main` | shared `qt-main` | ✅ works when versions align; fragile when conda-forge bumps |
| **pip `opencv-contrib-python-headless` + pip `PyQt5`** | **none** (headless) | bundled 5.15  | ✅ no conflict ever, because OpenCV has nothing to conflict with |

The cost is that `cv2.imshow()` doesn't work. We absorb that — every UI in
this codebase is PyQt5. If you ever need cv2's imshow for ad-hoc debugging,
spin up a separate env (`opencv-python` non-headless on its own) rather
than re-mixing Qt's in this one.

## Troubleshooting

### Conda install hangs at "examining package conflicts"

You're on the legacy classic solver — install libmamba and retry:

```bash
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba
# Then re-run the conda install that hung.
```

### `flash-attn` fails to build

```
ERROR: Failed building wheel for flash-attn
```

**Cause**: missing or mis-located CUDA toolkit, or compiler mismatch.

```bash
# Verify CUDA toolkit is available
nvcc --version              # should show CUDA 12.x

# If nvcc is not found, add to PATH:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Retry:
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

If it still fails, **skip it**. Depth estimation works without flash-attn,
just slower.

### PyQt5 GUI: "Could not load the Qt platform plugin 'xcb'"

Almost always means an extra OpenCV got installed that bundles its own Qt.
Check and remove:

```bash
pip list | grep -i opencv
# Should ONLY show: opencv-contrib-python-headless 4.5.3.56
# If you also see opencv-python or opencv-contrib-python (no -headless):
pip uninstall opencv-python opencv-contrib-python -y
pip install "opencv-contrib-python-headless==4.5.3.56" --force-reinstall
```

Other causes:
- Missing system xcb libraries on Ubuntu 20.04:
  ```bash
  sudo apt-get install -y libxcb-xinerama0 libxkbcommon-x11-0 libxcb-icccm4 \
      libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
      libxcb-xfixes0 libxcb-shape0
  ```
- Over SSH without X11 forwarding — connect with `ssh -X user@host`.

### `import cv2` fails after a `pip install` of some other package

A pip dependency pulled in non-headless OpenCV behind your back. Same fix
as the PyQt5 crash above.

### cv2 `_registerMatType` circular import error

```
AttributeError: partially initialized module 'cv2' has no attribute
    '_registerMatType' (most likely due to a circular import)
```

Caused by multiple opencv distributions installed simultaneously — they
overwrite each other's files in `site-packages/cv2/`. Common offenders are
`imgaug` (declares `opencv-python` as a dep) and `albumentations` (declares
`opencv-python-headless`). Check and normalize:

```bash
pip list | grep -i opencv
# If more than one entry, run:
pip uninstall -y \
    opencv-python \
    opencv-contrib-python \
    opencv-python-headless \
    opencv-contrib-python-headless
pip install --no-deps --force-reinstall \
    "opencv-contrib-python-headless==4.5.3.56"
```

### `import dvrk_data_processing` fails

The local editable install didn't take effect — usually because you ran
`pip install -e .` in the wrong env or wrong directory.

```bash
conda activate dvrk_multimodal_process
cd <project_root>     # the directory containing pyproject.toml
pip install -e .
```

### CUDA out of memory during depth estimation

Lower the input resolution or enable hierarchical inference in the Hydra
config:

```yaml
preprocess:
  scale: 0.5                  # halve input resolution
  hierarchical_inference: true # process in tiles
```

### FoundationStereo: `ModuleNotFoundError: No module named 'core'`

The submodule wasn't initialised or `sys.path` wasn't updated.

```bash
git submodule update --init --recursive
# Or, when calling FoundationStereo from a standalone script:
export PYTHONPATH="${PYTHONPATH}:<project_root>/FoundationStereo"
```

`gen_depth_estimate.py` handles this automatically — the error only shows
up if you're calling FoundationStereo internals directly.

## Removing the environment

```bash
conda deactivate
conda env remove -n dvrk_multimodal_process
```
