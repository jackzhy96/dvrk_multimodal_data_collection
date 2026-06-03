"""
Extract Online Raw Data Script for dVRK Multi-modal Data Collection

This script extracts the most raw (original) data from the convert_data folder
for online dataset publishing. It selectively copies only essential files:
  - Video files (*.avi) at the data folder level
  - Kinematic data (kin/ folder with *.bin files)
  - Select metadata files (timing and timetable files only)
  - Calibration data (camera_calibration/, hand_eye_calibration/)

Input folder structure:
  ./input_folder/
  ├── calibration/
  ├── <name_A>/
  │   ├── hand_eye_calibration/
  │   ├── <name_B>/
  │   │   ├── <idx_num>/       (data folder)
  │   │   │   ├── left.avi
  │   │   │   ├── right.avi
  │   │   │   ├── side1.avi    (optional)
  │   │   │   ├── kin/         (*.bin files)
  │   │   │   ├── meta/
  │   │   │   └── ...
  │   │   └── ...
  │   └── ...
  └── ...

Output folder structure:
  ./output_folder/
  ├── calibration/             (directly copied)
  ├── <name_A>/
  │   ├── hand_eye_calibration/ (directly copied)
  │   ├── <name_B>/
  │   │   ├── <idx_num>/
  │   │   │   ├── left.avi
  │   │   │   ├── right.avi
  │   │   │   ├── side1.avi    (if exists)
  │   │   │   ├── kin/         (all *.bin files)
  │   │   │   └── meta/        (only timing & timetable files)
  │   │   └── ...
  │   └── ...
  └── ...
"""

import os
import sys
import shutil
import fnmatch
import gc
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import hydra
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from tqdm import tqdm


@dataclass
class ExtractConfig:
    """
    Configuration for the online raw data extraction script.

    All configuration parameters are at the top level for simplicity.
    """
    workspace: str
    input_folder: str
    output_folder: str


# ---------------------------------------------------------------------------
# Meta folder whitelist: only these files are copied from each meta/ folder
# Includes timing JSONs and all timetable .bin files (side, side1, side2, etc.)
# ---------------------------------------------------------------------------
# Use a set for O(1) lookup — matters when meta/ folders have many files
META_WHITELIST_EXACT = {
    "end_times.json",
    "FPS.json",
    "start_times.json",
}

# Glob pattern to match any camera timetable binary:
#   left_timetable.bin, right_timetable.bin,
#   side_timetable.bin, side1_timetable.bin, side2_timetable.bin, etc.
META_WHITELIST_PATTERNS = [
    "*_timetable.bin",
]


def is_meta_whitelisted(filename: str) -> bool:
    """
    Check whether a file in the meta/ folder should be copied.

    We allow exact-name matches (end_times.json, FPS.json, start_times.json)
    and glob-pattern matches (*_timetable.bin) so that any number of side
    cameras are automatically included.

    Args:
        filename: The filename (not full path) to check.

    Returns:
        True if the file should be copied, False otherwise.
    """
    # Check exact matches first (fast path)
    if filename in META_WHITELIST_EXACT:
        return True

    # Check glob patterns (handles side, side1, side2, etc.)
    for pattern in META_WHITELIST_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True

    return False


def is_data_folder(folder: Path) -> bool:
    """
    Determine whether a folder is a data folder (leaf-level with recordings).

    A data folder is identified by having at least one .avi file AND a kin/
    subdirectory. This distinguishes it from intermediate grouping folders.

    Optimization: check kin/ existence first (cheap os.path.isdir) before
    scanning directory entries for .avi files. Uses os.scandir for speed —
    avoids constructing full Path objects for every entry.

    Args:
        folder: Path to check.

    Returns:
        True if the folder looks like a data recording folder.
    """
    # Cheap directory existence check first — avoids scanning large dirs
    if not (folder / 'kin').is_dir():
        return False

    # Use os.scandir for speed — stops as soon as the first .avi is found
    # Much faster than Path.iterdir() for folders with thousands of files
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False) and entry.name.endswith('.avi'):
                    return True
    except PermissionError:
        return False

    return False


def find_data_folders(input_root: Path) -> List[Path]:
    """
    Discover all data folders under the input root using BFS traversal.

    Skips known non-data directories (calibration, hand_eye_calibration,
    .git, __pycache__) during traversal. Uses queue-based BFS to safely
    handle deep nested structures without hitting recursion limits.

    Args:
        input_root: The top-level input folder.

    Returns:
        Sorted list of absolute Paths to data folders.
    """
    data_folders = []

    # Directories to skip entirely during BFS
    skip_dirs = {
        '.git', '__pycache__', 'calibration', 'camera_calibration',
    }

    # Use deque for O(1) popleft — list.pop(0) is O(n) which matters at scale
    queue = deque()
    for child in sorted(input_root.iterdir()):
        if child.is_dir() and child.name not in skip_dirs:
            queue.append(child)

    while queue:
        current = queue.popleft()

        # If this looks like a data folder, record it and stop descending
        if is_data_folder(current):
            data_folders.append(current)
            continue

        # Otherwise keep descending into subdirectories
        try:
            for child in sorted(current.iterdir()):
                if child.is_dir() and child.name not in skip_dirs:
                    queue.append(child)
        except PermissionError:
            print(f"  Warning: Permission denied, skipping: {current}")

    return sorted(data_folders)


def copy_data_folder(src: Path, dst: Path) -> Tuple[int, int, int]:
    """
    Selectively copy a single data folder to the output location.

    Copies:
      - All *.avi files at the top level
      - The entire kin/ directory (all *.bin files)
      - Whitelisted files from meta/ (timing JSONs + timetable bins)

    Args:
        src: Source data folder path.
        dst: Destination data folder path.

    Returns:
        Tuple of (files_copied, files_skipped, bytes_copied) counts.
    """
    files_copied = 0
    files_skipped = 0
    bytes_copied = 0

    dst.mkdir(parents=True, exist_ok=True)

    # --- 1) Copy video files (*.avi) at the top level ---
    for avi_file in sorted(src.glob('*.avi')):
        shutil.copy2(avi_file, dst / avi_file.name)
        bytes_copied += avi_file.stat().st_size
        files_copied += 1

    # --- 2) Copy the entire kin/ directory ---
    # Count source files before copying to avoid a second walk of the destination
    src_kin = src / 'kin'
    dst_kin = dst / 'kin'
    if src_kin.is_dir():
        # Pre-count source files and total size (single walk)
        kin_file_count = 0
        kin_bytes = 0
        for f in src_kin.rglob('*'):
            if f.is_file():
                kin_file_count += 1
                kin_bytes += f.stat().st_size

        if dst_kin.exists():
            shutil.rmtree(dst_kin)
        shutil.copytree(src_kin, dst_kin)
        files_copied += kin_file_count
        bytes_copied += kin_bytes

    # --- 3) Selectively copy from meta/ ---
    src_meta = src / 'meta'
    dst_meta = dst / 'meta'
    if src_meta.is_dir():
        dst_meta.mkdir(parents=True, exist_ok=True)
        for meta_file in sorted(src_meta.iterdir()):
            if meta_file.is_file():
                if is_meta_whitelisted(meta_file.name):
                    shutil.copy2(meta_file, dst_meta / meta_file.name)
                    bytes_copied += meta_file.stat().st_size
                    files_copied += 1
                else:
                    files_skipped += 1

    return files_copied, files_skipped, bytes_copied


def _format_bytes(nbytes: int) -> str:
    """
    Format a byte count into a human-readable string (KB, MB, GB, etc.).

    Args:
        nbytes: Number of bytes.

    Returns:
        Formatted string, e.g. "1.23 GB".
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def copy_calibration_folders(input_root: Path, output_root: Path) -> int:
    """
    Copy top-level and per-name_A calibration folders.

    Handles two types of calibration data:
      - Top-level: input_root/calibration/ -> output_root/calibration/
      - Per-session: input_root/<name_A>/hand_eye_calibration/
                     -> output_root/<name_A>/hand_eye_calibration/

    Args:
        input_root: The input folder root.
        output_root: The output folder root.

    Returns:
        Number of calibration folders copied.
    """
    copied = 0

    # --- Top-level calibration/ folder (camera calibration) ---
    # Also handle camera_calibration/ naming variant
    for calib_name in ['calibration', 'camera_calibration']:
        src_calib = input_root / calib_name
        if src_calib.is_dir():
            dst_calib = output_root / calib_name
            if dst_calib.exists():
                shutil.rmtree(dst_calib)
            shutil.copytree(src_calib, dst_calib)
            print(f"  Copied top-level {calib_name}/")
            copied += 1

    # --- Per-session hand_eye_calibration/ folders ---
    # These live at input_root/<name_A>/hand_eye_calibration/
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        # Skip known top-level non-session folders
        if child.name in {'calibration', 'camera_calibration', '.git', '__pycache__'}:
            continue

        src_he = child / 'hand_eye_calibration'
        if src_he.is_dir():
            dst_he = output_root / child.name / 'hand_eye_calibration'
            if dst_he.exists():
                shutil.rmtree(dst_he)
            dst_he.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_he, dst_he)
            print(f"  Copied {child.name}/hand_eye_calibration/")
            copied += 1

    return copied


def extract_raw_data(config: ExtractConfig) -> int:
    """
    Main extraction function — orchestrates the full copy pipeline.

    Steps:
      1. Copy calibration data (top-level and per-session)
      2. Discover all data folders via BFS
      3. For each data folder, selectively copy raw files to output

    Args:
        config: Extraction configuration.

    Returns:
        Number of data folders successfully processed.
    """
    input_path = Path(config.input_folder)
    output_path = Path(config.output_folder)

    # Validate input folder
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Copy calibration data ---
    print("\n--- Copying calibration data ---")
    calib_count = copy_calibration_folders(input_path, output_path)
    print(f"  Total calibration folders copied: {calib_count}")

    # --- Step 2: Discover data folders ---
    print("\n--- Discovering data folders ---")
    data_folders = find_data_folders(input_path)

    if not data_folders:
        print("  No data folders found in input directory.")
        return 0

    print(f"  Found {len(data_folders)} data folder(s):")
    for df in data_folders:
        # Show relative path from input root for clarity
        rel = df.relative_to(input_path)
        print(f"    {rel}")

    # --- Step 3: Copy each data folder selectively ---
    print("\n--- Extracting raw data ---")
    total_copied = 0
    total_skipped = 0
    total_bytes = 0
    processed = 0
    failed_folders = []

    for src_folder in tqdm(data_folders, desc="Extracting data folders"):
        # Preserve the relative path structure from input to output
        rel_path = src_folder.relative_to(input_path)
        dst_folder = output_path / rel_path

        try:
            copied, skipped, nbytes = copy_data_folder(src_folder, dst_folder)
            total_copied += copied
            total_skipped += skipped
            total_bytes += nbytes
            processed += 1
        except Exception as e:
            # Log error but keep processing remaining folders
            print(f"\n  Error processing {rel_path}: {e}")
            failed_folders.append(str(rel_path))

        # Periodically release memory for large batches
        if processed % 50 == 0:
            gc.collect()

    # Print summary statistics
    print(f"\n--- Summary ---")
    print(f"  Files copied:  {total_copied}")
    print(f"  Files skipped: {total_skipped}")
    print(f"  Total size:    {_format_bytes(total_bytes)}")
    if failed_folders:
        print(f"  Failed folders ({len(failed_folders)}):")
        for ff in failed_folders:
            print(f"    {ff}")

    return processed


# ---------------------------------------------------------------------------
# Hydra configuration setup
# ---------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(name="extract_online_raw_data", node=ExtractConfig)

# Resolve config path relative to this script's location
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_extract_online_raw_data"
)
def main(cfg: DictConfig):
    """
    Main entry point — prints config summary and runs extraction.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf).
    """
    print("=" * 70)
    print("dVRK Multi-modal Data — Extract Online Raw Data")
    print("=" * 70)
    print(f"Workspace:     {cfg.workspace}")
    print(f"Input folder:  {cfg.input_folder}")
    print(f"Output folder: {cfg.output_folder}")
    print("=" * 70)

    extract_config = ExtractConfig(
        workspace=cfg.workspace,
        input_folder=cfg.input_folder,
        output_folder=cfg.output_folder,
    )

    try:
        processed = extract_raw_data(extract_config)

        print(f"\n{'=' * 70}")
        print("Extraction complete!")
        print(f"  Data folders processed: {processed}")
        print(f"  Input:  {cfg.input_folder}")
        print(f"  Output: {cfg.output_folder}")
        print("=" * 70)

        if processed > 0:
            print(f"\nSuccessfully extracted {processed} dataset(s)")
        else:
            print("\nNo datasets were extracted")

    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
