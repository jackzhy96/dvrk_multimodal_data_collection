#!/usr/bin/env python3
"""
Data Trimming Script for dVRK Multi-modal Data Collection

This script ensures all subfolders within dataset subsets have the same number of files.
It finds the minimum file count across specified folders and trims excess files from other
folders to match this minimum.

Purpose:
- Synchronize file counts across image/, kinematic/, time_syn/, and annotation/ folders
- Handle cases where different cameras or sensors captured different numbers of frames
- Flexible removal strategies: from end, from front, or skip first n files

Workflow:
1. Find all subset folders in the data directory
2. For each subset, identify which image subfolders to check (stereo or all mode)
3. Find the minimum file count across all relevant folders
4. Remove excess files based on remove_from_end strategy:
   - remove_from_end=-1: Remove from end (keep 0 to n-1)
   - remove_from_end=0: Remove from front (keep latest files)
   - remove_from_end=n>0: Remove first n files, then keep next n files
5. Verify all folders have the target count after trimming

Input structure:
  <data_folder>/
    raw/                    # Or any structure containing subsets
      0/                    # Subset folder
        image/
          left/             # PNG images: 0.png, 1.png, ..., n.png
          right/
          side1/            # Optional
        kinematic/
          ECM/              # JSON files: 0.json, 1.json, ..., n.json
          PSM1/
          PSM2/
        time_syn/           # Timestamp files
        annotation/         # Optional
          contact_detection/

Key Features:
- Two modes: "stereo" (left/right only) or "all" (all image subfolders)
- Dry run mode for safe testing
- Detailed reporting of file counts before and after
- Preserves temporal consistency by removing from end
- User confirmation before deletion (optional)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm


@dataclass
class DataTrimConfig:
    """
    Configuration for data trimming script.

    All configuration parameters are at the top level for simplicity.
    """
    workspace: str
    data_folder: str
    trim_mode: str  # "stereo" or "all"
    remove_from_end: int  # -1: from end, 0: from front, n>0: first n files
    remove_files: bool  # True: actually remove, False: dry run
    verbose: bool
    require_confirmation: bool
    target_folders: List[str]


def is_subset_folder(folder_path: Path, required_folders: List[str]) -> bool:
    """
    Check if a folder is a valid subset folder by verifying it contains required subfolders.

    Args:
        folder_path: Path to check
        required_folders: List of required subfolder names (e.g., ['image', 'kinematic', 'time_syn'])

    Returns:
        True if folder contains at least image, kinematic, and time_syn subfolders

    Note:
        A subset folder must have at minimum:
        - image/ (with camera subfolders)
        - kinematic/ (with robot arm subfolders)
        - time_syn/ (timestamp data)
        The annotation/ folder is optional.
    """
    # Must have at least image, kinematic, and time_syn
    core_folders = ['image', 'kinematic', 'time_syn']

    for core_folder in core_folders:
        folder_to_check = folder_path / core_folder
        if not folder_to_check.exists() or not folder_to_check.is_dir():
            return False

    return True


def find_subset_folders(root_path: Path, verbose: bool = True, max_depth: int = 10) -> List[Path]:
    """
    Efficiently find all valid subset folders in the data directory using iterative search.

    Args:
        root_path: Root directory to search from
        verbose: Print detailed information
        max_depth: Maximum depth to search (prevents infinite recursion)

    Returns:
        List of paths to subset folders

    Note:
        A subset folder must contain image/, kinematic/, and time_syn/ subfolders.
        Uses iterative breadth-first search which is much faster than rglob on large datasets.
        Stops searching within a folder once it's identified as a subset (doesn't recurse into data folders).
    """
    subset_folders = []

    if verbose:
        print(f"\nSearching for subset folders in {root_path}...")

    # Common non-data folders to skip
    skip_set = {
        'camera_calibration', 'hand_eye_calibration',
        '.git', '__pycache__', 'pretrained_models',
        '.venv', 'venv', 'env', '.idea',
        # Skip actual data subfolders to avoid deep recursion
        'image', 'kinematic', 'time_syn', 'annotation'
    }

    # Use iterative breadth-first search instead of rglob
    # This is MUCH faster and allows us to control depth
    to_search = [(root_path, 0)]  # (path, depth)

    while to_search:
        current_path, depth = to_search.pop(0)

        # Stop if we've gone too deep
        if depth > max_depth:
            continue

        # Check if this is a subset folder BEFORE recursing
        if is_subset_folder(current_path, ['image', 'kinematic', 'time_syn']):
            subset_folders.append(current_path)
            if verbose:
                try:
                    rel_path = current_path.relative_to(root_path)
                    print(f"  Found subset: {rel_path}")
                except ValueError:
                    print(f"  Found subset: {current_path}")
            # Don't recurse into subset folders (they contain thousands of files)
            continue

        # List immediate subdirectories only (not all files)
        try:
            subdirs = [d for d in current_path.iterdir() if d.is_dir()]
        except (PermissionError, OSError) as e:
            if verbose:
                print(f"  Warning: Cannot access {current_path}: {e}")
            continue

        # Add subdirectories to search queue (excluding skip list)
        for subdir in subdirs:
            # Skip directories in skip list or starting with '.'
            if subdir.name in skip_set or subdir.name.startswith('.'):
                continue

            to_search.append((subdir, depth + 1))

    if verbose:
        print(f"  Total subsets found: {len(subset_folders)}")

    return subset_folders


def count_files_in_folder(folder_path: Path, pattern: str = '*') -> int:
    """
    Count number of files matching pattern in a folder.

    Args:
        folder_path: Path to folder
        pattern: File pattern to match (default: all files)

    Returns:
        Number of files matching pattern

    Note:
        Only counts files, not directories.
    """
    if not folder_path.exists():
        return 0

    try:
        return len([f for f in folder_path.glob(pattern) if f.is_file()])
    except Exception as e:
        print(f"  Warning: Could not count files in {folder_path}: {e}")
        return 0


def get_all_subfolders(parent_folder: Path) -> List[Path]:
    """
    Get all immediate subfolders of a parent folder.

    Args:
        parent_folder: Parent folder path

    Returns:
        List of subfolder paths

    Note:
        Only returns immediate children, not recursive.
    """
    if not parent_folder.exists():
        return []

    return [f for f in parent_folder.iterdir() if f.is_dir()]


def find_minimum_file_count(subset_path: Path, target_folders: List[str], trim_mode: str,
                           verbose: bool = True) -> Tuple[int, Dict[str, int]]:
    """
    Find the minimum file count across target folders in a subset.

    Args:
        subset_path: Path to subset folder
        target_folders: List of folder names to check (from config)
        trim_mode: "stereo" (left/right only) or "all" (all image subfolders)
        verbose: Print detailed information

    Returns:
        Tuple of (minimum_count, file_counts_dict)
        where file_counts_dict maps folder_path_str -> count

    Note:
        Only checks folders listed in target_folders.
        For "image" folder: trim_mode determines which subfolders to check.
        For other folders: checks all subfolders or files directly (time_syn).
        Empty folders (0 files in all subfolders) are excluded from minimum calculation.
    """
    file_counts = {}

    if verbose:
        print(f"  Counting files in subset {subset_path.name}...", end='', flush=True)

    # Process each target folder
    for folder_name in target_folders:
        folder_path = subset_path / folder_name

        # Skip if folder doesn't exist
        if not folder_path.exists():
            if verbose:
                print(f"\n    Note: {folder_name}/ doesn't exist, skipping", end='', flush=True)
            continue

        # Handle each folder type
        if folder_name == "image":
            # Check image subfolders based on trim_mode
            image_subfolders = get_all_subfolders(folder_path)

            if trim_mode == "stereo":
                # Only check left and right
                for camera_name in ['left', 'right']:
                    camera_folder = folder_path / camera_name
                    if camera_folder.exists():
                        count = count_files_in_folder(camera_folder)
                        file_counts[str(camera_folder.relative_to(subset_path))] = count
            else:  # "all" mode
                # Check all image subfolders
                for subfolder in image_subfolders:
                    count = count_files_in_folder(subfolder)
                    file_counts[str(subfolder.relative_to(subset_path))] = count

        elif folder_name == "time_syn":
            # time_syn has files directly (not in subfolders)
            count = count_files_in_folder(folder_path)
            file_counts[str(folder_path.relative_to(subset_path))] = count

        else:
            # For kinematic, annotation, and any other folders: check all subfolders
            subfolders = get_all_subfolders(folder_path)

            if not subfolders:
                # Folder exists but has no subfolders - check files directly
                count = count_files_in_folder(folder_path)
                if count > 0:  # Only add if there are files
                    file_counts[str(folder_path.relative_to(subset_path))] = count
            else:
                # Check each subfolder
                for subfolder in subfolders:
                    count = count_files_in_folder(subfolder)
                    file_counts[str(subfolder.relative_to(subset_path))] = count

    # Find minimum count across target folders
    # Only folders specified in target_folders are checked
    # Empty subfolders (0 files) within target folders are excluded from minimum calculation
    # This handles cases where optional data (e.g., annotation) doesn't exist yet
    if not file_counts:
        if verbose:
            print(" No target folders found or all are empty!")
        return 0, file_counts

    # Filter out empty folders (0 files)
    # If a target folder has no data, we don't want it to force everything to 0
    non_empty_counts = {folder: count for folder, count in file_counts.items() if count > 0}

    if not non_empty_counts:
        if verbose:
            print(" All target folders are empty!")
        return 0, file_counts

    min_count = min(non_empty_counts.values())

    if verbose:
        print(f" Done!")
        print(f"\n  File counts:")
        for folder_name, count in sorted(file_counts.items()):
            if count == 0:
                marker = " <- EMPTY (not included in minimum)"
            elif count == min_count:
                marker = " <- MIN"
            else:
                marker = ""
            print(f"    {folder_name}: {count}{marker}")
        print(f"  Target file count: {min_count}")

    return min_count, file_counts


def get_files_to_remove(folder_path: Path, target_count: int, remove_from_end: int) -> Tuple[List[Path], Optional[str]]:
    """
    Get list of files that need to be removed to reach target count.

    Args:
        folder_path: Folder containing files to trim
        target_count: Target number of files to keep
        remove_from_end: -1 for end, 0 for front, n>0 for first n files

    Returns:
        Tuple of (list of file paths to remove, error message if any)

    Note:
        Assumes files are named with numeric indices (0.png, 1.png, 2.png, etc.)
        or (0.json, 1.json, 2.json, etc.).

        Examples with current_count=200, target_count=100 (need to remove 100 files):
        - remove_from_end=-1: Keep 0-99, remove 100-199
        - remove_from_end=0: Keep 100-199, remove 0-99 (not recommended)
        - remove_from_end=50: Remove 0-49, keep 50-149 (remove first 50, then keep 100)
        - remove_from_end=150: ERROR (trying to remove 150 but only need to remove 100)
    """
    if not folder_path.exists():
        return [], None

    # Get all files with numeric indices, sorted by index
    indexed_files = []
    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue

        try:
            # Extract numeric index from filename
            idx = int(file_path.stem)  # stem removes extension
            indexed_files.append((idx, file_path))
        except ValueError:
            # If filename is not numeric, skip it (might be metadata files)
            continue

    # Sort by index
    indexed_files.sort(key=lambda x: x[0])

    current_count = len(indexed_files)

    if current_count <= target_count:
        return [], None  # No trimming needed

    num_to_remove = current_count - target_count

    if remove_from_end == -1:
        # Remove from end: Keep indices 0 to target_count-1, remove indices >= target_count
        files_to_remove = [f for idx, f in indexed_files if idx >= target_count]
        return files_to_remove, None

    elif remove_from_end == 0:
        # Remove from front: Keep latest files, remove oldest
        # Keep files from index (current_count - target_count) onwards
        files_to_remove = [f for idx, f in indexed_files if idx < num_to_remove]
        return files_to_remove, None

    elif remove_from_end > 0:
        # Remove first n files, then keep next target_count files
        n = remove_from_end

        # Check if n is larger than required removal amount
        if n > num_to_remove:
            error_msg = (f"remove_from_end={n} is larger than required removal amount ({num_to_remove}). "
                        f"Current count: {current_count}, target count: {target_count}. "
                        f"Set remove_from_end to a value <= {num_to_remove}")
            return [], error_msg

        # Remove files with indices 0 to n-1
        # Then keep files from index n to n+target_count-1
        # Then remove remaining files from index n+target_count onwards
        files_to_remove = []

        for idx, file_path in indexed_files:
            # Remove first n files (indices 0 to n-1)
            if idx < n:
                files_to_remove.append(file_path)
            # Keep files from n to n+target_count-1
            elif n <= idx < n + target_count:
                continue  # Keep these
            # Remove remaining files (index >= n+target_count)
            else:
                files_to_remove.append(file_path)

        return files_to_remove, None

    else:
        error_msg = f"Invalid remove_from_end value: {remove_from_end}. Must be -1, 0, or positive integer."
        return [], error_msg


def trim_files_in_folder(folder_path: Path, target_count: int, remove_from_end: int,
                         remove_files: bool, verbose: bool = True) -> Tuple[int, Optional[str]]:
    """
    Trim files in a folder to reach target count.

    Args:
        folder_path: Folder containing files to trim
        target_count: Target number of files to keep
        remove_from_end: -1 for end, 0 for front, n>0 for first n files
        remove_files: If True, actually delete files; if False, dry run mode
        verbose: Print detailed information

    Returns:
        Tuple of (number of files removed/would be removed, error message if any)

    Note:
        In dry run mode (remove_files=False), no files are actually deleted.
    """
    files_to_remove, error_msg = get_files_to_remove(folder_path, target_count, remove_from_end)

    # Check for errors
    if error_msg:
        if verbose:
            print(f"    ERROR in {folder_path.name}/: {error_msg}")
        return 0, error_msg

    if not files_to_remove:
        return 0, None

    if verbose:
        action = "Removing" if remove_files else "Would remove"
        print(f"    {action} {len(files_to_remove)} file(s) from {folder_path.name}/")
        if not remove_files and len(files_to_remove) <= 5:
            print(f"      Files: {', '.join([f.name for f in files_to_remove])}")

    if remove_files:
        for file_path in files_to_remove:
            try:
                file_path.unlink()
            except Exception as e:
                print(f"      Error removing {file_path.name}: {e}")

    return len(files_to_remove), None


def trim_subset(subset_path: Path, config: DataTrimConfig) -> Tuple[str, Optional[str], int]:
    """
    Trim all folders in a subset to have the same number of files.

    Args:
        subset_path: Path to subset folder
        config: Data trimming configuration

    Returns:
        Tuple of (status, error_message, files_removed)
        status: "trimmed", "no_trim_needed", "skipped", or "error"
        error_message: Error description if status is "error"
        files_removed: Number of files removed (or would be removed)

    Note:
        This is the main processing function for a single subset.
        Only checks folders specified in config.target_folders.
        Finds minimum file count, then trims all folders to match.
        Returns detailed status for accurate statistics.
    """
    # Find minimum file count across target folders only
    min_count, file_counts = find_minimum_file_count(subset_path, config.target_folders,
                                                      config.trim_mode, config.verbose)

    if min_count == 0:
        if config.verbose:
            print(f"  Warning: Minimum file count is 0, skipping subset")
        return "skipped", "Minimum file count is 0", 0

    # Check if trimming is needed (only consider non-empty folders)
    needs_trimming = any(count > min_count for count in file_counts.values() if count > 0)

    if not needs_trimming:
        if config.verbose:
            print(f"  ✓ All non-empty folders already have {min_count} files, no trimming needed")
        return "no_trim_needed", None, 0

    # Trim files in target folders only
    total_removed = 0
    error_encountered = None

    # Process each target folder
    for folder_name in config.target_folders:
        folder_path = subset_path / folder_name

        # Skip if folder doesn't exist
        if not folder_path.exists():
            continue

        # Handle each folder type
        if folder_name == "image":
            # Trim image subfolders based on trim_mode
            image_subfolders = get_all_subfolders(folder_path)

            if config.trim_mode == "stereo":
                # Only trim left and right
                for camera_name in ['left', 'right']:
                    camera_folder = folder_path / camera_name
                    if camera_folder.exists():
                        folder_rel_path = str(camera_folder.relative_to(subset_path))
                        # Only trim folders that were counted (in file_counts)
                        if folder_rel_path in file_counts:
                            removed, error = trim_files_in_folder(camera_folder, min_count, config.remove_from_end,
                                                                  config.remove_files, config.verbose)
                            if error:
                                return "error", error, 0
                            total_removed += removed
            else:  # "all" mode
                # Trim all image subfolders
                for subfolder in image_subfolders:
                    folder_rel_path = str(subfolder.relative_to(subset_path))
                    # Only trim folders that were counted (in file_counts)
                    if folder_rel_path in file_counts:
                        removed, error = trim_files_in_folder(subfolder, min_count, config.remove_from_end,
                                                              config.remove_files, config.verbose)
                        if error:
                            return "error", error, 0
                        total_removed += removed

        elif folder_name == "time_syn":
            # time_syn has files directly (not in subfolders)
            folder_rel_path = str(folder_path.relative_to(subset_path))
            # Only trim if it was counted (in file_counts)
            if folder_rel_path in file_counts:
                removed, error = trim_files_in_folder(folder_path, min_count, config.remove_from_end,
                                                      config.remove_files, config.verbose)
                if error:
                    return "error", error, 0
                total_removed += removed

        else:
            # For kinematic, annotation, and any other folders: trim all subfolders
            subfolders = get_all_subfolders(folder_path)

            if not subfolders:
                # Folder exists but has no subfolders - trim files directly
                folder_rel_path = str(folder_path.relative_to(subset_path))
                # Only trim if it was counted (in file_counts)
                if folder_rel_path in file_counts:
                    removed, error = trim_files_in_folder(folder_path, min_count, config.remove_from_end,
                                                          config.remove_files, config.verbose)
                    if error:
                        return "error", error, 0
                    total_removed += removed
            else:
                # Trim each subfolder
                for subfolder in subfolders:
                    folder_rel_path = str(subfolder.relative_to(subset_path))
                    # Only trim folders that were counted (in file_counts)
                    if folder_rel_path in file_counts:
                        removed, error = trim_files_in_folder(subfolder, min_count, config.remove_from_end,
                                                              config.remove_files, config.verbose)
                        if error:
                            return "error", error, 0
                        total_removed += removed

    action = "Removed" if config.remove_files else "Would remove"
    if config.verbose:
        print(f"  {action} {total_removed} total file(s)")

    return "trimmed", None, total_removed


def trim_all_subsets(config: DataTrimConfig) -> int:
    """
    Main data trimming function. Processes all subset folders in data directory.

    Args:
        config: Data trimming configuration

    Returns:
        Number of subsets successfully processed

    Note:
        This is the main orchestration function that:
        1. Finds all subset folders
        2. For each subset, finds minimum file count
        3. Trims all folders to match minimum count
        4. Reports results and errors
    """
    data_path = Path(config.data_folder)

    # Validate data folder exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    # Find all subset folders
    subset_folders = find_subset_folders(data_path, config.verbose)

    if not subset_folders:
        print(f"\nNo subset folders found in {data_path}")
        print("A subset folder must contain image/, kinematic/, and time_syn/ subfolders")
        return 0

    print(f"\nFound {len(subset_folders)} subset folder(s) to process")
    print(f"Trim mode: {config.trim_mode}")
    print(f"Remove from end: {config.remove_from_end}")
    if config.remove_from_end == -1:
        print(f"  (Will remove from end - keep indices 0 to n-1)")
    elif config.remove_from_end == 0:
        print(f"  (Will remove from front - keep latest files)")
    else:
        print(f"  (Will remove first {config.remove_from_end} files, then keep next n files)")
    print(f"Remove files: {config.remove_files}")

    # Ask for confirmation if required
    if config.require_confirmation and config.remove_files:
        print(f"\n{'='*70}")
        print("WARNING: This will delete files from your data folders!")
        print(f"{'='*70}")
        response = input("Press ENTER to continue or Ctrl+C to cancel: ")

    # Process each subset
    trimmed_count = 0       # Subsets that had files removed
    no_trim_count = 0       # Subsets that didn't need trimming
    skipped_count = 0       # Subsets skipped (min_count=0, etc.)
    error_count = 0         # Subsets with errors
    total_files_removed = 0 # Total files removed across all subsets
    trimmed_paths = []      # List of trimmed subset paths
    errors = []             # List of (path, error_message) tuples
    skipped_paths = []      # List of skipped subset paths

    print(f"\n{'='*70}")
    print("Starting data trimming...")
    print(f"{'='*70}")

    # Don't use tqdm for verbose mode (conflicts with our detailed output)
    if config.verbose:
        for i, subset_path in enumerate(subset_folders, 1):
            print(f"\n[{i}/{len(subset_folders)}] Processing subset: {subset_path.name}")
            status, error, files_removed = trim_subset(subset_path, config)

            if status == "trimmed":
                trimmed_count += 1
                total_files_removed += files_removed
                trimmed_paths.append(subset_path)
            elif status == "no_trim_needed":
                no_trim_count += 1
            elif status == "skipped":
                skipped_count += 1
                skipped_paths.append(subset_path)
            elif status == "error":
                error_count += 1
                if error:
                    errors.append((subset_path, error))
    else:
        # Use tqdm only in non-verbose mode
        for subset_path in tqdm(subset_folders, desc="Processing subsets"):
            status, error, files_removed = trim_subset(subset_path, config)

            if status == "trimmed":
                trimmed_count += 1
                total_files_removed += files_removed
                trimmed_paths.append(subset_path)
            elif status == "no_trim_needed":
                no_trim_count += 1
            elif status == "skipped":
                skipped_count += 1
                skipped_paths.append(subset_path)
            elif status == "error":
                error_count += 1
                if error:
                    errors.append((subset_path, error))

    # Calculate totals
    total_processed = trimmed_count + no_trim_count
    total_subsets = len(subset_folders)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Data trimming complete!")
    print(f"{'='*70}")
    print(f"Total subsets found: {total_subsets}")
    print(f"")
    print(f"✓ Successfully processed: {total_processed}/{total_subsets} subsets")
    print(f"  - Trimmed (files removed): {trimmed_count} subsets")
    if trimmed_paths:
        print(f"    Trimmed subset paths:")
        for path in trimmed_paths:
            print(f"      {path}")
    print(f"  - Already matched (no trim needed): {no_trim_count} subsets")

    if skipped_count > 0:
        print(f"")
        print(f"⊘ Skipped: {skipped_count} subsets (empty or invalid)")
        if skipped_paths:
            print(f"  Skipped subset paths:")
            for path in skipped_paths:
                print(f"    {path}")

    if error_count > 0:
        print(f"")
        print(f"✗ Errors: {error_count} subsets")
        if errors:
            print(f"  Error details:")
            for subset_path, error in errors:
                print(f"    {subset_path}:")
                print(f"      {error}")

    print(f"")
    if config.remove_files:
        print(f"Total files removed: {total_files_removed:,}")
    else:
        print(f"Total files that would be removed: {total_files_removed:,}")
        print(f"")
        print(f"DRY RUN MODE: No files were actually deleted")
        print(f"Set remove_files=true in config to perform actual trimming")

    print(f"")
    print(f"Data folder: {data_path}")
    print(f"{'='*70}")

    return total_processed


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="data_trim", node=DataTrimConfig)

# Set config path - use absolute path for reliability
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

# Verify config path exists
if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_data_trim"
)
def main(cfg: DictConfig):
    """
    Main entry point for the data trimming script.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf)

    Note:
        This function is decorated with @hydra.main to enable Hydra-based
        configuration management. Configuration can be overridden from command line.

        Example usage:
            python data_trim.py
            python data_trim.py remove_files=true
            python data_trim.py trim_mode=stereo
            python data_trim.py remove_from_end=50
            python data_trim.py data_folder=/path/to/data
    """
    print("="*70)
    print("dVRK Multi-modal Data Trimming Script")
    print("="*70)
    print(f"Workspace: {cfg.workspace}")
    print(f"Data folder: {cfg.data_folder}")
    print(f"Trim mode: {cfg.trim_mode}")
    print(f"Remove from end: {cfg.remove_from_end}")
    print(f"Remove files: {cfg.remove_files}")
    print(f"Verbose: {cfg.verbose}")
    print("="*70)

    # Convert DictConfig to DataTrimConfig
    trim_config = DataTrimConfig(
        workspace=cfg.workspace,
        data_folder=cfg.data_folder,
        trim_mode=cfg.trim_mode,
        remove_from_end=cfg.remove_from_end,
        remove_files=cfg.remove_files,
        verbose=cfg.verbose,
        require_confirmation=cfg.require_confirmation,
        target_folders=list(cfg.target_folders)
    )

    # Run the trimming
    try:
        processed = trim_all_subsets(trim_config)

        # The summary is now printed within trim_all_subsets()
        # Just add a final status message
        if processed == 0:
            print(f"\n⚠ Warning: No subsets were successfully processed")
            print("Please check:")
            print("  1. Data folder contains subset folders")
            print("  2. Subset folders have image/, kinematic/, and time_syn/")
            print("  3. remove_from_end value is valid")

    except Exception as e:
        print(f"\n✗ Error during data trimming: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
