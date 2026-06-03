"""
Time Synchronization Data Reorganization Script for dVRK Multi-modal Data Collection

This script reorganizes time synchronization test JSON files based on the data
organization note. It finds folders containing <camera_name>_sync_test.json files
and copies ONLY those JSON files to the reorganized structure.

Workflow:
1. Scan input folder (test_sync) for folders containing *_sync_test.json files
2. Extract the relative path structure (e.g., open_h_new_data/data_20251210/dissection/0)
3. Look up the matching entry in data_organization_note.json based on original_data_path
4. Copy ONLY the *_sync_test.json files to the new_data_path folder structure
5. Preserve the folder structure from org note's new_data_path in the output

Input structure (test_sync):
  <input_folder>/
    open_h_new_data/
      data_20251210/
        dissection/
          0/
            left_sync_test.json
            right_sync_test.json
            side1_sync_test.json
          1/
            ...

Data organization note format (data_organization_note.json):
  {
    "0": {
      "original_data_path": "open_h_new_data/data_20251210/dissection/0",
      "new_data_path": "raw/0",
      "full_path": {
        "original": "/absolute/path/to/original",
        "new": "/absolute/path/to/new"
      },
      "user_info": {...}
    },
    ...
  }

Output structure:
  <output_folder>/
    raw/                              # Following new_data_path from org note
      0/
        left_sync_test.json           # Only sync test JSON files
        right_sync_test.json
        side1_sync_test.json
      1/
        left_sync_test.json
        right_sync_test.json
        ...
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm


@dataclass
class SyncReorgConfig:
    """
    Configuration for time synchronization data reorganization.

    All configuration parameters are at the top level for simplicity.
    """
    workspace: str
    input_folder: str
    output_folder: str
    data_org_note: str  # Path to data_organization_note.json
    folder_initialize: bool
    verbose: bool


def find_sync_test_folders(root_path: Path, verbose: bool = True) -> List[Tuple[Path, str]]:
    """
    Recursively find all folders containing <camera_name>_sync_test.json files.

    Args:
        root_path: Root directory to search from (e.g., test_sync folder)
        verbose: Print detailed information during search

    Returns:
        List of tuples (folder_path, relative_path_from_root)
        where relative_path_from_root is the path structure relative to root_path

    Note:
        A folder is considered a sync test folder if it contains at least one file
        matching the pattern *_sync_test.json (e.g., left_sync_test.json,
        right_sync_test.json, side1_sync_test.json).

        The relative path is important for matching with the data_organization_note.json
        file, which uses paths like "open_h_new_data/data_20251210/dissection/0".
    """
    sync_folders = []

    # Skip directories that should not be searched
    skip_dirs = {'.git', '__pycache__', 'camera_calibration', 'hand_eye_calibration'}

    if verbose:
        print(f"Searching for folders with *_sync_test.json files in {root_path}...")

    # Walk through all directories recursively
    for current_path in root_path.rglob('*'):
        if not current_path.is_dir():
            continue

        # Skip directories we know don't contain sync test data
        if current_path.name in skip_dirs or current_path.name.startswith('.'):
            continue

        # Check if this folder contains any *_sync_test.json files
        sync_json_files = list(current_path.glob('*_sync_test.json'))

        if sync_json_files:
            # Found a sync test folder! Extract the relative path from root
            try:
                relative_path = current_path.relative_to(root_path)
                sync_folders.append((current_path, str(relative_path)))

                if verbose:
                    print(f"  Found sync test folder: {relative_path}")
                    print(f"    Contains: {', '.join([f.name for f in sync_json_files])}")
            except ValueError:
                # Path is not relative to root_path, skip it
                if verbose:
                    print(f"  Warning: Could not compute relative path for {current_path}")
                continue

    return sync_folders


def load_data_organization_note(note_file: Path) -> Dict[str, Dict]:
    """
    Load the data_organization_note.json file.

    Args:
        note_file: Path to data_organization_note.json

    Returns:
        Dictionary mapping dataset indices to organization metadata

    Note:
        The organization note contains mappings like:
        {
          "0": {
            "original_data_path": "open_h_new_data/data_20251210/dissection/0",
            "new_data_path": "raw/0",
            "full_path": {"original": "/abs/path/orig", "new": "/abs/path/new"},
            "user_info": {...}
          },
          ...
        }

        We need this to map from the sync test folder structure to the actual
        data locations.
    """
    if not note_file.exists():
        raise FileNotFoundError(f"Data organization note file not found: {note_file}")

    try:
        with open(note_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in organization note file {note_file}: {e}")


def create_reverse_mapping(org_note: Dict[str, Dict]) -> Dict[str, Tuple[str, Dict]]:
    """
    Create a reverse mapping from original_data_path to (dataset_idx, entry).

    Args:
        org_note: Data organization note dictionary

    Returns:
        Dictionary mapping original_data_path -> (dataset_idx, full_entry)

    Note:
        This enables fast lookup when we have a sync test folder path and need
        to find the corresponding entry in the organization note.

        Example:
        {
          "open_h_new_data/data_20251210/dissection/0": ("0", {full entry dict}),
          "open_h_new_data/data_20251210/dissection/1": ("1", {full entry dict}),
          ...
        }
    """
    reverse_map = {}

    for dataset_idx, entry in org_note.items():
        if "original_data_path" in entry:
            original_path = entry["original_data_path"]
            reverse_map[original_path] = (dataset_idx, entry)

    return reverse_map


def copy_sync_test_json_files(sync_test_folder: Path, dst_folder: Path, verbose: bool = True) -> int:
    """
    Copy only the <camera_name>_sync_test.json files from sync test folder to destination.

    Args:
        sync_test_folder: Source folder containing *_sync_test.json files
        dst_folder: Destination folder where JSON files will be copied
        verbose: Print detailed copy information

    Returns:
        Number of sync test JSON files successfully copied

    Note:
        This function only copies the sync test JSON files (e.g., left_sync_test.json,
        right_sync_test.json, side1_sync_test.json), not the entire dataset folder.
        The destination folder structure is created based on the organization note's
        new_data_path.
    """
    if not sync_test_folder.exists():
        if verbose:
            print(f"  Error: Source folder does not exist: {sync_test_folder}")
        return 0

    # Find all *_sync_test.json files in the sync test folder
    sync_json_files = list(sync_test_folder.glob('*_sync_test.json'))

    if not sync_json_files:
        if verbose:
            print(f"  Warning: No sync test JSON files found in {sync_test_folder}")
        return 0

    try:
        # Create destination folder if it doesn't exist
        dst_folder.mkdir(parents=True, exist_ok=True)

        copied_count = 0

        # Copy each sync test JSON file
        for src_file in sync_json_files:
            dst_file = dst_folder / src_file.name

            if verbose:
                print(f"  Copying: {src_file.name}")

            shutil.copy2(src_file, dst_file)
            copied_count += 1

        if verbose:
            print(f"  -> Copied {copied_count} sync test JSON file(s) to {dst_folder}")

        return copied_count

    except Exception as e:
        print(f"  Error copying sync test files from {sync_test_folder} to {dst_folder}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return 0


def clear_output_folder(output_path: Path, verbose: bool = True):
    """
    Clear the output folder contents.

    Args:
        output_path: Path to output folder
        verbose: Print detailed information

    Note:
        This is a destructive operation. User confirmation is required.
    """
    if not output_path.exists():
        if verbose:
            print(f"Output folder does not exist yet: {output_path}")
        return

    # Ask for user confirmation before deleting
    print(f"\nWARNING: You are about to remove all files in {output_path}")
    response = input("Press ENTER to continue or Ctrl+C to cancel: ")

    for item in output_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                if verbose:
                    print(f"  Removed subfolder: {item}")
            else:
                item.unlink(missing_ok=True)
                if verbose:
                    print(f"  Removed file: {item}")
        except Exception as e:
            print(f"  Error removing {item}: {e}")


def reorganize_sync_data(config: SyncReorgConfig) -> int:
    """
    Main time synchronization data reorganization function.

    Args:
        config: Synchronization reorganization configuration

    Returns:
        Number of sync test folders successfully processed

    Note:
        This is the main orchestration function that:
        1. Finds all sync test folders in input (folders containing *_sync_test.json)
        2. Loads the data organization note
        3. Creates reverse mapping for fast lookup
        4. For each sync test folder, finds the corresponding entry in org note
        5. Copies ONLY the *_sync_test.json files (not entire dataset)
        6. Maintains the folder structure from new_data_path in output
    """
    input_path = Path(config.input_folder)
    output_path = Path(config.output_folder)
    org_note_path = Path(config.data_org_note)

    # Validate input folder exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Load data organization note
    if config.verbose:
        print(f"\nLoading data organization note from {org_note_path}...")
    org_note = load_data_organization_note(org_note_path)

    if config.verbose:
        print(f"Loaded {len(org_note)} entries from organization note")

    # Create reverse mapping for fast lookup
    reverse_map = create_reverse_mapping(org_note)

    if config.verbose:
        print(f"Created reverse mapping with {len(reverse_map)} paths")

    # Find all sync test folders
    sync_folders = find_sync_test_folders(input_path, config.verbose)

    if not sync_folders:
        print(f"\nNo sync test folders found in {input_path}")
        print("A sync test folder should contain at least one *_sync_test.json file")
        return 0

    print(f"\nFound {len(sync_folders)} sync test folder(s) to process")

    # Clear output folder if requested
    if config.folder_initialize:
        clear_output_folder(output_path, config.verbose)

    # Create output folder if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each sync test folder
    processed_count = 0
    not_found_count = 0
    failed_count = 0
    not_found_paths = []

    print(f"\n{'='*70}")
    print("Starting data reorganization...")
    print(f"{'='*70}\n")

    for i, (sync_folder, relative_path) in enumerate(tqdm(sync_folders, desc="Processing sync folders")):
        print(f"\n[{i+1}/{len(sync_folders)}] Processing: {relative_path}")

        # Look up this path in the reverse mapping
        if relative_path not in reverse_map:
            print(f"  Warning: No matching entry found in organization note")
            print(f"           Looking for: {relative_path}")
            not_found_count += 1
            not_found_paths.append(relative_path)
            continue

        # Get the corresponding dataset index and entry
        dataset_idx, entry = reverse_map[relative_path]

        # Construct output path
        # We want to preserve the new_data_path structure in our output
        # e.g., if new_data_path is "raw/0", we copy sync test files to output_folder/raw/0
        if "new_data_path" in entry:
            new_data_rel = entry["new_data_path"]
            output_dataset_path = output_path / new_data_rel
        else:
            # Fallback: use dataset_idx as folder name
            output_dataset_path = output_path / dataset_idx

        if config.verbose:
            print(f"  Dataset index: {dataset_idx}")
            print(f"  Original path: {entry.get('original_data_path', 'N/A')}")
            print(f"  New path: {entry.get('new_data_path', 'N/A')}")

        # Copy only the sync test JSON files from the sync test folder
        # sync_folder is the folder in test_sync that contains the *_sync_test.json files
        num_copied = copy_sync_test_json_files(sync_folder, output_dataset_path, config.verbose)

        if num_copied > 0:
            processed_count += 1
        else:
            failed_count += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"Time synchronization data reorganization complete!")
    print(f"{'='*70}")
    print(f"Successfully processed: {processed_count}/{len(sync_folders)} sync test folders")
    print(f"(Copied *_sync_test.json files only, not entire datasets)")

    if not_found_count > 0:
        print(f"Not found in org note: {not_found_count} datasets")
        if config.verbose and not_found_paths:
            print("\nPaths not found in organization note:")
            for path in not_found_paths:
                print(f"  - {path}")

    if failed_count > 0:
        print(f"Failed to copy: {failed_count} datasets")

    print(f"\nInput folder: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Organization note: {org_note_path}")
    print(f"{'='*70}")

    return processed_count


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="sync_reorg", node=SyncReorgConfig)

# Set config path - use absolute path for reliability
# Get the project root (parent of scripts directory)
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

# Verify config path exists
if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_sync_reorg"
)
def main(cfg: DictConfig):
    """
    Main entry point for the time synchronization data reorganization script.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf)

    Note:
        This function is decorated with @hydra.main to enable Hydra-based
        configuration management. Configuration can be overridden from command line.

        Example usage:
            python time_sync_reorg.py
            python time_sync_reorg.py verbose=false
            python time_sync_reorg.py output_folder=/custom/path
    """
    print("="*70)
    print("dVRK Time Synchronization Data Reorganization Script")
    print("="*70)
    print(f"Workspace: {cfg.workspace}")
    print(f"Input folder: {cfg.input_folder}")
    print(f"Output folder: {cfg.output_folder}")
    print(f"Organization note: {cfg.data_org_note}")
    print(f"Folder initialize: {cfg.folder_initialize}")
    print(f"Verbose mode: {cfg.verbose}")
    print("="*70)

    # Convert DictConfig to SyncReorgConfig
    sync_config = SyncReorgConfig(
        workspace=cfg.workspace,
        input_folder=cfg.input_folder,
        output_folder=cfg.output_folder,
        data_org_note=cfg.data_org_note,
        folder_initialize=cfg.folder_initialize,
        verbose=cfg.verbose
    )

    # Run the reorganization
    try:
        processed = reorganize_sync_data(sync_config)

        if processed > 0:
            print(f"\n✓ Successfully reorganized sync test JSON files from {processed} folder(s)")
        else:
            print(f"\n✗ No sync test folders were reorganized")
            print("Please check:")
            print("  1. Input folder contains folders with *_sync_test.json files")
            print("  2. data_organization_note.json contains matching entries")
            print("  3. The relative paths in test_sync match original_data_path in org note")

    except Exception as e:
        print(f"\n✗ Error during synchronization data reorganization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
