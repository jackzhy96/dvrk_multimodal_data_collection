"""
Label Reorganization Script for dVRK Multi-modal Data Collection

This script reorganizes annotation labels from nested folder structures into a flat,
indexed structure based on data_organization_note.json mapping.

Input structure (nested, date-based):
  <input_folder>/
    data_<date>/
      <subfolder>/.../
        <int_number>/
          annotation/
            contact_detection/
              0.json, 1.json, ...
            event/
              0.json, 1.json, ...
            phase/
              0.json, 1.json, ...

Output structure (flat, indexed):
  <output_folder>/
    <new_int_number>/
      annotation/
        contact_detection/
          0.json, 1.json, ...
        event/
          0.json, 1.json, ...
        phase/
          0.json, 1.json, ...

The mapping between original paths and new indices is defined in data_organization_note.json.
"""

import os
import sys
import json
import shutil
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class LabelReorgConfig:
    """
    Configuration for label reorganization script.

    All configuration parameters are at the top level for simplicity and
    consistency with other processing scripts.
    """
    workspace: str
    input_folder: str
    output_folder: str
    reorg_config: str  # Path to data_organization_note.json


def load_reorganization_mapping(reorg_file: Path) -> Dict[str, str]:
    """
    Load the reorganization mapping from data_organization_note.json.

    Args:
        reorg_file: Path to data_organization_note.json

    Returns:
        Dictionary mapping new output indices to original data paths
        Format: {"0": "JHU/data_20250911/suturing/strict_match/1", ...}

    Note:
        The data_organization_note.json contains metadata for each dataset including:
        - original_data_path: relative path in the source data structure
        - new_data_path: relative path in the reorganized structure (e.g., "raw/0")

        We extract the mapping: new_index -> original_data_path for label copying.
    """
    if not reorg_file.exists():
        raise FileNotFoundError(f"Reorganization config file not found: {reorg_file}")

    try:
        with open(reorg_file, 'r') as f:
            reorg_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in reorganization file {reorg_file}: {e}")

    # Build mapping: new_index -> original_data_path
    # The JSON has format: {"0": {"original_data_path": "...", "new_data_path": "raw/0", ...}, ...}
    mapping = {}
    for idx, entry in reorg_data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry format for index {idx}: expected dict, got {type(entry)}")

        if "original_data_path" not in entry:
            raise ValueError(f"Missing 'original_data_path' in entry for index {idx}")

        # Extract the original data path
        original_path = entry["original_data_path"]
        mapping[idx] = original_path

    return mapping


def find_annotation_folder(input_root: Path, original_path: str) -> Optional[Path]:
    """
    Find the annotation folder for a given original data path.

    Args:
        input_root: Root input folder (e.g., data/raw_label)
        original_path: Original data path from mapping (e.g., "JHU/data_20250911/suturing/strict_match/1")

    Returns:
        Path to annotation folder if found, None otherwise

    Note:
        The function searches for the annotation folder by traversing the path structure.
        It handles various nesting levels and validates that the annotation folder exists.

        Expected structure: <input_root>/<original_path>/annotation/
        or variations where we need to find the numeric folder containing annotation/
    """
    # Strategy 1: Direct path construction
    # Try: input_root / original_path / annotation
    direct_path = input_root / original_path / "annotation"
    if direct_path.exists() and direct_path.is_dir():
        return direct_path

    # Strategy 2: Search in data_<date> folders
    # Extract the data_<date> folder and search within it
    path_parts = Path(original_path).parts

    # Find the data_<date> part
    data_date_folder = None
    remaining_parts = []
    for i, part in enumerate(path_parts):
        if part.startswith('data_'):
            data_date_folder = part
            remaining_parts = path_parts[i+1:]  # Everything after data_<date>
            break

    if data_date_folder:
        # Construct the search path
        search_root = input_root / data_date_folder
        if search_root.exists():
            # Navigate through remaining parts to find the annotation folder
            current_path = search_root
            for part in remaining_parts:
                current_path = current_path / part
                if not current_path.exists():
                    # Path doesn't exist in this structure
                    break

            # Check if annotation folder exists at this location
            annotation_path = current_path / "annotation"
            if annotation_path.exists() and annotation_path.is_dir():
                return annotation_path

    # If we couldn't find it, return None
    return None


def has_json_files(folder: Path) -> bool:
    """
    Check if a folder contains any JSON files.

    Args:
        folder: Path to folder to check

    Returns:
        True if folder contains at least one .json file, False otherwise

    Note:
        Only checks for direct children (non-recursive) to avoid
        counting nested JSON files.
    """
    if not folder.exists() or not folder.is_dir():
        return False

    # Check if any .json files exist in this folder
    return any(f.suffix.lower() == '.json' for f in folder.iterdir() if f.is_file())


def copy_annotation_folder(src_annotation: Path, dst_annotation: Path) -> Tuple[bool, List[str]]:
    """
    Copy annotation folder from source to destination, only copying subfolders with JSON files.

    Args:
        src_annotation: Source annotation folder path
        dst_annotation: Destination annotation folder path

    Returns:
        Tuple of (success, warnings) where:
        - success: True if at least one subfolder was copied, False otherwise
        - warnings: List of warning messages for skipped subfolders

    Note:
        This function selectively copies annotation subfolders:
        - contact_detection/ (only if it contains .json files)
        - event/ (only if it contains .json files)
        - phase/ (only if it contains .json files)
        - Any other subfolders (only if they contain .json files)

        IMPORTANT: Subfolders without JSON files are skipped with warnings.
        This prevents creating empty output folder structures.
    """
    if not src_annotation.exists():
        return False, [f"Source annotation folder does not exist: {src_annotation}"]

    try:
        warnings = []
        copied_count = 0

        # Remove destination if it already exists to ensure clean copy
        if dst_annotation.exists():
            shutil.rmtree(dst_annotation)

        # Iterate through each subfolder in the annotation folder
        for subfolder in src_annotation.iterdir():
            if not subfolder.is_dir():
                # Skip non-directory items
                continue

            # Check if this subfolder contains JSON files
            if has_json_files(subfolder):
                # Create destination annotation folder if this is the first subfolder
                if copied_count == 0:
                    dst_annotation.mkdir(parents=True, exist_ok=True)

                # Copy this subfolder to destination
                dst_subfolder = dst_annotation / subfolder.name
                shutil.copytree(subfolder, dst_subfolder, symlinks=False,
                              copy_function=shutil.copy2,
                              ignore_dangling_symlinks=False,
                              dirs_exist_ok=False)
                copied_count += 1
            else:
                # Subfolder is empty or doesn't contain JSON files - skip it
                warning_msg = f"Skipping empty subfolder '{subfolder.name}' (no JSON files found)"
                warnings.append(warning_msg)

        # If no subfolders were copied, consider it a failure
        if copied_count == 0:
            warnings.append(f"No valid subfolders with JSON files found in {src_annotation}")
            return False, warnings

        return True, warnings

    except Exception as e:
        error_msg = f"Error copying annotation folder from {src_annotation} to {dst_annotation}: {e}"
        return False, [error_msg]


def copy_single_label(args: Tuple[str, str, Path, Path]) -> Tuple[str, bool, Optional[str], List[str]]:
    """
    Worker function for copying a single label dataset (for parallel processing).

    Args:
        args: Tuple of (new_idx, original_path, input_root, output_root)

    Returns:
        Tuple of (new_idx, success, error_message, warnings) where:
        - new_idx: The output index being processed
        - success: True if copy succeeded, False otherwise
        - error_message: Error message if failed, None if succeeded
        - warnings: List of warning messages (e.g., skipped empty subfolders)

    Note:
        This function is designed to be called in parallel by ProcessPoolExecutor.
        It's a standalone function (not a method) to support multiprocessing.
    """
    new_idx, original_path, input_root, output_root = args

    # Find source annotation folder
    src_annotation = find_annotation_folder(input_root, original_path)

    if src_annotation is None:
        error_msg = f"Could not find annotation folder for {original_path}"
        return (new_idx, False, error_msg, [])

    # Construct destination path: output_root / new_idx / annotation
    dst_annotation = output_root / new_idx / "annotation"

    # Copy the annotation folder (only subfolders with JSON files)
    success, warnings = copy_annotation_folder(src_annotation, dst_annotation)

    if success:
        return (new_idx, True, None, warnings)
    else:
        # Combine warnings into error message
        error_msg = "; ".join(warnings) if warnings else f"Failed to copy annotation from {src_annotation}"
        return (new_idx, False, error_msg, [])


def reorganize_labels(
    input_folder: Path,
    output_folder: Path,
    mapping: Dict[str, str],
    max_workers: Optional[int] = None
) -> Tuple[int, int]:
    """
    Reorganize label folders based on the provided mapping.

    Args:
        input_folder: Root input folder containing nested label structure
        output_folder: Root output folder for flat indexed structure
        mapping: Dictionary mapping new indices to original data paths
        max_workers: Maximum number of parallel workers (None = use CPU count)

    Returns:
        Tuple of (successful_count, failed_count)

    Note:
        This function uses parallel processing to speed up the reorganization.
        It creates a pool of workers and distributes the copy operations across them.
        Progress is tracked with tqdm for user feedback.
    """
    # Determine number of workers for parallel processing
    if max_workers is None:
        # Use min(CPU count, 4) for balanced performance
        # Too many workers can saturate I/O on some systems
        max_workers = min(multiprocessing.cpu_count(), 4)

    print(f"Using {max_workers} parallel worker(s) for copying")
    print(f"\nReorganization plan:")
    print(f"  Source: {input_folder}")
    print(f"  Destination: {output_folder}")
    print(f"  Total datasets to process: {len(mapping)}\n")

    # Prepare arguments for parallel processing
    copy_args = [
        (new_idx, original_path, input_folder, output_folder)
        for new_idx, original_path in mapping.items()
    ]

    # Track results
    successful_copies = []
    failed_copies = []
    all_warnings = []

    # Execute copies in parallel with progress bar
    print("Starting parallel copy operations...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(copy_single_label, args): args[0]  # args[0] is new_idx
            for args in copy_args
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(futures), desc="Reorganizing labels") as pbar:
            for future in as_completed(futures):
                new_idx, success, error_msg, warnings = future.result()

                if success:
                    successful_copies.append(new_idx)
                    # Print warnings if any subfolders were skipped
                    if warnings:
                        for warning in warnings:
                            all_warnings.append(f"Index {new_idx}: {warning}")
                else:
                    failed_copies.append((new_idx, error_msg))
                    print(f"\n  ✗ Failed: Index {new_idx} - {error_msg}")

                pbar.update(1)

    # Print summary
    print("\n" + "=" * 70)
    print("Label reorganization complete!")
    print(f"Successfully reorganized: {len(successful_copies)}/{len(mapping)} datasets")

    if all_warnings:
        print(f"\nWarnings (skipped empty subfolders):")
        for warning in all_warnings:
            print(f"  ⚠ {warning}")

    if failed_copies:
        print(f"\nFailed copies ({len(failed_copies)}):")
        for idx, error_msg in failed_copies:
            print(f"  ✗ Index {idx}: {error_msg}")

    print("=" * 70)

    return len(successful_copies), len(failed_copies)


# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="label_reorg_config", node=LabelReorgConfig)

project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

# Verify config path exists
if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")

@hydra.main(version_base=None, config_path=str(p_config), config_name="config_label_org")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for label reorganization script.

    Args:
        cfg: Hydra configuration object

    Note:
        This function orchestrates the label reorganization process:
        1. Loads the reorganization mapping from JSON
        2. Validates input/output folders
        3. Executes parallel copy operations
        4. Reports results
    """
    print("=" * 70)
    print("dVRK Multi-modal Data - Label Reorganization Script")
    print("=" * 70)

    # Convert configuration to paths
    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)
    reorg_config_file = Path(cfg.reorg_config)

    # Validate input folder exists
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder ready: {output_folder}\n")

    # Load reorganization mapping
    print(f"Loading reorganization mapping from: {reorg_config_file}")
    try:
        mapping = load_reorganization_mapping(reorg_config_file)
        print(f"Found {len(mapping)} label mapping(s)\n")
    except Exception as e:
        print(f"Error loading reorganization mapping: {e}")
        sys.exit(1)

    # Perform reorganization
    try:
        successful, failed = reorganize_labels(
            input_folder=input_folder,
            output_folder=output_folder,
            mapping=mapping,
            max_workers=None  # Auto-detect optimal worker count
        )

        # Exit with error code if any copies failed
        if failed > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nReorganization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during reorganization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
