"""
Label Conversion Script for dVRK Multi-modal Data Collection

This script converts annotation labels from old naming convention to new naming convention:
- event -> step (folder name and JSON key "event" -> "step")
- phase -> gesture (folder name and JSON key "phase" -> "gesture")
- contact_detection -> no change (copy as-is)

Input structure:
  <input_folder>/
    <int_number>/
      annotation/
        contact_detection/
          0.json, 1.json, ...
        event/
          0.json, 1.json, ...  # {"event": ...}
        phase/
          0.json, 1.json, ...  # {"phase": {...}}

Output structure:
  <output_folder>/
    <int_number>/
      annotation/
        contact_detection/
          0.json, 1.json, ...
        step/
          0.json, 1.json, ...  # {"step": ...}
        gesture/
          0.json, 1.json, ...  # {"gesture": {...}}

The script processes subfolders in sorted order (numerically, then alphabetically).
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
class LabelConvertConfig:
    """
    Configuration for label conversion script.

    All configuration parameters are at the top level for simplicity and
    consistency with other processing scripts.
    """
    workspace: str
    input_folder: str
    output_folder: str


def natural_sort_key(path: Path) -> Tuple:
    """
    Generate a sort key for natural sorting (numeric then alphabetic).

    Args:
        path: Path object to generate sort key for

    Returns:
        Tuple suitable for sorting (tries to parse as int, falls back to string)

    Note:
        This function enables proper sorting like: 0, 1, 2, 10, 20 instead of 0, 1, 10, 2, 20
        For non-numeric folder names, falls back to alphabetic sorting
    """
    name = path.name
    try:
        # Try to parse as integer for numeric sorting
        return (0, int(name))
    except ValueError:
        # If not a number, use string sorting (secondary priority)
        return (1, name)


def get_sorted_subfolders(root_folder: Path) -> List[Path]:
    """
    Get all subfolders in a directory, sorted numerically then alphabetically.

    Args:
        root_folder: Root directory to search for subfolders

    Returns:
        List of Path objects representing subfolders, sorted naturally

    Note:
        Only returns immediate child directories, not recursive.
        Sorting ensures consistent processing order across runs.
    """
    if not root_folder.exists():
        return []

    # Get all subdirectories
    subfolders = [item for item in root_folder.iterdir() if item.is_dir()]

    # Sort using natural sort key (numeric first, then alphabetic)
    subfolders.sort(key=natural_sort_key)

    return subfolders


def copy_json_files_as_is(src_folder: Path, dst_folder: Path) -> Tuple[bool, Optional[str]]:
    """
    Copy all JSON files from source folder to destination folder without modification.

    Args:
        src_folder: Source folder containing JSON files
        dst_folder: Destination folder for copied files

    Returns:
        Tuple of (success, error_message)
        - success: True if copy succeeded, False otherwise
        - error_message: Error message if failed, None if succeeded

    Note:
        This is used for contact_detection folder which doesn't need any conversion.
        Creates destination folder if it doesn't exist.
    """
    try:
        # Create destination folder if it doesn't exist
        dst_folder.mkdir(parents=True, exist_ok=True)

        # Copy all JSON files
        json_files = list(src_folder.glob("*.json"))

        if not json_files:
            return True, None  # No files to copy is not an error

        for json_file in json_files:
            dst_file = dst_folder / json_file.name
            shutil.copy2(json_file, dst_file)

        return True, None

    except Exception as e:
        return False, f"Error copying JSON files: {e}"


def convert_json_key(src_folder: Path, dst_folder: Path, old_key: str, new_key: str) -> Tuple[bool, Optional[str], int]:
    """
    Convert JSON files by renaming a top-level key from old_key to new_key.

    Args:
        src_folder: Source folder containing JSON files with old key
        dst_folder: Destination folder for converted JSON files
        old_key: Old key name to replace (e.g., "event", "phase")
        new_key: New key name to use (e.g., "step", "gesture")

    Returns:
        Tuple of (success, error_message, converted_count)
        - success: True if conversion succeeded, False otherwise
        - error_message: Error message if failed, None if succeeded
        - converted_count: Number of JSON files successfully converted

    Note:
        This function:
        1. Reads each JSON file
        2. Checks if it has the old_key at top level
        3. Renames old_key to new_key
        4. Writes the modified JSON to destination

        The function preserves all other JSON structure and formatting.
        Files without the old_key are skipped with a warning in the error message.
    """
    try:
        # Create destination folder if it doesn't exist
        dst_folder.mkdir(parents=True, exist_ok=True)

        # Get all JSON files
        json_files = list(src_folder.glob("*.json"))

        if not json_files:
            return True, None, 0  # No files to convert is not an error

        converted_count = 0
        warnings = []

        for json_file in json_files:
            try:
                # Read JSON file
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Check if old_key exists at top level
                if old_key in data:
                    # Create new dictionary with renamed key
                    # This preserves the order and structure of the JSON
                    converted_data = {}
                    for key, value in data.items():
                        if key == old_key:
                            converted_data[new_key] = value
                        else:
                            converted_data[key] = value

                    # Write converted JSON to destination
                    dst_file = dst_folder / json_file.name
                    with open(dst_file, 'w') as f:
                        json.dump(converted_data, f, indent=2)

                    converted_count += 1
                else:
                    # Old key not found - this might be unexpected
                    warnings.append(f"Key '{old_key}' not found in {json_file.name}")
                    # ### If you want to copy the incorrect file as-is, uncomment the following lines:
                    # dst_file = dst_folder / json_file.name
                    # shutil.copy2(json_file, dst_file)

            except json.JSONDecodeError as e:
                warnings.append(f"Invalid JSON in {json_file.name}: {e}")
            except Exception as e:
                warnings.append(f"Error processing {json_file.name}: {e}")

        # Construct error message from warnings if any
        error_msg = "; ".join(warnings) if warnings else None

        return True, error_msg, converted_count

    except Exception as e:
        return False, f"Error converting JSON files: {e}", 0


def process_single_folder(args: Tuple[Path, Path, Path]) -> Tuple[str, bool, Optional[str], Dict[str, int]]:
    """
    Worker function for processing a single annotation folder (for parallel processing).

    Args:
        args: Tuple of (subfolder, input_root, output_root)

    Returns:
        Tuple of (folder_name, success, error_message, stats) where:
        - folder_name: Name of the folder being processed
        - success: True if processing succeeded, False otherwise
        - error_message: Error message if failed, None if succeeded
        - stats: Dictionary with conversion statistics (contact_count, step_count, gesture_count)

    Note:
        This function is designed to be called in parallel by ProcessPoolExecutor.
        It's a standalone function (not a method) to support multiprocessing.

        Processing steps:
        1. Check if annotation folder exists
        2. Copy contact_detection folder as-is (if exists)
        3. Convert event -> step (folder name and JSON key)
        4. Convert phase -> gesture (folder name and JSON key)
    """
    subfolder, input_root, output_root = args
    folder_name = subfolder.name

    # Initialize statistics
    stats = {
        'contact_count': 0,
        'step_count': 0,
        'gesture_count': 0
    }

    warnings = []

    # Check if annotation folder exists
    src_annotation = subfolder / "annotation"
    if not src_annotation.exists():
        return (folder_name, False, f"Annotation folder not found in {folder_name}", stats)

    dst_annotation = output_root / folder_name / "annotation"

    try:
        # Process contact_detection folder (copy as-is)
        src_contact = src_annotation / "contact_detection"
        if src_contact.exists() and src_contact.is_dir():
            dst_contact = dst_annotation / "contact_detection"
            success, error_msg = copy_json_files_as_is(src_contact, dst_contact)
            if success:
                # Count how many files were copied
                stats['contact_count'] = len(list(dst_contact.glob("*.json")))
            else:
                warnings.append(f"contact_detection: {error_msg}")

        # Process event folder -> step (rename folder and convert JSON key)
        src_event = src_annotation / "event"
        if src_event.exists() and src_event.is_dir():
            dst_step = dst_annotation / "step"  # New folder name
            success, error_msg, converted = convert_json_key(src_event, dst_step, "event", "step")
            if success:
                stats['step_count'] = converted
                if error_msg:  # Warnings from conversion
                    warnings.append(f"step: {error_msg}")
            else:
                warnings.append(f"step: {error_msg}")

        # Process phase folder -> gesture (rename folder and convert JSON key)
        src_phase = src_annotation / "phase"
        if src_phase.exists() and src_phase.is_dir():
            dst_gesture = dst_annotation / "gesture"  # New folder name
            success, error_msg, converted = convert_json_key(src_phase, dst_gesture, "phase", "gesture")
            if success:
                stats['gesture_count'] = converted
                if error_msg:  # Warnings from conversion
                    warnings.append(f"gesture: {error_msg}")
            else:
                warnings.append(f"gesture: {error_msg}")

        # Check if at least one folder was processed successfully
        if stats['contact_count'] == 0 and stats['step_count'] == 0 and stats['gesture_count'] == 0:
            error_msg = "No annotation subfolders found or all conversions failed"
            if warnings:
                error_msg += f": {'; '.join(warnings)}"
            return (folder_name, False, error_msg, stats)

        # Return success with any warnings
        final_msg = "; ".join(warnings) if warnings else None
        return (folder_name, True, final_msg, stats)

    except Exception as e:
        return (folder_name, False, f"Unexpected error: {e}", stats)


def convert_labels(
    input_folder: Path,
    output_folder: Path,
    max_workers: Optional[int] = None
) -> Tuple[int, int, Dict[str, int]]:
    """
    Convert label folders from old format to new format.

    Args:
        input_folder: Root input folder containing subfolders with old annotation format
        output_folder: Root output folder for converted annotations
        max_workers: Maximum number of parallel workers (None = use CPU count)

    Returns:
        Tuple of (successful_count, failed_count, total_stats) where:
        - successful_count: Number of successfully converted folders
        - failed_count: Number of failed conversions
        - total_stats: Dictionary with total conversion statistics

    Note:
        This function uses parallel processing to speed up conversion.
        It creates a pool of workers and distributes the conversion operations across them.
        Progress is tracked with tqdm for user feedback.

        The function processes subfolders in sorted order (numerically, then alphabetically)
        to ensure consistent and predictable behavior.
    """
    # Determine number of workers for parallel processing
    if max_workers is None:
        # Use min(CPU count, 4) for balanced performance
        # Too many workers can saturate I/O on some systems
        max_workers = min(multiprocessing.cpu_count(), 4)

    print(f"Using {max_workers} parallel worker(s) for conversion")

    # Get sorted list of subfolders to process
    subfolders = get_sorted_subfolders(input_folder)

    if not subfolders:
        print(f"Warning: No subfolders found in {input_folder}")
        return 0, 0, {}

    print(f"\nConversion plan:")
    print(f"  Source: {input_folder}")
    print(f"  Destination: {output_folder}")
    print(f"  Total folders to process: {len(subfolders)}\n")

    # Prepare arguments for parallel processing
    convert_args = [
        (subfolder, input_folder, output_folder)
        for subfolder in subfolders
    ]

    # Track results
    successful_converts = []
    failed_converts = []
    all_warnings = []
    total_stats = {
        'contact_count': 0,
        'step_count': 0,
        'gesture_count': 0
    }

    # Execute conversions in parallel with progress bar
    print("Starting parallel conversion operations...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_folder, args): args[0].name  # args[0].name is folder name
            for args in convert_args
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(futures), desc="Converting labels") as pbar:
            for future in as_completed(futures):
                folder_name, success, error_msg, stats = future.result()

                if success:
                    successful_converts.append(folder_name)
                    # Accumulate statistics
                    for key in total_stats:
                        total_stats[key] += stats[key]
                    # Print warnings if any
                    if error_msg:
                        all_warnings.append(f"Folder {folder_name}: {error_msg}")
                else:
                    failed_converts.append((folder_name, error_msg))
                    print(f"\n  ✗ Failed: Folder {folder_name} - {error_msg}")

                pbar.update(1)

    # Print summary
    print("\n" + "=" * 70)
    print("Label conversion complete!")
    print(f"Successfully converted: {len(successful_converts)}/{len(subfolders)} folders")
    print(f"\nTotal files converted:")
    print(f"  - contact_detection: {total_stats['contact_count']} files (copied as-is)")
    print(f"  - event -> step: {total_stats['step_count']} files")
    print(f"  - phase -> gesture: {total_stats['gesture_count']} files")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for warning in all_warnings:
            print(f"  ⚠ {warning}")

    if failed_converts:
        print(f"\nFailed conversions ({len(failed_converts)}):")
        for folder_name, error_msg in failed_converts:
            print(f"  ✗ Folder {folder_name}: {error_msg}")

    print("=" * 70)

    return len(successful_converts), len(failed_converts), total_stats


# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="label_convert_config", node=LabelConvertConfig)

# Get project root and config path
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

# Verify config path exists
if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")

@hydra.main(version_base=None, config_path=str(p_config), config_name="config_label_convert_from_old")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for label conversion script.

    Args:
        cfg: Hydra configuration object

    Note:
        This function orchestrates the label conversion process:
        1. Validates input/output folders
        2. Executes parallel conversion operations
        3. Reports results

        The conversion changes:
        - event -> step (folder name and JSON key)
        - phase -> gesture (folder name and JSON key)
        - contact_detection -> no change (copied as-is)
    """
    print("=" * 70)
    print("dVRK Multi-modal Data - Label Conversion Script")
    print("=" * 70)
    print("\nConversion mappings:")
    print("  - event/ -> step/ (JSON key 'event' -> 'step')")
    print("  - phase/ -> gesture/ (JSON key 'phase' -> 'gesture')")
    print("  - contact_detection/ -> contact_detection/ (no change)")
    print()

    # Convert configuration to paths
    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)

    # Validate input folder exists
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder ready: {output_folder}\n")

    # Perform conversion
    try:
        successful, failed, stats = convert_labels(
            input_folder=input_folder,
            output_folder=output_folder,
            max_workers=None  # Auto-detect optimal worker count
        )

        # Exit with error code if any conversions failed
        if failed > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
