"""
Data Reorganization Script for dVRK Multi-modal Data Collection

This script reorganizes raw data from nested folder structures into a flat,
indexed structure suitable for training and processing.

Input structure (nested, date-based):
  <input_folder>/
    camera_calibration/
    data_<date>/
      <int_number>/ or <subfolder>/<int_number>/
        regular/
          image/
            left/, right/, side/
          kinematic/
          time_syn/
        annotation/
      hand_eye_calibration/

Output structure (flat, indexed):
  <output_folder>/
    <start_idx>/
      image/
        left/, right/, side/
      kinematic/
      time_syn/
      annotation/
      camera_calibration/
      hand_eye_calibration/
    <start_idx + 1>/
    ...
    data_organization_note.json
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
from dvrk_data_processing.utils.hydra_config import (
    DataOrganizationConfig,
    UserInfo,
    UserSkillLevel
)
from dataclasses import dataclass

@dataclass
class AppCfg:
    """
    Application configuration with flat structure (no nested path_config/preprocess).

    All configuration parameters are at the top level, matching the simplified
    config_data_org.yaml structure.
    """
    workspace: str
    input_folder: str
    output_folder: str
    copy_image_name: List[str]
    enable_kinematic_copy: bool
    enable_timestamp_copy: bool
    enable_label_copy: bool
    start_idx: int
    user_info: UserInfo


def find_numeric_folders(root_path: Path, max_depth: int = 5) -> List[Tuple[Path, int]]:
    """
    Recursively find all folders with integer names (e.g., "0", "1", "2", etc.)
    up to a specified depth.

    Args:
        root_path: Root directory to search from
        max_depth: Maximum recursion depth

    Returns:
        List of tuples (folder_path, numeric_value) sorted by path

    Note:
        This function efficiently searches for data collection folders which
        are typically named with integers. It skips common non-data directories.

        The search pattern expects:
        - Top level: date-based folders like "data_20250809"
        - Within date folders: either direct numeric folders or task folders containing numeric folders
        - Skips calibration folders at all levels
    """
    numeric_folders = []

    # Directories to skip during search (calibration, git, python cache, etc.)
    skip_dirs = {'.git', '__pycache__', 'camera_calibration', 'hand_eye_calibration',
                 'rectify_resize', 'preprocess', 'raw'}

    def _search(current_path: Path, current_depth: int):
        """Recursive helper function for depth-limited search"""
        if current_depth > max_depth:
            return

        try:
            for item in current_path.iterdir():
                if not item.is_dir():
                    continue

                # Skip directories we know don't contain data or are already processed
                if item.name in skip_dirs or item.name.startswith('.'):
                    continue

                # Check if folder name is a valid integer
                try:
                    num_value = int(item.name)
                    # Found a numeric folder - this is a data collection folder
                    numeric_folders.append((item, num_value))
                    # Don't recurse into numeric folders (they are the leaf data folders)
                except ValueError:
                    # Not a numeric folder, recurse into it to find numeric subfolders
                    _search(item, current_depth + 1)
        except PermissionError:
            # Skip directories we don't have permission to read
            pass

    _search(root_path, 0)

    # Sort by full path for consistent ordering
    numeric_folders.sort(key=lambda x: str(x[0]))

    return numeric_folders


def get_next_available_index(output_folder: Path) -> int:
    """
    Find the next available index in the output folder by checking existing
    numeric folder names.

    Args:
        output_folder: Path to the output directory

    Returns:
        Next available integer index (0 if folder is empty/doesn't exist)

    Note:
        This ensures no gaps in the output folder numbering when using
        start_idx = -1 in the configuration.
    """
    if not output_folder.exists():
        return 0

    max_idx = -1
    for item in output_folder.iterdir():
        if item.is_dir():
            try:
                idx = int(item.name)
                max_idx = max(max_idx, idx)
            except ValueError:
                # Not a numeric folder, skip it
                continue

    return max_idx + 1


def copy_folder_contents(src: Path, dst: Path, description: str = "") -> bool:
    """
    Copy entire folder contents from source to destination with progress tracking.

    Args:
        src: Source folder path
        dst: Destination folder path
        description: Description for progress bar

    Returns:
        True if successful, False otherwise

    Note:
        Uses shutil.copytree for efficient recursive copying. Creates parent
        directories as needed. Preserves file metadata.
    """
    if not src.exists():
        return False

    try:
        # Create parent directory if needed
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy the entire directory tree
        # dirs_exist_ok=True allows copying into existing directories (Python 3.8+)
        shutil.copytree(src, dst, dirs_exist_ok=True)

        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def copy_image_folders(src_image_path: Path, dst_image_path: Path,
                       camera_names: List[str]) -> int:
    """
    Copy selected camera image folders from source to destination.

    Args:
        src_image_path: Source image directory path
        dst_image_path: Destination image directory path
        camera_names: List of camera names to copy (e.g., ["left", "right"])

    Returns:
        Number of camera folders successfully copied

    Note:
        Only copies the camera folders specified in camera_names. This allows
        selective copying when not all camera views are needed.
    """
    dst_image_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for camera_name in camera_names:
        src_cam = src_image_path / camera_name
        dst_cam = dst_image_path / camera_name

        if src_cam.exists():
            if copy_folder_contents(src_cam, dst_cam, f"Copying {camera_name}"):
                copied_count += 1
        else:
            print(f"  Warning: Camera folder '{camera_name}' not found at {src_cam}")

    return copied_count


def find_parent_folder_with_name(start_path: Path, folder_name: str,
                                 max_levels_up: int = 5) -> Optional[Path]:
    """
    Search upward from start_path to find a sibling folder with the given name.

    Args:
        start_path: Starting directory path
        folder_name: Name of the folder to find (e.g., "hand_eye_calibration")
        max_levels_up: Maximum levels to search upward

    Returns:
        Path to the found folder, or None if not found

    Note:
        This is used to locate hand_eye_calibration and camera_calibration folders
        which are typically stored at the date-level directory (e.g., data_20250809/).
    """
    current = start_path

    for _ in range(max_levels_up):
        # Check if folder exists at current level
        target = current / folder_name
        if target.exists() and target.is_dir():
            return target

        # Move up one level
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    return None


def process_single_dataset(src_folder: Path, dst_folder: Path,
                          config: DataOrganizationConfig,
                          input_root: Path) -> bool:
    """
    Process a single dataset folder (one numbered folder from input).

    Args:
        src_folder: Source data folder (e.g., data_20250809/0)
        dst_folder: Destination folder (e.g., raw_data/0)
        config: Data organization configuration
        input_root: Root of input folder (for finding calibration folders)

    Returns:
        True if processing was successful, False otherwise

    Note:
        This function handles the core data copying logic:
        1. Copies selected camera images
        2. Optionally copies kinematic data
        3. Optionally copies timestamp data
        4. Optionally copies annotation data
        5. Copies calibration files (camera and hand-eye)
    """
    regular_path = src_folder / "regular"

    # Validate that regular folder exists
    if not regular_path.exists():
        print(f"  Warning: 'regular' folder not found in {src_folder}")
        return False

    # Create destination folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    success = True

    # 1. Copy image folders for selected cameras
    src_image = regular_path / "image"
    dst_image = dst_folder / "image"
    if src_image.exists():
        copied_cams = copy_image_folders(src_image, dst_image, config.copy_image_name)
        if copied_cams == 0:
            print(f"  Warning: No camera images copied from {src_image}")
            success = False
    else:
        print(f"  Warning: Image folder not found at {src_image}")
        success = False

    # 2. Copy kinematic data if enabled
    if config.enable_kinematic_copy:
        src_kinematic = regular_path / "kinematic"
        dst_kinematic = dst_folder / "kinematic"
        if src_kinematic.exists():
            copy_folder_contents(src_kinematic, dst_kinematic, "kinematic")
        else:
            print(f"  {dst_folder.name} does not contain kinematic folder")

    # 3. Copy timestamp data if enabled
    if config.enable_timestamp_copy:
        src_time_syn = regular_path / "time_syn"
        dst_time_syn = dst_folder / "time_syn"
        if src_time_syn.exists():
            copy_folder_contents(src_time_syn, dst_time_syn, "time_syn")
        else:
            print(f"  {dst_folder.name} does not contain time_syn folder")

    # 4. Copy annotation data if enabled
    if config.enable_label_copy:
        src_annotation = src_folder / "annotation"
        dst_annotation = dst_folder / "annotation"
        if src_annotation.exists():
            copy_folder_contents(src_annotation, dst_annotation, "annotation")
        else:
            print(f"  {dst_folder.name} does not contain annotation folder")

    # 5. Copy camera_calibration (search from input_root)
    camera_calib_src = input_root / "camera_calibration"
    camera_calib_dst = dst_folder / "camera_calibration"
    if camera_calib_src.exists():
        copy_folder_contents(camera_calib_src, camera_calib_dst, "camera_calibration")
    else:
        print(f"  {dst_folder.name} does not contain camera_calibration")

    # 6. Copy hand_eye_calibration (search upward from source folder)
    hand_eye_src = find_parent_folder_with_name(src_folder, "hand_eye_calibration")
    hand_eye_dst = dst_folder / "hand_eye_calibration"
    if hand_eye_src:
        copy_folder_contents(hand_eye_src, hand_eye_dst, "hand_eye_calibration")
    else:
        print(f"  {dst_folder.name} does not contain hand_eye_calibration")

    return success


def update_organization_note(note_path: Path, idx: int, entry: Dict):
    """
    Update the data organization note JSON file with a new entry.

    Args:
        note_path: Path to the JSON note file
        idx: Integer index (start_idx) to use as the key in the dictionary
        entry: Dictionary containing organization metadata for one dataset

    Note:
        The JSON file is a dictionary where keys are string representations of
        the dataset index (start_idx), and values contain:
        - original_data_path: relative path where the data came from (e.g., "data/data_20250809/0")
        - new_data_path: relative path where it was copied to (e.g., "raw_data/0")
        - full_path_name: dictionary with "original" and "new" absolute paths
        - user_info: metadata about the user/operator (user_id, skill_level, description)

        Example structure:
        {
          "0": { "original_data_path": "...", "new_data_path": "...", ... },
          "1": { "original_data_path": "...", "new_data_path": "...", ... },
          ...
        }

        The entries are written in sequential order (0, 1, 2, ...) to maintain
        numerical ordering in the JSON file.
    """
    # Load existing notes or create new dictionary
    if note_path.exists() and note_path.stat().st_size > 0:
        try:
            with open(note_path, 'r') as f:
                notes = json.load(f)
        except (json.JSONDecodeError, ValueError):
            # If file is corrupted or empty, start fresh
            notes = {}
    else:
        notes = {}

    # Add new entry with integer key (converted to string for JSON)
    notes[str(idx)] = entry

    # Write back to file with nice formatting
    # Sort keys numerically (not alphabetically) to ensure sequential order: 0, 1, 2, ...
    # This prevents alphabetical ordering like "0", "1", "10", "2", ...
    with open(note_path, 'w') as f:
        sorted_notes = {k: notes[k] for k in sorted(notes.keys(), key=lambda x: int(x))}
        json.dump(sorted_notes, f, indent=2)


def app_cfg_to_data_org_config(cfg) -> DataOrganizationConfig:
    """
    Convert flat AppCfg (or DictConfig from Hydra) to DataOrganizationConfig for internal processing.

    Args:
        cfg: Flat application configuration from Hydra (can be DictConfig or AppCfg)

    Returns:
        DataOrganizationConfig object for processing functions

    Note:
        Handles conversion from Hydra's DictConfig to proper dataclass objects.
        Constructs UserInfo and UserSkillLevel from nested dictionaries.
    """
    # Handle user_info which might be a DictConfig or dict
    if hasattr(cfg.user_info, 'user_skill_level'):
        # Extract user_skill_level data
        skill_level_data = cfg.user_info.user_skill_level
        user_skill_level = UserSkillLevel(
            dVRK=skill_level_data.get('dVRK', 0) if hasattr(skill_level_data, 'get') else skill_level_data.dVRK,
            clinical=skill_level_data.get('clinical', 0) if hasattr(skill_level_data, 'get') else skill_level_data.clinical
        )

        # Construct UserInfo with the UserSkillLevel
        user_info = UserInfo(
            user_id=cfg.user_info.get('user_id', '') if hasattr(cfg.user_info, 'get') else cfg.user_info.user_id,
            user_skill_level=user_skill_level,
            user_description=cfg.user_info.get('user_description', '') if hasattr(cfg.user_info, 'get') else cfg.user_info.user_description
        )
    else:
        # Fallback: use default UserInfo
        user_info = UserInfo()

    return DataOrganizationConfig(
        stage="data_organization",
        input_folder=cfg.input_folder,
        output_folder=cfg.output_folder,
        copy_image_name=list(cfg.copy_image_name),  # Convert to list if needed
        enable_kinematic_copy=cfg.enable_kinematic_copy,
        enable_timestamp_copy=cfg.enable_timestamp_copy,
        enable_label_copy=cfg.enable_label_copy,
        start_idx=cfg.start_idx,
        user_info=user_info,
        folder_initialize=False  # Default value
    )


def organize_data(config: DataOrganizationConfig) -> int:
    """
    Main data organization function. Processes all numeric folders in input
    and reorganizes them into the output folder.

    Args:
        config: Data organization configuration

    Returns:
        Number of datasets successfully processed

    Note:
        This is the main orchestration function that:
        1. Finds all numeric folders in input
        2. Determines starting index for output
        3. Processes each dataset folder
        4. Updates the organization note file
    """
    input_path = Path(config.input_folder)
    output_path = Path(config.output_folder)

    # Validate input folder exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Create output folder if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all numeric folders in input
    print(f"Searching for numeric folders in {input_path}...")
    numeric_folders = find_numeric_folders(input_path)

    if not numeric_folders:
        print(f"No numeric folders found in {input_path}")
        return 0

    print(f"Found {len(numeric_folders)} numeric folder(s) to process")

    # Determine starting index
    if config.start_idx == -1:
        start_idx = get_next_available_index(output_path)
        print(f"Auto-detected starting index: {start_idx}")
    else:
        start_idx = config.start_idx
        print(f"Using specified starting index: {start_idx}")

    # Process each numeric folder
    note_path = output_path / "data_organization_note.json"
    processed_count = 0

    for i, (src_folder, _) in enumerate(tqdm(numeric_folders, desc="Processing datasets")):
        dst_idx = start_idx + i
        dst_folder = output_path / str(dst_idx)

        print(f"\n[{i+1}/{len(numeric_folders)}] Processing: {src_folder} -> {dst_folder}")

        # Process the dataset
        success = process_single_dataset(src_folder, dst_folder, config, input_path)

        if success:
            processed_count += 1

            # Compute relative paths for the organization note
            # new_data_path: relative to workspace (e.g., "raw_data/0")
            try:
                new_data_rel = dst_folder.relative_to(input_path.parent)
            except ValueError:
                # If paths are not relative, just use the folder name structure
                new_data_rel = Path(output_path.name) / str(dst_idx)

            # original_data_path: relative starting from /data/...
            # Extract the path starting from the data folder
            src_parts = src_folder.parts
            # Find the index of 'data' in the path
            try:
                data_idx = next(i for i, part in enumerate(src_parts) if part == 'data')
                original_data_rel = Path(*src_parts[data_idx:])
            except StopIteration:
                # If 'data' is not found, try to make it relative to input_path parent
                try:
                    original_data_rel = src_folder.relative_to(input_path.parent)
                except ValueError:
                    original_data_rel = src_folder

            # Create organization note entry with both relative and full paths
            note_entry = {
                "original_data_path": str(original_data_rel),
                "new_data_path": str(new_data_rel),
                "full_path_name": {
                    "original": str(src_folder.resolve()),
                    "new": str(dst_folder.resolve())
                },
                "user_info": {
                    "user_id": config.user_info.user_id,
                    "user_skill_level": {
                        "dVRK": config.user_info.user_skill_level.dVRK,
                        "clinical": config.user_info.user_skill_level.clinical
                    },
                    "user_description": config.user_info.user_description
                }
            }

            # Update the organization note with the dataset index as key
            update_organization_note(note_path, dst_idx, note_entry)

    print(f"\n{'='*70}")
    print(f"Data organization complete!")
    print(f"Successfully processed: {processed_count}/{len(numeric_folders)} datasets")
    print(f"Output folder: {output_path}")
    print(f"Organization note: {note_path}")
    print(f"{'='*70}")

    return processed_count


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="data_organization", node=AppCfg)

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
    config_name="config_data_org"  # Update this to your personal config if needed
)
def main(cfg: DictConfig):
    """
    Main entry point for the data reorganization script.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf)

    Note:
        This function is decorated with @hydra.main to enable Hydra-based
        configuration management. Configuration can be overridden from command line.
        The cfg parameter is a DictConfig, not the AppCfg dataclass.
    """
    print("="*70)
    print("dVRK Multi-modal Data Organization Script")
    print("="*70)
    print(f"Workspace: {cfg.workspace}")
    print(f"Input folder: {cfg.input_folder}")
    print(f"Output folder: {cfg.output_folder}")
    print(f"Camera names to copy: {cfg.copy_image_name}")
    print(f"Copy kinematic: {cfg.enable_kinematic_copy}")
    print(f"Copy timestamp: {cfg.enable_timestamp_copy}")
    print(f"Copy annotations: {cfg.enable_label_copy}")
    print(f"Starting index: {cfg.start_idx}")
    print("="*70)

    # Convert flat config to DataOrganizationConfig
    data_org_config = app_cfg_to_data_org_config(cfg)

    # Run the organization
    try:
        processed = organize_data(data_org_config)

        if processed > 0:
            print(f"\n✓ Successfully organized {processed} dataset(s)")
        else:
            print(f"\n✗ No datasets were processed")

    except Exception as e:
        print(f"\n✗ Error during data organization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
