#!/usr/bin/env python3
"""
Old Data Format Converter for dVRK Multi-modal Data Collection

This script converts old dVRK kinematic and time_syn data formats to the new standardized format.
It handles three distinct old format types and performs comprehensive data structure transformation
and timestamp reorganization.

Purpose:
- Convert old Format A (simple header) and Format B (complex header) to new standardized format
- Restructure kinematic JSON files (rename fields, add new fields, wrap in list)
- Reorganize time_syn JSON files (extract timestamps, create centralized structure)
- Preserve all data including setpoint_cp (moved to setpoint_data in new format)
- Handle missing timestamps with intelligent backfilling
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
import copy
import gc  # For explicit garbage collection in large-scale processing


# ============================================================================
# Configuration Dataclasses
# ============================================================================

class FormatType(Enum):
    """Old data format types"""
    FORMAT_A = "format_a"  # Simple header (sec, nsec only)
    FORMAT_B = "format_b"  # Complex header (per-topic timestamps)
    UNKNOWN = "unknown"


class ErrorMode(Enum):
    """Error handling modes"""
    STOP = "stop"      # Stop on first error
    SKIP = "skip"      # Skip problematic files and continue
    WARN = "warn"      # Continue with warnings


@dataclass
class DefaultValues:
    """Default values for new fields"""
    measured_cp_velocity: List[float]


@dataclass
class OutputFormat:
    """Output JSON formatting options"""
    json_indent: Optional[int]
    sort_keys: bool
    compact: bool


@dataclass
class ConvertConfig:
    """Configuration for data format conversion script"""
    workspace: str
    input_folder: str
    max_depth: int
    dry_run: bool
    test_mode: bool
    verbose: bool
    num_workers: int
    require_confirmation: bool
    validate_output: bool
    error_mode: str
    backup_originals: bool
    expected_arms: List[str]
    default_values: DefaultValues
    output_format: OutputFormat


# ============================================================================
# Utility Functions
# ============================================================================

def is_dataset_folder(folder_path: Path) -> bool:
    """
    Check if a folder is a valid dataset folder by verifying it contains required subfolders.

    A valid dataset folder must have:
    - kinematic/ folder with at least ECM subfolder
    - time_syn/ folder

    Args:
        folder_path: Path to check

    Returns:
        True if folder is a valid dataset folder, False otherwise (including on errors)
    """
    try:
        kinematic_path = folder_path / "kinematic"
        time_syn_path = folder_path / "time_syn"

        # Must have both kinematic and time_syn folders
        if not (kinematic_path.exists() and time_syn_path.exists()):
            return False

        # kinematic folder must have at least one arm subfolder (ECM)
        if not (kinematic_path / "ECM").exists():
            return False

        # time_syn must have at least one JSON file
        time_syn_files = list(time_syn_path.glob("*.json"))
        if not time_syn_files:
            return False

        return True

    except (PermissionError, OSError, Exception):
        # If we can't access the folder or its contents, skip it gracefully
        return False


def find_dataset_folders(root_path: Path, verbose: bool = True, max_depth: int = 5) -> List[Path]:
    """
    Find all valid dataset folders in the input directory using efficient iterative search.

    Optimized for large-scale data:
    - Uses iterative breadth-first search (not recursive)
    - Limits search depth to prevent deep recursion
    - Skips non-data directories early
    - Stops searching within dataset folders (they contain thousands of files)
    - Efficient for directories with millions of files
    - Gracefully skips folders that don't match dataset pattern or cause errors

    Args:
        root_path: Root directory to search
        verbose: Print detailed information
        max_depth: Maximum depth to search (default: 5, prevents infinite loops)

    Returns:
        List of paths to dataset folders (sorted by folder name as integer)

    Note:
        For typical structure like data/old/0/, data/old/3/, max_depth=3 is sufficient.
        Default max_depth=5 balances search capability with performance.
        Can be configured via config file's max_depth parameter.

        Folders that don't match the dataset pattern are silently skipped, allowing
        the search to continue with other folders without interruption.
    """
    dataset_folders = []

    if verbose:
        print(f"\nSearching for dataset folders in {root_path}...")
        print(f"  (Max search depth: {max_depth})")

    # Directories to skip (common non-data folders)
    skip_dirs = {
        'camera_calibration', 'hand_eye_calibration',
        '.git', '__pycache__', '.venv', 'venv', 'env', '.idea',
        # Skip backup folders created by this script
        '__backup__'
    }

    # Use iterative breadth-first search for efficiency
    # Format: (path, depth)
    # Depth counting: root_path is depth 0
    #   - depth 0: root_path (input_folder)
    #   - depth 1: root_path/subfolder1
    #   - depth 2: root_path/subfolder1/subfolder2
    #   - depth N: N levels of nested subfolders
    # max_depth=5 means we search up to 5 levels deep from root_path
    to_search = [(root_path, 0)]
    visited = set()  # Track visited directories to avoid duplicates

    while to_search:
        current_path, depth = to_search.pop(0)

        # Skip if already visited (handles symlinks)
        if current_path in visited:
            continue
        visited.add(current_path)

        # Stop if we've gone too deep
        # If max_depth=5, we process depths 0,1,2,3,4,5 and skip depth 6+
        if depth > max_depth:
            continue

        # Check if this is a dataset folder BEFORE recursing into it
        # This prevents searching thousands of files inside dataset folders
        # Gracefully skip folders that don't match the pattern or cause errors
        is_valid_dataset = False
        try:
            is_valid_dataset = is_dataset_folder(current_path)
        except Exception as e:
            # If validation fails, just skip adding this folder to results
            # but still continue searching its subfolders
            if verbose:
                print(f"  Warning: Cannot validate {current_path.name} ({e}), continuing search")

        if is_valid_dataset:
            dataset_folders.append(current_path)
            if verbose:
                print(f"  Found dataset: {current_path.name}")
            # Don't recurse into dataset folders (they contain thousands of data files)
            continue

        # List immediate subdirectories only (not files, for efficiency)
        try:
            subdirs = [d for d in current_path.iterdir() if d.is_dir()]
        except (PermissionError, OSError) as e:
            if verbose:
                print(f"  Warning: Cannot access {current_path}: {e}")
            continue

        # Add subdirectories to search queue
        # This ensures we keep searching deeper (subfolder of subfolder of ...)
        # until we reach max_depth or no more subfolders exist
        for subdir in subdirs:
            # Skip directories in skip list or starting with '.' or ending with '_backup'
            if (subdir.name in skip_dirs or
                subdir.name.startswith('.') or
                subdir.name.endswith('_backup') or
                subdir.name.endswith('_converted')):
                continue

            # Add to search queue with incremented depth
            # Will continue searching this subfolder and its subfolders up to max_depth
            to_search.append((subdir, depth + 1))

    # Sort by folder name (as integer if possible, otherwise alphabetically)
    def sort_key(path):
        try:
            return (0, int(path.name))  # Integer folders first
        except ValueError:
            return (1, path.name)  # Non-integer folders second

    dataset_folders.sort(key=sort_key)

    if verbose:
        print(f"  Total datasets found: {len(dataset_folders)}")
        if len(dataset_folders) > 100:
            print(f"  Large-scale dataset detected - memory optimizations enabled")

    return dataset_folders


def detect_format_type(kinematic_json: Dict) -> FormatType:
    """
    Detect the format type of a kinematic JSON file by inspecting the header.

    Format A (Simple): header with only "sec" and "nsec"
    Format B (Complex): header with additional fields like "header_img_left", "header_js_meas", etc.

    Args:
        kinematic_json: Loaded kinematic JSON data

    Returns:
        FormatType enum value
    """
    # Check if header exists
    if "header" not in kinematic_json:
        return FormatType.UNKNOWN

    header = kinematic_json["header"]

    # Check for required base fields
    if "sec" not in header or "nsec" not in header:
        return FormatType.UNKNOWN

    # Format A: Only sec and nsec (possibly with additional simple fields)
    # Format B: Has complex header fields like header_img_left, header_js_meas, etc.
    complex_header_markers = {
        "header_img_left", "header_img_right", "header_img_side",
        "header_js_meas", "header_js_set", "header_cp_set", "header_lcp"
    }

    # If any complex header marker is present, it's Format B
    if any(marker in header for marker in complex_header_markers):
        return FormatType.FORMAT_B

    # Otherwise, it's Format A (simple header)
    return FormatType.FORMAT_A


def parse_timestamp_string(timestamp_str: str) -> Dict[str, int]:
    """
    Parse timestamp string from old time_syn format to new dict format.

    Old format: "1754610345_860709712"
    New format: {"nsec": 860709712, "sec": 1754610345}

    Args:
        timestamp_str: Timestamp string in "sec_nsec" format

    Returns:
        Dictionary with "nsec" and "sec" keys (nsec first)
    """
    try:
        parts = timestamp_str.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")

        return {
            "nsec": int(parts[1]),
            "sec": int(parts[0])
        }
    except Exception as e:
        print(f"  Warning: Failed to parse timestamp '{timestamp_str}': {e}")
        return {"nsec": 0, "sec": 0}


def get_arm_subfolders(kinematic_path: Path, delete_empty_psm: bool = True) -> List[str]:
    """
    Get list of arm subfolder names in kinematic folder.

    Empty PSM folders (PSM1, PSM2, PSM3) will be deleted if delete_empty_psm=True.
    ECM folder is always kept even if empty (required).

    Args:
        kinematic_path: Path to kinematic folder
        delete_empty_psm: Whether to delete empty PSM folders

    Returns:
        List of arm names (e.g., ["ECM", "PSM1", "PSM2"])
    """
    arm_folders = []

    for subfolder in kinematic_path.iterdir():
        if subfolder.is_dir() and not subfolder.name.endswith("_converted"):
            # Check if this arm folder has JSON files
            json_files = list(subfolder.glob("*.json"))

            if json_files:
                arm_folders.append(subfolder.name)
            else:
                # Empty folder
                if subfolder.name.startswith("PSM") and delete_empty_psm:
                    # Delete empty PSM folders
                    try:
                        import shutil
                        shutil.rmtree(subfolder)
                        print(f"  Deleted empty PSM folder: {subfolder}")
                    except Exception as e:
                        print(f"  Warning: Failed to delete empty folder {subfolder}: {e}")
                elif subfolder.name == "ECM":
                    # Keep ECM even if empty (will warn later)
                    arm_folders.append(subfolder.name)

    return sorted(arm_folders)


def count_json_files(folder_path: Path) -> int:
    """
    Count JSON files in a folder.

    Args:
        folder_path: Path to folder

    Returns:
        Number of JSON files
    """
    if not folder_path.exists():
        return 0

    return len(list(folder_path.glob("*.json")))


# ============================================================================
# Kinematic Conversion Functions
# ============================================================================

def convert_kinematic_structure(old_data: Dict, default_values: DefaultValues) -> List[Dict]:
    """
    Convert old kinematic structure to new format.

    Changes:
    1. Wrap entire structure in a list
    2. Rename fields:
       - arm.local_cp → arm.measured_data.local_measured_cp
       - arm.measured_data.cp → arm.measured_data.measured_cp
       - arm.measured_data.js → arm.measured_data.measured_js
       - arm.setpoint_data.cp → arm.setpoint_data.setpoint_cp (PRESERVE!)
       - arm.setpoint_data.js → arm.setpoint_data.setpoint_js
    3. Add new fields:
       - arm.measured_data.measured_cv (initialized to zeros)
       - arm.measured_data.measured_cp.velocity (6D, initialized to zeros)
    4. Remove header

    Args:
        old_data: Old kinematic JSON data
        default_values: Default values configuration

    Returns:
        New kinematic structure (list with single element)
    """
    # Deep copy to avoid modifying original
    new_data = copy.deepcopy(old_data)

    # Extract arm data
    if "arm" not in new_data:
        raise ValueError("Missing 'arm' field in kinematic data")

    arm = new_data["arm"]

    # Create new arm structure
    new_arm = {
        "measured_data": {},
        "setpoint_data": {}
    }

    # ========== Process measured_data ==========

    # 1. Move local_cp to measured_data.local_measured_cp
    if "local_cp" in arm:
        new_arm["measured_data"]["local_measured_cp"] = arm["local_cp"]

    # 2. Rename measured_data.cp to measured_cp and add velocity if missing
    if "measured_data" in arm and "cp" in arm["measured_data"]:
        measured_cp = copy.deepcopy(arm["measured_data"]["cp"])
        # Add velocity field ONLY if it doesn't exist (ECM case)
        # PSM already has velocity field in old format, so preserve it
        if "velocity" not in measured_cp:
            measured_cp["velocity"] = default_values.measured_cp_velocity.copy()
        new_arm["measured_data"]["measured_cp"] = measured_cp

        # 3. Derive measured_cv from measured_cp.velocity
        # measured_cv is just a restructured version of velocity
        # velocity = [vx, vy, vz, wx, wy, wz]
        # measured_cv.linear = [vx, vy, vz]
        # measured_cv.angular = [wx, wy, wz]
        velocity = measured_cp["velocity"]
        new_arm["measured_data"]["measured_cv"] = {
            "linear": velocity[0:3],   # First 3 elements: linear velocity
            "angular": velocity[3:6]   # Last 3 elements: angular velocity
        }

    # 4. Rename measured_data.js to measured_js
    if "measured_data" in arm and "js" in arm["measured_data"]:
        new_arm["measured_data"]["measured_js"] = arm["measured_data"]["js"]

    # ========== Process setpoint_data ==========

    # 5. Rename setpoint_data.cp to setpoint_cp (PRESERVE THIS DATA!)
    if "setpoint_data" in arm and "cp" in arm["setpoint_data"]:
        new_arm["setpoint_data"]["setpoint_cp"] = arm["setpoint_data"]["cp"]

    # 6. Rename setpoint_data.js to setpoint_js
    if "setpoint_data" in arm and "js" in arm["setpoint_data"]:
        new_arm["setpoint_data"]["setpoint_js"] = arm["setpoint_data"]["js"]

    # ========== Process jaw data (PSM only) ==========

    # 7. Copy jaw data directly (structure is the same in old and new formats)
    if "jaw" in old_data:
        new_data_with_jaw = {
            "arm": new_arm,
            "jaw": old_data["jaw"]  # Jaw structure unchanged
        }
        # Also copy measured_frequency if present
        if "measured_frequency" in old_data:
            new_data_with_jaw["measured_frequency"] = old_data["measured_frequency"]

        # Wrap in list (new format requirement)
        return [new_data_with_jaw]

    # Wrap in list (new format requirement) - ECM case (no jaw)
    return [{"arm": new_arm}]


def convert_kinematic_file(
    src_path: Path,
    dst_path: Path,
    config: ConvertConfig
) -> Tuple[bool, Optional[str]]:
    """
    Convert a single kinematic JSON file from old to new format.

    Optimized for large-scale data:
    - Minimizes memory usage by processing one file at a time
    - Releases memory immediately after writing
    - Uses efficient JSON serialization

    Args:
        src_path: Source file path
        dst_path: Destination file path
        config: Conversion configuration

    Returns:
        Tuple of (success, error_message)
    """
    old_data = None
    new_data = None

    try:
        # Read old format (memory-efficient: read once, process, discard)
        with open(src_path, 'r') as f:
            old_data = json.load(f)

        # Convert structure (always do this to validate)
        new_data = convert_kinematic_structure(old_data, config.default_values)

        # Release old_data memory immediately
        del old_data

        # In dry run mode, skip writing
        if config.dry_run:
            return True, None

        # Note: Directory creation is now handled by caller for better performance
        # Write new format (efficient JSON dumping with separators for speed)
        with open(dst_path, 'w') as f:
            if config.output_format.compact:
                # Compact format for smaller files (no spaces)
                json.dump(new_data, f, sort_keys=config.output_format.sort_keys,
                         separators=(',', ':'))
            elif config.output_format.json_indent is not None:
                # Pretty format with indentation
                json.dump(new_data, f, indent=config.output_format.json_indent,
                         sort_keys=config.output_format.sort_keys)
            else:
                # Default format
                json.dump(new_data, f, sort_keys=config.output_format.sort_keys)

        return True, None

    except Exception as e:
        error_msg = f"Error converting {src_path}: {e}"
        return False, error_msg
    finally:
        # Explicitly release memory for large-scale processing
        del new_data


# ============================================================================
# Timestamp Extraction Functions
# ============================================================================

def extract_timestamps_format_a(header: Dict, arm_name: str) -> Dict[str, Any]:
    """
    Extract timestamps from Format A (simple header) kinematic file.

    Format A has only sec and nsec in header. Each arm uses its OWN header timestamp
    for its kinematic data:
    - ECM header → ECM kinematic timestamps
    - PSM1 header → PSM1 kinematic timestamps + jaw timestamps
    - PSM2 header → PSM2 kinematic timestamps + jaw timestamps
    - PSM3 header → PSM3 kinematic timestamps + jaw timestamps

    The timestamp is used for measured_js and backfilled to all other kinematic topics
    for that specific arm.

    Args:
        header: Header dictionary from this arm's kinematic JSON
        arm_name: Name of the arm (ECM, PSM1, PSM2, PSM3)

    Returns:
        Dictionary with timestamp structure for this arm in time_syn
    """
    # Extract base timestamp (used for measured_js and backfill)
    # IMPORTANT: nsec first, then sec
    base_stamp = {
        "nsec": header.get("nsec", 0),
        "sec": header.get("sec", 0)
    }

    # Build arm timestamp structure
    # Using dict.copy() instead of deepcopy for simple dicts (faster)
    arm_timestamps = {
        "header_cv_stamp": base_stamp.copy(),
        "measured_data": {
            "local_measured_cp_stamp": base_stamp.copy(),
            "measured_cp_stamp": base_stamp.copy(),
            "measured_cv_stamp": base_stamp.copy(),
            "measured_js_stamp": base_stamp.copy()  # Primary source
        },
        "reference_js_stamp": base_stamp.copy(),
        "setpoint_data": {
            "setpoint_cp_stamp": base_stamp.copy(),
            "setpoint_js_stamp": base_stamp.copy()
        }
    }

    # Add PSM-specific jaw timestamps (use PSM's own header timestamp)
    # Each PSM uses its own header timestamp for its jaw data
    if arm_name.startswith("PSM"):
        arm_timestamps["jaw"] = {
            "measured_stamp": base_stamp.copy(),
            "setpoint_stamp": base_stamp.copy()
        }

    # Note: measured_frequency is added separately in convert_time_syn_file

    return arm_timestamps


def extract_timestamps_format_b(header: Dict, arm_name: str) -> Dict[str, Any]:
    """
    Extract timestamps from Format B (complex header) kinematic file.

    Format B has per-topic timestamps in header fields like header_img_left,
    header_js_meas, etc. Extract and map them to new time_syn structure.

    Args:
        header: Header dictionary from kinematic JSON
        arm_name: Name of the arm (ECM, PSM1, etc.)

    Returns:
        Dictionary with timestamp structure for this arm in time_syn
    """
    # Extract base timestamp (measured_cp)
    # IMPORTANT: nsec first, then sec
    base_stamp = {
        "nsec": header.get("nsec", 0),
        "sec": header.get("sec", 0)
    }

    # Helper to get timestamp or use base as fallback
    def get_stamp(key: str, fallback: Dict = None) -> Dict[str, int]:
        if key in header and isinstance(header[key], dict):
            # IMPORTANT: nsec first, then sec
            return {
                "nsec": header[key].get("nsec", 0),
                "sec": header[key].get("sec", 0)
            }
        # Use dict.copy() for simple dicts (faster than deepcopy)
        return (fallback or base_stamp).copy()

    # Build arm timestamp structure with specific timestamps from header
    arm_timestamps = {
        "header_cv_stamp": get_stamp("header_cv", base_stamp),
        "measured_data": {
            "local_measured_cp_stamp": get_stamp("header_lcp", base_stamp),
            "measured_cp_stamp": base_stamp.copy(),  # header.sec + header.nsec
            "measured_cv_stamp": get_stamp("header_cv", base_stamp),
            "measured_js_stamp": get_stamp("header_js_meas", base_stamp)
        },
        "reference_js_stamp": get_stamp("header_js_meas", base_stamp),
        "setpoint_data": {
            "setpoint_cp_stamp": get_stamp("header_cp_set", base_stamp),
            "setpoint_js_stamp": get_stamp("header_js_set", base_stamp)
        }
    }

    # Add PSM-specific jaw timestamps if available
    if arm_name.startswith("PSM"):
        arm_timestamps["jaw"] = {
            "measured_stamp": get_stamp("header_jaw_meas", base_stamp),
            "setpoint_stamp": get_stamp("header_jaw_set", base_stamp)
        }

    return arm_timestamps


def extract_image_timestamps_format_a(headers: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    """
    Extract image timestamps from Format A kinematic headers.

    Format A uses ECM header timestamp (sec, nsec) as the image left timestamp.
    Right and side timestamps are backfilled with the same ECM timestamp.

    Args:
        headers: Dictionary mapping arm names to their headers (must contain ECM)

    Returns:
        Dictionary with image timestamps
    """
    # Get ECM header timestamp (IMPORTANT: nsec first, then sec)
    ecm_stamp = {"nsec": 0, "sec": 0}

    if "ECM" in headers:
        ecm_header = headers["ECM"]
        ecm_stamp = {
            "nsec": ecm_header.get("nsec", 0),
            "sec": ecm_header.get("sec", 0)
        }

    # Use ECM timestamp for image_left and backfill missing image timestamps
    # Use dict.copy() for simple dicts (faster than deepcopy)
    return {
        "image_left_stamp": ecm_stamp.copy(),
        "image_right_stamp": ecm_stamp.copy(),
        "side_image_1_stamp": ecm_stamp.copy()
    }


def extract_image_timestamps_format_b(headers: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    """
    Extract image timestamps from Format B kinematic headers.

    Format B has header_img_left, header_img_right, header_img_side in kinematic headers.
    Use the first available arm's header (typically ECM).

    Args:
        headers: Dictionary mapping arm names to their headers

    Returns:
        Dictionary with image timestamps
    """
    # Find first arm with image timestamps (typically ECM)
    image_stamps = {}

    # Try to get from any available arm
    for arm_name, header in headers.items():
        if "header_img_left" in header:
            # IMPORTANT: nsec first, then sec
            image_stamps["image_left_stamp"] = {
                "nsec": header["header_img_left"].get("nsec", 0),
                "sec": header["header_img_left"].get("sec", 0)
            }
        if "header_img_right" in header:
            image_stamps["image_right_stamp"] = {
                "nsec": header["header_img_right"].get("nsec", 0),
                "sec": header["header_img_right"].get("sec", 0)
            }
        if "header_img_side" in header:
            image_stamps["side_image_1_stamp"] = {
                "nsec": header["header_img_side"].get("nsec", 0),
                "sec": header["header_img_side"].get("sec", 0)
            }

        # If we found image timestamps, break
        if image_stamps:
            break

    # Fallback: if no image timestamps found, use zeros
    if "image_left_stamp" not in image_stamps:
        image_stamps["image_left_stamp"] = {"nsec": 0, "sec": 0}
    if "image_right_stamp" not in image_stamps:
        image_stamps["image_right_stamp"] = image_stamps["image_left_stamp"].copy()
    if "side_image_1_stamp" not in image_stamps:
        image_stamps["side_image_1_stamp"] = image_stamps["image_left_stamp"].copy()

    return image_stamps


def convert_time_syn_file(
    time_syn_src: Path,
    kinematic_folder: Path,
    dst_path: Path,
    file_index: int,
    format_type: FormatType,
    config: ConvertConfig
) -> Tuple[bool, Optional[str]]:
    """
    Convert a single time_syn JSON file from old to new format.

    This involves:
    1. Reading kinematic files for this index to extract headers and data
    2. Extracting timestamps based on format type (from kinematic headers only)
    3. Building new time_syn structure with Kinematics_set_1
    4. Adding PSM-specific fields (measured_frequency)
    5. Writing new time_syn file

    Note: For Format A, old time_syn file is NOT read (not needed - all timestamps from kinematic headers).
          For Format B, old time_syn file is also NOT read (uses header_img_xxx from kinematic headers).

    Args:
        time_syn_src: Source time_syn file path (not used, kept for consistency)
        kinematic_folder: Path to kinematic folder (to read arm headers)
        dst_path: Destination time_syn file path
        file_index: File index (for frame number)
        format_type: Detected format type
        config: Conversion configuration

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Note: We do NOT read the old time_syn file - all timestamps come from kinematic headers
        # This is a performance optimization (saves I/O for every time_syn file)

        # Get list of arms from kinematic folder
        arm_names = get_arm_subfolders(kinematic_folder)

        if not arm_names:
            return False, f"No arm subfolders found in {kinematic_folder}"

        # Read kinematic files for all arms (need full data, not just headers)
        kinematic_headers = {}
        kinematic_data_all = {}
        for arm_name in arm_names:
            kinematic_file = kinematic_folder / arm_name / f"{file_index}.json"

            if not kinematic_file.exists():
                if config.verbose:
                    print(f"  Warning: Kinematic file not found for {arm_name}: {kinematic_file}")
                continue

            try:
                with open(kinematic_file, 'r') as f:
                    kinematic_data = json.load(f)

                kinematic_data_all[arm_name] = kinematic_data

                if "header" in kinematic_data:
                    kinematic_headers[arm_name] = kinematic_data["header"]
                else:
                    if config.verbose:
                        print(f"  Warning: No header in kinematic file {kinematic_file}")
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: Failed to read {kinematic_file}: {e}")

        # Build new time_syn structure
        new_time_syn = {
            "Kinematics_set_1": {},
            "frame": file_index
        }

        # Extract timestamps for each arm based on format type
        for arm_name, header in kinematic_headers.items():
            if format_type == FormatType.FORMAT_A:
                arm_timestamps = extract_timestamps_format_a(header, arm_name)
            elif format_type == FormatType.FORMAT_B:
                arm_timestamps = extract_timestamps_format_b(header, arm_name)
            else:
                return False, f"Unknown format type: {format_type}"

            # Add measured_frequency if this is a PSM and it exists in the data
            if arm_name.startswith("PSM") and arm_name in kinematic_data_all:
                if "measured_frequency" in kinematic_data_all[arm_name]:
                    arm_timestamps["measured_frequency"] = kinematic_data_all[arm_name]["measured_frequency"]

            new_time_syn["Kinematics_set_1"][arm_name] = arm_timestamps

        # Extract image timestamps based on format type
        if format_type == FormatType.FORMAT_A:
            # Format A: Use ECM timestamp for all image timestamps
            image_stamps = extract_image_timestamps_format_a(kinematic_headers)
        else:  # FORMAT_B
            # Format B: Use header_img_xxx timestamps from kinematic headers
            image_stamps = extract_image_timestamps_format_b(kinematic_headers)

        # Add image timestamps to root level
        new_time_syn.update(image_stamps)

        # In dry run mode, skip writing
        if config.dry_run:
            return True, None

        # Note: Directory creation is now handled by caller for better performance
        # Write new time_syn file
        with open(dst_path, 'w') as f:
            if config.output_format.json_indent is not None:
                json.dump(new_time_syn, f, indent=config.output_format.json_indent,
                         sort_keys=config.output_format.sort_keys)
            else:
                json.dump(new_time_syn, f, sort_keys=config.output_format.sort_keys)

        return True, None

    except Exception as e:
        error_msg = f"Error converting time_syn {time_syn_src}: {e}"
        return False, error_msg


# ============================================================================
# Dataset Conversion Function
# ============================================================================

def convert_dataset(
    dataset_path: Path,
    config: ConvertConfig
) -> Tuple[str, Dict[str, Any]]:
    """
    Convert a single dataset folder from old to new format.

    Args:
        dataset_path: Path to dataset folder
        config: Conversion configuration

    Returns:
        Tuple of (status, stats_dict)
        status: "success", "partial", "failed"
        stats_dict: Dictionary with conversion statistics
    """
    stats = {
        "dataset_name": dataset_path.name,
        "format_type": None,
        "arms_found": [],
        "kinematic_converted": 0,
        "kinematic_failed": 0,
        "time_syn_converted": 0,
        "time_syn_failed": 0,
        "errors": [],
        "backup_created": False
    }

    kinematic_folder = dataset_path / "kinematic"
    time_syn_folder = dataset_path / "time_syn"

    if config.verbose:
        print(f"\n{'='*70}")
        print(f"Converting dataset: {dataset_path.name}")
        if config.dry_run:
            print(f"  (DRY RUN - no files will be written)")
        print(f"{'='*70}")

    # Create backup if not in test mode and backup_originals is enabled
    # Skip backup in dry run mode
    if not config.dry_run and not config.test_mode and config.backup_originals:
        backup_path = dataset_path.parent / f"{dataset_path.name}_backup"

        if backup_path.exists():
            if config.verbose:
                print(f"  Backup already exists: {backup_path}")
                print(f"  Skipping backup creation to avoid overwriting")
        else:
            try:
                if config.verbose:
                    print(f"  Creating backup: {backup_path}")

                shutil.copytree(dataset_path, backup_path)
                stats["backup_created"] = True

                if config.verbose:
                    print(f"  ✓ Backup created successfully")
            except Exception as e:
                error_msg = f"Failed to create backup: {e}"
                stats["errors"].append(error_msg)
                if config.verbose:
                    print(f"  ERROR: {error_msg}")
                if config.error_mode == "stop":
                    return "failed", stats

    # Get list of arms
    arm_names = get_arm_subfolders(kinematic_folder)
    stats["arms_found"] = arm_names

    if not arm_names:
        error_msg = f"No arm subfolders found in {kinematic_folder}"
        stats["errors"].append(error_msg)
        if config.verbose:
            print(f"  ERROR: {error_msg}")
        return "failed", stats

    if config.verbose:
        print(f"  Arms found: {', '.join(arm_names)}")

    # Detect format type by reading first kinematic file of first arm
    format_type = FormatType.UNKNOWN
    first_arm = arm_names[0]
    first_kinematic_file = kinematic_folder / first_arm / "0.json"

    try:
        with open(first_kinematic_file, 'r') as f:
            sample_data = json.load(f)
        format_type = detect_format_type(sample_data)
        stats["format_type"] = format_type.value

        if config.verbose:
            print(f"  Detected format: {format_type.value.upper()}")

        if format_type == FormatType.UNKNOWN:
            error_msg = "Could not detect format type (unrecognized header structure)"
            stats["errors"].append(error_msg)
            if config.verbose:
                print(f"  ERROR: {error_msg}")
            return "failed", stats

    except Exception as e:
        error_msg = f"Failed to detect format: {e}"
        stats["errors"].append(error_msg)
        if config.verbose:
            print(f"  ERROR: {error_msg}")
        return "failed", stats

    # IMPORTANT: Convert time_syn files FIRST, before kinematic files
    # Reason: time_syn conversion reads kinematic headers from old format
    # If we convert kinematic files first in test_mode=False (in-place),
    # the old headers are overwritten and time_syn conversion will fail

    # Convert time_syn files
    if config.verbose:
        print(f"\n  Converting time_syn files...")

    # Determine output folder for time_syn
    if config.test_mode:
        time_syn_output = dataset_path / "time_syn_converted"
    else:
        time_syn_output = time_syn_folder

    # Pre-create output directory
    if not config.dry_run and not time_syn_output.exists():
        time_syn_output.mkdir(parents=True, exist_ok=True)

    # Get all time_syn JSON files and sort by numeric stem (optimized)
    time_syn_files = [(int(p.stem), p) for p in time_syn_folder.glob("*.json")]
    time_syn_files.sort(key=lambda x: x[0])
    total_files = len(time_syn_files)

    if config.verbose:
        print(f"    Converting {total_files} files...")

    # Convert each time_syn file (memory-efficient processing)
    for file_index, src_file in time_syn_files:
        dst_file = time_syn_output / src_file.name

        success, error = convert_time_syn_file(
            src_file, kinematic_folder, dst_file, file_index, format_type, config
        )

        if success:
            stats["time_syn_converted"] += 1
        else:
            stats["time_syn_failed"] += 1
            stats["errors"].append(f"time_syn/{src_file.name}: {error}")

            if config.error_mode == "stop":
                return "failed", stats
            elif config.verbose:
                print(f"    ERROR: {error}")

    if config.verbose:
        print(f"    time_syn: {stats['time_syn_converted']} converted, {stats['time_syn_failed']} failed")

    # Convert kinematic files for each arm
    # This must come AFTER time_syn conversion to preserve old headers
    if config.verbose:
        print(f"\n  Converting kinematic files...")

    for arm_name in arm_names:
        arm_folder = kinematic_folder / arm_name

        # Determine output folder based on test_mode
        if config.test_mode:
            output_folder = kinematic_folder / f"{arm_name}_converted"
        else:
            output_folder = arm_folder

        # Pre-create output directory to avoid repeated existence checks in the loop
        if not config.dry_run and not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        # Get all JSON files and sort by numeric stem (optimized)
        # Collect files into list first for faster iteration
        json_files = [(int(p.stem), p) for p in arm_folder.glob("*.json")]
        json_files.sort(key=lambda x: x[0])  # Sort by numeric index
        json_files = [p for _, p in json_files]  # Extract paths
        total_files = len(json_files)

        if config.verbose:
            print(f"    {arm_name}: Converting {total_files} files...")

        # Convert each file (memory-efficient: one at a time)
        arm_success = 0
        arm_failed = 0

        for src_file in json_files:
            dst_file = output_folder / src_file.name
            success, error = convert_kinematic_file(src_file, dst_file, config)

            if success:
                arm_success += 1
                stats["kinematic_converted"] += 1
            else:
                arm_failed += 1
                stats["kinematic_failed"] += 1
                stats["errors"].append(f"{arm_name}/{src_file.name}: {error}")

                if config.error_mode == "stop":
                    return "failed", stats
                elif config.verbose:
                    print(f"      ERROR: {error}")

        if config.verbose:
            print(f"    {arm_name}: {arm_success} converted, {arm_failed} failed")

    # Determine overall status
    if stats["kinematic_failed"] == 0 and stats["time_syn_failed"] == 0:
        status = "success"
    elif stats["kinematic_converted"] > 0 or stats["time_syn_converted"] > 0:
        status = "partial"
    else:
        status = "failed"

    if config.verbose:
        print(f"\n  Dataset conversion {status.upper()}")

    return status, stats


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_all_datasets(config: ConvertConfig) -> Dict[str, Any]:
    """
    Main conversion function. Processes all datasets in input folder.

    Args:
        config: Conversion configuration

    Returns:
        Dictionary with overall conversion statistics
    """
    input_path = Path(config.input_folder)

    # Validate input folder
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Find all dataset folders (optimized for large-scale data)
    dataset_folders = find_dataset_folders(input_path, config.verbose, config.max_depth)

    if not dataset_folders:
        print(f"\nNo dataset folders found in {input_path}")
        print("A dataset folder must contain kinematic/ and time_syn/ subfolders")
        return {"total_datasets": 0, "success": 0, "partial": 0, "failed": 0}

    print(f"\nFound {len(dataset_folders)} dataset(s) to convert")

    if config.dry_run:
        print(f"DRY RUN MODE: No files will be written")
        print("  (Validation only - checking conversion logic)")
    else:
        print(f"Test mode: {config.test_mode}")
        if config.test_mode:
            print("  (Converted files will be saved to *_converted folders)")
        else:
            print("  (WARNING: Original files will be replaced!)")
            if config.backup_originals:
                print("  (Backups will be created as <dataset_name>_backup)")

    # Ask for confirmation if required (skip in dry run mode)
    if config.require_confirmation and not config.dry_run:
        print(f"\n{'='*70}")
        if not config.test_mode:
            print("WARNING: This will modify your original data files!")
        print(f"{'='*70}")
        response = input("Press ENTER to continue or Ctrl+C to cancel: ")

    # Process each dataset
    overall_stats = {
        "total_datasets": len(dataset_folders),
        "success": 0,
        "partial": 0,
        "failed": 0,
        "total_kinematic_converted": 0,
        "total_kinematic_failed": 0,
        "total_time_syn_converted": 0,
        "total_time_syn_failed": 0,
        "backups_created": 0,
        "dataset_stats": []
    }

    print(f"\n{'='*70}")
    print("Starting data conversion...")
    print(f"{'='*70}")

    # Use tqdm for progress (disable in verbose mode to avoid output conflicts)
    for idx, dataset_path in enumerate(tqdm(dataset_folders, desc="Converting datasets", disable=config.verbose)):
        status, stats = convert_dataset(dataset_path, config)

        # Update overall stats
        overall_stats[status] += 1
        overall_stats["total_kinematic_converted"] += stats["kinematic_converted"]
        overall_stats["total_kinematic_failed"] += stats["kinematic_failed"]
        overall_stats["total_time_syn_converted"] += stats["time_syn_converted"]
        overall_stats["total_time_syn_failed"] += stats["time_syn_failed"]
        if stats.get("backup_created", False):
            overall_stats["backups_created"] += 1
        overall_stats["dataset_stats"].append(stats)

        # Garbage collection after each dataset completes
        # This ensures memory is released between datasets for large-scale processing
        gc.collect()

    # Print summary
    print(f"\n{'='*70}")
    if config.dry_run:
        print(f"DRY RUN validation complete!")
    else:
        print(f"Data conversion complete!")
    print(f"{'='*70}")
    print(f"Total datasets: {overall_stats['total_datasets']}")
    print(f"")
    print(f"✓ Successful: {overall_stats['success']} datasets")
    print(f"⊘ Partial: {overall_stats['partial']} datasets (some files failed)")
    print(f"✗ Failed: {overall_stats['failed']} datasets")
    print(f"")
    if config.dry_run:
        print(f"Kinematic files (would be converted):")
    else:
        print(f"Kinematic files:")
    print(f"  Converted: {overall_stats['total_kinematic_converted']}")
    print(f"  Failed: {overall_stats['total_kinematic_failed']}")
    print(f"")
    if config.dry_run:
        print(f"time_syn files (would be converted):")
    else:
        print(f"time_syn files:")
    print(f"  Converted: {overall_stats['total_time_syn_converted']}")
    print(f"  Failed: {overall_stats['total_time_syn_failed']}")
    print(f"")
    if not config.dry_run and not config.test_mode and config.backup_originals:
        print(f"Backups created: {overall_stats['backups_created']}/{overall_stats['total_datasets']}")
        if overall_stats['backups_created'] > 0:
            print(f"  (Backup folders: <dataset_name>_backup)")
    print(f"")
    print(f"Input folder: {input_path}")
    if config.dry_run:
        print(f"Mode: DRY RUN (no files written)")
    else:
        print(f"Test mode: {config.test_mode}")
    print(f"{'='*70}")

    if config.dry_run:
        print(f"\n💡 DRY RUN completed successfully!")
        print(f"   All files validated. No actual changes were made.")
        print(f"   Set dry_run=false to perform actual conversion.")

    # Print error summary if there were any
    all_errors = []
    for stats in overall_stats["dataset_stats"]:
        if stats["errors"]:
            all_errors.append((stats["dataset_name"], stats["errors"]))

    if all_errors:
        print(f"\n{'='*70}")
        print(f"Error Summary ({len(all_errors)} datasets with errors)")
        print(f"{'='*70}")
        for dataset_name, errors in all_errors:
            print(f"\nDataset: {dataset_name}")
            for error in errors[:5]:  # Limit to first 5 errors per dataset
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

    return overall_stats


# ============================================================================
# Configuration and Main Entry Point
# ============================================================================

# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="convert_config", node=ConvertConfig)

# Set config path
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_convert_old_data"
)
def main(cfg: DictConfig):
    """
    Main entry point for the data format conversion script.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf)
    """
    print("="*70)
    print("dVRK Old Data Format Converter")
    print("="*70)
    print(f"Workspace: {cfg.workspace}")
    print(f"Input folder: {cfg.input_folder}")
    print(f"Max search depth: {cfg.max_depth}")
    print(f"Dry run: {cfg.dry_run}")
    if not cfg.dry_run:
        print(f"Test mode: {cfg.test_mode}")
    print(f"Verbose: {cfg.verbose}")
    print(f"Error mode: {cfg.error_mode}")
    print("="*70)

    # Convert DictConfig to ConvertConfig dataclass
    convert_config = ConvertConfig(
        workspace=cfg.workspace,
        input_folder=cfg.input_folder,
        max_depth=cfg.max_depth,
        dry_run=cfg.dry_run,
        test_mode=cfg.test_mode,
        verbose=cfg.verbose,
        num_workers=cfg.num_workers,
        require_confirmation=cfg.require_confirmation,
        validate_output=cfg.validate_output,
        error_mode=cfg.error_mode,
        backup_originals=cfg.backup_originals,
        expected_arms=list(cfg.expected_arms),
        default_values=DefaultValues(
            measured_cp_velocity=list(cfg.default_values.measured_cp_velocity)
        ),
        output_format=OutputFormat(
            json_indent=cfg.output_format.json_indent,
            sort_keys=cfg.output_format.sort_keys,
            compact=cfg.output_format.compact
        )
    )

    # Run the conversion
    try:
        stats = convert_all_datasets(convert_config)

        if stats["failed"] > 0 and stats["success"] == 0 and stats["partial"] == 0:
            print(f"\n✗ All conversions failed")
            sys.exit(1)
        elif stats["failed"] > 0 or stats["partial"] > 0:
            print(f"\n⚠ Conversion completed with some errors")
            sys.exit(0)
        else:
            print(f"\n✓ All conversions successful")
            sys.exit(0)

    except Exception as e:
        print(f"\n✗ Fatal error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
