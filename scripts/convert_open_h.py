"""
Convert dVRK Multi-modal Data to LeRobot v2.1 Format

This script converts dVRK surgical robot data from the custom format to the
LeRobot v2.1 dataset format compatible with lerobot==0.3.3.

The script supports two modes:
1. "single": Convert a single dataset folder
2. "recursive": Recursively find and convert all dataset folders

Input structure (per dataset folder):
  <dataset_folder>/
    image/
      left/, right/, side1/  # PNG images
    kinematic/
      ECM/, PSM1/, PSM2/, PSM3/  # JSON files
    time_syn/  # Time synchronization JSON files
    annotation/
      phase/, step/  # Annotation JSON files
    meta_data.json
    camera_calibration/  # Camera calibration YAML files
    hand_eye_calibration/  # Hand-eye calibration JSON files

Output structure (LeRobot v2.1):
  <output_folder>/
    <phase_name>/
      data/
        case-xxx/  # Chunk formatted as case-000, case-001, etc. (uses episode_chunk in templates)
          episode_xxxxxx.parquet  # Episode index formatted as episode_000000, episode_000001, etc.
      videos/
        case-xxx/  # Chunk formatted as case-000, case-001, etc. (uses episode_chunk in templates)
          observation.images.xxx/
            episode_xxxxxx.mp4  # Episode index formatted as episode_000000, episode_000001, etc.
      meta/
        info.json
        episodes.jsonl
        tasks.jsonl
        episodes_stats.jsonl
    calibrations/
      <phase_name>/
        case-xxx/  # Chunk formatted as case-000, case-001, etc.
          camera_calibration/
          hand_eye_calibration/
    total_time.json
"""

import os
import sys
import gc
import json
import shutil
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
import numpy as np
import pandas as pd

# Configure logging - provides better debugging and production monitoring
# Default level is INFO; can be changed via config or environment variable
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure warnings - suppress specific warnings that are expected during normal operation
# but keep important warnings visible for debugging and error detection
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyarrow")  # Suppress pyarrow internal warnings

# Try to import optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("opencv-python not found. Video conversion will not be available.")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logger.warning("pyarrow not found. Parquet file creation will not be available.")


@dataclass
class ConversionConfig:
    """Configuration for data conversion"""
    workspace: str
    mode: str  # "single" or "recursive"
    data_folder: str
    output_folder: str
    task_description_folder: str
    start_idx: int
    fps: int
    get_total_time: bool
    video_encoding: Dict[str, Any] = field(default_factory=dict)
    parquet_settings: Dict[str, Any] = field(default_factory=dict)
    processing: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    robot: Dict[str, Any] = field(default_factory=dict)
    video_name_mapping: Dict[str, str] = field(default_factory=dict)


def validate_dataset_folder(folder: Path) -> bool:
    """
    Validate that a folder contains all required subfolders/files for a valid dataset.

    Args:
        folder: Path to the potential dataset folder

    Returns:
        True if folder has valid dataset structure, False otherwise

    Required structure:
        - image/ (with subfolders containing PNG files)
        - kinematic/ (with subfolders for ECM, PSMx)
        - time_syn/ (with JSON files)
        - annotation/ (with phase/ and step/ subfolders)
        - meta_data.json
        - camera_calibration/ (with YAML files)
        - hand_eye_calibration/ (with JSON files)
    """
    required_folders = ['image', 'kinematic', 'time_syn', 'annotation',
                       'camera_calibration', 'hand_eye_calibration']
    required_files = ['meta_data.json']

    # Check required folders
    for folder_name in required_folders:
        if not (folder / folder_name).exists():
            return False

    # Check required files
    for file_name in required_files:
        if not (folder / file_name).exists():
            return False

    # Check annotation subfolders
    if not (folder / 'annotation' / 'phase').exists():
        return False
    if not (folder / 'annotation' / 'step').exists():
        return False

    # Check that image folder has at least one subfolder with PNG files
    image_folder = folder / 'image'
    has_images = False
    for subfolder in image_folder.iterdir():
        if subfolder.is_dir():
            png_files = list(subfolder.glob('*.png'))
            if len(png_files) > 0:
                has_images = True
                break

    return has_images


def find_dataset_folders(root_path: Path, max_depth: int = 10) -> List[Path]:
    """
    Iteratively find all folders that contain valid dataset structures.

    Uses breadth-first search (iterative) instead of recursion to avoid stack overflow
    on very deep folder structures. This is more efficient and robust for large-scale
    data processing.

    Args:
        root_path: Root directory to search from
        max_depth: Maximum search depth (default: 10)

    Returns:
        List of Path objects to valid dataset folders, sorted numerically by folder name

    Note:
        Uses validate_dataset_folder() to check if a folder is a valid dataset.
        Stops searching deeper once a valid dataset folder is found.

        Sorting follows numeric sequence (0, 1, 2, ..., 9, 10, 11, ...) rather than
        alphabetical order (which would give 0, 1, 10, 11, ..., 2, 20, ...).
        This ensures consecutive case indexing when processing multiple datasets.

    Performance:
        - Uses iterative BFS instead of recursive DFS for memory efficiency
        - Skips known non-data directories (calibration, git, cache, etc.)
        - Early termination: stops descending into valid dataset folders
    """
    dataset_folders = []

    # Directories to skip during search (calibration, git, python cache, etc.)
    # These directories never contain dataset structures, so skipping them saves time
    skip_dirs = {
        '.git', '__pycache__', 'camera_calibration', 'hand_eye_calibration',
        'rectify_resize', 'preprocess', '.venv', 'venv', 'env', '.idea',
        'outputs', 'logs', 'build', 'dist', '__MACOSX'
    }

    # Use iterative breadth-first search instead of recursion
    # This prevents stack overflow and is more memory-efficient
    to_search = [(root_path, 0)]  # List of (path, depth) tuples

    while to_search:
        current_path, depth = to_search.pop(0)

        # Stop if we've gone too deep
        if depth > max_depth:
            continue

        # Check if current path is a valid dataset
        if validate_dataset_folder(current_path):
            dataset_folders.append(current_path)
            # Don't recurse into dataset folders (they contain data, not more datasets)
            continue

        # Otherwise, search subdirectories
        try:
            subdirs = [d for d in current_path.iterdir() if d.is_dir()]
        except (PermissionError, OSError) as e:
            # Skip directories we can't access
            continue

        # Add subdirectories to search queue (excluding skip list)
        for subdir in subdirs:
            # Skip directories in skip list or starting with '.'
            if subdir.name in skip_dirs or subdir.name.startswith('.'):
                continue
            to_search.append((subdir, depth + 1))

    # Sort numerically by folder name to ensure proper sequence (0, 1, 2, ..., 10, 11, ...)
    # This handles cases where folder names are numeric indices (e.g., 0, 1, 2, etc.)
    # Falls back to string sorting for non-numeric folder names
    def numeric_sort_key(path: Path) -> tuple:
        """
        Extract sorting key: (parent_path, numeric_value or string_name)

        This ensures:
        1. Folders are grouped by parent path first
        2. Within each parent, folders are sorted numerically if names are numbers
        3. Non-numeric names fall back to string comparison

        Example correct ordering:
          raw/0, raw/1, raw/2, ..., raw/9, raw/10, raw/11, ...
        Instead of alphabetical:
          raw/0, raw/1, raw/10, raw/11, ..., raw/2, raw/20, ...
        """
        try:
            # Try to parse folder name as integer for numeric sorting
            num_value = int(path.name)
            # Return parent path and numeric value (numeric sorts before strings)
            return (str(path.parent), 0, num_value, "")
        except ValueError:
            # Not a numeric folder, sort by string name
            # Return parent path and string value (strings sort after numbers)
            return (str(path.parent), 1, 0, path.name)

    dataset_folders.sort(key=numeric_sort_key)

    return dataset_folders


def load_json_file(file_path: Path) -> Optional[Dict]:
    """
    Load JSON file safely with proper error handling and logging.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with JSON contents, or None if error

    Note:
        Logs warnings for file not found or JSON decode errors.
        Logs debug information for successful loads (useful for troubleshooting).
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        return None


def load_description_mappings(task_desc_folder: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load idx-to-text mapping files for phase, step, and user info.

    Args:
        task_desc_folder: Path to folder containing description JSON files

    Returns:
        Tuple of (phase_desc, step_desc, user_info) dictionaries
    """
    phase_desc = load_json_file(task_desc_folder / 'phase_description.json') or {}
    step_desc = load_json_file(task_desc_folder / 'step_description.json') or {}
    user_info = load_json_file(task_desc_folder / 'user_id_info.json') or {}

    return phase_desc, step_desc, user_info


def get_next_case_index(output_folder: Path, phase_name: str = None) -> int:
    """
    Find the next available case index in the output folder.

    The output structure uses case-xxx format (e.g., case-000, case-001, case-002).
    This function scans existing case folders and returns the next consecutive index.

    Args:
        output_folder: Root output folder
        phase_name: Deprecated, kept for compatibility but ignored

    Returns:
        Next available case index (0 if no cases exist)

    Note:
        Parses folder names matching the pattern 'case-NNN' where NNN is a 3-digit
        zero-padded number. Returns max_existing + 1 to ensure consecutive numbering.
    """
    # All cases are now directly under output_folder/data/ (no phase_name level)
    data_folder = output_folder / 'data'

    if not data_folder.exists():
        return 0

    max_idx = -1
    for item in data_folder.iterdir():
        if item.is_dir():
            # Parse case-xxx format (e.g., case-000, case-001, case-002)
            if item.name.startswith('case-'):
                try:
                    # Extract numeric part after 'case-' prefix
                    idx_str = item.name[5:]  # Remove 'case-' prefix
                    idx = int(idx_str)
                    max_idx = max(max_idx, idx)
                except ValueError:
                    # Skip folders that don't match the expected format
                    continue

    return max_idx + 1


def read_kinematic_data(kinematic_folder: Path, frame_id: int, arm_name: str) -> Optional[Dict]:
    """
    Read kinematic JSON file for a specific frame and arm.

    Args:
        kinematic_folder: Path to kinematic folder
        frame_id: Frame ID number
        arm_name: Name of the arm (ECM, PSM1, PSM2, PSM3)

    Returns:
        Dictionary with kinematic data, or None if not found
    """
    json_file = kinematic_folder / arm_name / f"{frame_id}.json"

    if not json_file.exists():
        return None

    data = load_json_file(json_file)
    if data is None or len(data) == 0:
        return None

    # The file contains a list with one element
    return data[0] if isinstance(data, list) else data


def read_time_sync_data(time_syn_folder: Path, frame_id: int) -> Optional[Dict]:
    """
    Read time synchronization JSON file for a specific frame.

    Args:
        time_syn_folder: Path to time_syn folder
        frame_id: Frame ID number

    Returns:
        Dictionary with time sync data, or None if not found
    """
    json_file = time_syn_folder / f"{frame_id}.json"
    return load_json_file(json_file) if json_file.exists() else None


def read_annotation_data(annotation_folder: Path, frame_id: int,
                        annotation_type: str) -> Optional[Dict]:
    """
    Read annotation JSON file for a specific frame.

    Args:
        annotation_folder: Path to annotation folder
        frame_id: Frame ID number
        annotation_type: Type of annotation ("phase" or "step")

    Returns:
        Dictionary with annotation data, or None if not found
    """
    json_file = annotation_folder / annotation_type / f"{frame_id}.json"
    data = load_json_file(json_file)

    if data is None:
        return None

    # Extract the actual phase/step value
    return data.get(annotation_type)


def calculate_time_tolerance_ms(time_sync_data: Dict, base_timestamp: Dict) -> float:
    """
    Calculate maximum time difference between timestamps in milliseconds.

    Args:
        time_sync_data: Time synchronization data dictionary
        base_timestamp: Base timestamp dict with 'sec' and 'nsec'

    Returns:
        Maximum time difference in milliseconds
    """
    base_time_s = base_timestamp['sec'] + base_timestamp['nsec'] / 1e9
    max_diff = 0.0

    # Recursively find all timestamps ending with '_stamp'
    def find_timestamps(data, prefix=''):
        nonlocal max_diff
        if isinstance(data, dict):
            for key, value in data.items():
                if key.endswith('_stamp') and isinstance(value, dict):
                    if 'sec' in value and 'nsec' in value:
                        curr_time_s = value['sec'] + value['nsec'] / 1e9
                        diff_s = abs(curr_time_s - base_time_s)
                        max_diff = max(max_diff, diff_s)
                elif isinstance(value, dict):
                    find_timestamps(value, f"{prefix}{key}.")

    find_timestamps(time_sync_data)

    return max_diff * 1000  # Convert to milliseconds


def group_frames_into_episodes(annotation_folder: Path,
                               phase_idx: str) -> List[List[int]]:
    """
    Group frame IDs into episodes based on phase and step annotations.

    Frames belong to the same episode if they have:
    1. Same phase index
    2. Same step index
    3. Consecutive frame IDs

    Args:
        annotation_folder: Path to annotation folder
        phase_idx: Phase index string to filter

    Returns:
        List of episode frame lists, where each episode is a list of frame IDs

    Performance optimizations for large-scale data:
        - Uses os.scandir() instead of glob() for faster directory traversal
        - Processes files incrementally to reduce peak memory usage
        - Reads step info lazily during grouping phase
    """
    phase_folder = annotation_folder / 'phase'
    step_folder = annotation_folder / 'step'

    # Use os.scandir for faster directory listing (avoids creating full Path objects)
    # This is more memory-efficient than glob() for large directories
    phase_frames = []
    try:
        # Collect frame IDs that match the target phase
        # Using scandir is faster than glob for large directories
        with os.scandir(phase_folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.json'):
                    try:
                        frame_id = int(entry.name[:-5])  # Remove .json suffix
                        # Read and check phase - use fast JSON loading
                        data = load_json_file(Path(entry.path))
                        if data and data.get('phase') == phase_idx:
                            phase_frames.append(frame_id)
                    except (ValueError, TypeError):
                        # Skip files with non-numeric names
                        continue
    except OSError as e:
        logger.warning(f"Error scanning phase folder {phase_folder}: {e}")
        return []

    if not phase_frames:
        return []

    # Sort frame IDs for sequential processing
    # Using Python's built-in sort which is efficient for large lists
    phase_frames.sort()

    # Group into episodes: same step + consecutive frames
    # Read step info lazily during iteration to reduce memory for large datasets
    episodes = []
    current_episode = []
    current_step = None
    prev_frame = None

    for frame_id in phase_frames:
        # Read step info lazily (only when needed)
        step_file = step_folder / f"{frame_id}.json"
        step_data = load_json_file(step_file)
        step_idx = step_data.get('step') if step_data else None

        # Start new episode if step changes or frame not consecutive
        if current_step != step_idx or (prev_frame is not None and frame_id != prev_frame + 1):
            if current_episode:
                episodes.append(current_episode)
            current_episode = [frame_id]
            current_step = step_idx
        else:
            current_episode.append(frame_id)

        prev_frame = frame_id

    # Add last episode
    if current_episode:
        episodes.append(current_episode)

    return episodes


def extract_observation_state(kinematic_data: Dict, arm_name: str,
                              is_psm: bool = True) -> Dict[str, List[float]]:
    """
    Extract observation state data from kinematic JSON for one arm.

    Args:
        kinematic_data: Kinematic data dictionary
        arm_name: Name of the arm (for dictionary keys)
        is_psm: True if PSM arm, False if ECM

    Returns:
        Dictionary with state data
    """
    arm_data = kinematic_data.get('arm', {})
    measured = arm_data.get('measured_data', {})
    measured_js = measured.get('measured_js', {})
    measured_cp = measured.get('measured_cp', {})
    local_measured_cp = measured.get('local_measured_cp', {})

    state = {}

    # Joint state
    state[f'observation.state.{arm_name}.joint_position'] = measured_js.get('position', [])
    state[f'observation.state.{arm_name}.joint_velocity'] = measured_js.get('velocity', [])
    state[f'observation.state.{arm_name}.joint_effort'] = measured_js.get('effort', [])

    # Gripper (PSM only)
    if is_psm:
        jaw_data = arm_data.get('jaw', {}).get('measured_data', {})
        gripper_pos = jaw_data.get('position', [])
        state[f'observation.state.{arm_name}.gripper'] = np.array([gripper_pos[0] if gripper_pos else 0.0])

    # Cartesian pose
    pos = measured_cp.get('position', [])
    orient = measured_cp.get('orientation', [])
    if pos and orient:
        state[f'observation.cartesian_state.{arm_name}.pose.position.x'] = np.array([pos[0]])
        state[f'observation.cartesian_state.{arm_name}.pose.position.y'] = np.array([pos[1]])
        state[f'observation.cartesian_state.{arm_name}.pose.position.z'] = np.array([pos[2]])
        state[f'observation.cartesian_state.{arm_name}.pose.orientation.x'] = np.array([orient[0]])
        state[f'observation.cartesian_state.{arm_name}.pose.orientation.y'] = np.array([orient[1]])
        state[f'observation.cartesian_state.{arm_name}.pose.orientation.z'] = np.array([orient[2]])
        state[f'observation.cartesian_state.{arm_name}.pose.orientation.w'] = np.array([orient[3]])

    # Cartesian twist
    velocity = measured_cp.get('velocity', [])
    if velocity and len(velocity) >= 6:
        state[f'observation.cartesian_state.{arm_name}.twist.linear.x'] = np.array([velocity[0]])
        state[f'observation.cartesian_state.{arm_name}.twist.linear.y'] = np.array([velocity[1]])
        state[f'observation.cartesian_state.{arm_name}.twist.linear.z'] = np.array([velocity[2]])
        state[f'observation.cartesian_state.{arm_name}.twist.angular.x'] = np.array([velocity[3]])
        state[f'observation.cartesian_state.{arm_name}.twist.angular.y'] = np.array([velocity[4]])
        state[f'observation.cartesian_state.{arm_name}.twist.angular.z'] = np.array([velocity[5]])

    # Local cartesian pose
    local_pos = local_measured_cp.get('position', [])
    local_orient = local_measured_cp.get('orientation', [])
    if local_pos and local_orient:
        state[f'observation.cartesian_state.{arm_name}.local_pose.position.x'] = np.array([local_pos[0]])
        state[f'observation.cartesian_state.{arm_name}.local_pose.position.y'] = np.array([local_pos[1]])
        state[f'observation.cartesian_state.{arm_name}.local_pose.position.z'] = np.array([local_pos[2]])
        state[f'observation.cartesian_state.{arm_name}.local_pose.orientation.x'] = np.array([local_orient[0]])
        state[f'observation.cartesian_state.{arm_name}.local_pose.orientation.y'] = np.array([local_orient[1]])
        state[f'observation.cartesian_state.{arm_name}.local_pose.orientation.z'] = np.array([local_orient[2]])
        state[f'observation.cartesian_state.{arm_name}.local_pose.orientation.w'] = np.array([local_orient[3]])

    return state


def extract_action_state(kinematic_data: Dict, arm_name: str,
                         is_psm: bool = True) -> Dict[str, List[float]]:
    """
    Extract action state data from kinematic JSON for one arm.

    Args:
        kinematic_data: Kinematic data dictionary
        arm_name: Name of the arm (for dictionary keys)
        is_psm: True if PSM arm, False if ECM

    Returns:
        Dictionary with action data
    """
    arm_data = kinematic_data.get('arm', {})
    setpoint = arm_data.get('setpoint_data', {})
    setpoint_js = setpoint.get('setpoint_js', {})

    action = {}

    # Joint setpoints
    action[f'action.{arm_name}.joint_position'] = setpoint_js.get('position', [])
    action[f'action.{arm_name}.joint_velocity'] = setpoint_js.get('velocity', [])
    action[f'action.{arm_name}.joint_effort'] = setpoint_js.get('effort', [])

    # Gripper (PSM only)
    if is_psm:
        jaw_data = arm_data.get('jaw', {}).get('setpoint_data', {})
        gripper_pos = jaw_data.get('position', [])
        action[f'action.{arm_name}.gripper'] = np.array([gripper_pos[0] if gripper_pos else 0.0])

    return action


def extract_metadata(time_sync_data: Dict, frame_id: int,
                     episode_start_frame: int, arm_names: List[str]) -> Dict[str, Any]:
    """
    Extract metadata from time synchronization data.

    Args:
        time_sync_data: Time sync data dictionary
        frame_id: Current frame ID
        episode_start_frame: First frame ID of the episode
        arm_names: List of arm names (PSM1, PSM2, etc.)

    Returns:
        Dictionary with metadata
    """
    kinematics_set = time_sync_data.get('Kinematics_set_1', {})
    image_timestamp = time_sync_data.get('image_left_stamp', {})

    # Calculate timestamp relative to episode start
    # Get start timestamp
    start_sec = image_timestamp.get('sec', 0)
    start_nsec = image_timestamp.get('nsec', 0)
    timestamp_s = start_sec + start_nsec / 1e9

    # Note: For first frame of episode, we'll need to store the start time
    # For now, just use frame_id as proxy for timestamp calculation

    metadata = {
        'frame_id': np.array([frame_id]),
        'timestamp': np.array([float(frame_id)]),  # Will be corrected in create_episode_parquet
        'observation.meta.ros_time_sec': np.array([image_timestamp.get('sec', 0)]),
        'observation.meta.ros_time_nsec': np.array([image_timestamp.get('nsec', 0)]),
        'observation.meta.time_tolerance_ms': np.array([calculate_time_tolerance_ms(
            time_sync_data, image_timestamp
        )])
    }

    # Add robot rates for each PSM
    for arm_name in arm_names:
        if arm_name.startswith('PSM'):
            arm_data = kinematics_set.get(arm_name, {})
            metadata[f'observation.meta.{arm_name}.robot_rate'] = np.array([arm_data.get('measured_frequency', 0.0)])

    return metadata


def compute_episode_statistics(parquet_file: Path, video_folders: Dict[str, Path],
                              episode_frames: List[int], max_video_samples: int = 30) -> Dict:
    """
    Compute statistical analysis for an episode.

    Omits meaningless statistics:
    - For quaternion orientations: excluded (quaternion components don't have meaningful stats)
    - For frame_id and ROS time fields: excluded (discrete identifiers, not continuous)
    - For timestamp: excluded (computed at case/phase level instead)
    - For video features: no count saved

    Args:
        parquet_file: Path to episode parquet file
        video_folders: Dictionary mapping video names to their folders
        episode_frames: List of frame IDs in this episode
        max_video_samples: Maximum number of frames to sample for video stats

    Returns:
        Dictionary with feature statistics

    Performance optimizations for large-scale data:
        - Video frame memory is bounded by max_video_samples parameter
        - Explicit memory cleanup after processing video frames
    """
    stats = {}

    if not HAS_PYARROW:
        return stats

    try:
        # Load parquet file for statistics computation
        df = pd.read_parquet(parquet_file)

        # Define fields to exclude from statistics
        exclude_from_stats = ['timestamp']  # Timestamp computed at case/phase level

        # Define fields where only count is meaningful (no mean/std/min/max)
        count_only_fields = [
            'frame_id',
            'observation.meta.ros_time_sec',
            'observation.meta.ros_time_nsec'
        ]

        # Important metadata fields that should always have statistics computed
        # These are explicitly listed to ensure they are included in the stats
        important_metadata_fields = [
            'observation.meta.time_tolerance_ms'
        ]

        # Patterns for quaternion orientation fields (only count is meaningful)
        orientation_patterns = [
            '.pose.orientation.',
            '.local_pose.orientation.'
        ]

        # Patterns for joint array fields that should preserve shape
        joint_array_patterns = [
            '.joint_position',
            '.joint_velocity',
            '.joint_effort'
        ]

        # First, explicitly compute statistics for important metadata fields
        # These fields may be stored as arrays in parquet, so we handle them specially
        for field in important_metadata_fields:
            if field in df.columns:
                try:
                    # Extract values - handle both scalar and array storage
                    raw_values = df[field].values
                    # If stored as arrays, flatten them to scalars
                    if hasattr(raw_values[0], '__len__') and not isinstance(raw_values[0], str):
                        data = np.array([v[0] if len(v) > 0 else 0.0 for v in raw_values])
                    else:
                        data = np.array(raw_values, dtype=float)

                    if len(data) > 0:
                        stats[field] = {
                            "min": [float(np.min(data))],
                            "max": [float(np.max(data))],
                            "mean": [float(np.mean(data))],
                            "std": [float(np.std(data))],
                            "count": [int(len(data))]
                        }
                except Exception as e:
                    # Log but continue if this field can't be processed
                    pass

        # Compute stats for all other columns
        for col in df.columns:
            # Skip if already processed as important metadata field
            if col in important_metadata_fields:
                continue

            # Skip excluded fields
            if col in exclude_from_stats:
                continue

            # Check if this is a count-only field
            if col in count_only_fields:
                continue

            # Check if this is an orientation field (quaternion component)
            if any(pattern in col for pattern in orientation_patterns):
                continue

            # Check if this column contains array data
            is_joint_array = any(pattern in col for pattern in joint_array_patterns)

            try:
                if is_joint_array:
                    # Handle joint arrays - preserve shape
                    # Stack all rows to get shape (num_timesteps, num_joints)
                    data_array = np.stack(df[col].values)

                    if len(data_array) > 0:
                        # Compute statistics along axis 0 (across timesteps)
                        # Result will have shape (num_joints,)
                        # LeRobot requires count to be shape (1,) - must be a list with one element
                        stats[col] = {
                            "min": np.min(data_array, axis=0).tolist(),
                            "max": np.max(data_array, axis=0).tolist(),
                            "mean": np.mean(data_array, axis=0).tolist(),
                            "std": np.std(data_array, axis=0).tolist(),
                            "count": [len(data_array)]  # Must be list for LeRobot compatibility
                        }
                else:
                    # Handle scalar numeric data
                    # Use np.issubdtype() instead of direct dtype comparison to avoid deprecation warning
                    # np.issubdtype checks if dtype is a subtype of np.number (includes int, float, etc.)
                    if np.issubdtype(df[col].dtype, np.number):
                        data = df[col].values
                        if len(data) > 0:
                            # Wrap all values in arrays for LeRobot compatibility
                            stats[col] = {
                                "min": [float(np.min(data))],
                                "max": [float(np.max(data))],
                                "mean": [float(np.mean(data))],
                                "std": [float(np.std(data))],
                                "count": [int(len(data))]
                            }
            except Exception as e:
                # Skip columns that can't be processed
                continue

        # Compute video statistics (sample frames for efficiency)
        # Note: Memory usage is bounded by max_video_samples parameter
        if HAS_CV2 and video_folders:
            # Determine sampling indices
            num_frames = len(episode_frames)
            if num_frames <= max_video_samples:
                sample_indices = list(range(num_frames))
            else:
                # Uniformly sample max_video_samples frames
                step = num_frames / max_video_samples
                sample_indices = [int(i * step) for i in range(max_video_samples)]

            sample_frame_ids = [episode_frames[i] for i in sample_indices]

            # Compute stats for each video stream
            for video_name, video_folder in video_folders.items():
                feature_name = f"observation.images.{video_name}"

                # Load sampled frames
                frames = []
                for frame_id in sample_frame_ids:
                    img_path = video_folder / f"{frame_id}.png"
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Convert BGR to RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            frames.append(img_rgb)

                if frames:
                    # Stack frames: (N, H, W, 3)
                    frames_array = np.stack(frames, axis=0)

                    # Compute per-channel statistics
                    # LeRobot expects shape (3,1,1) for image statistics
                    # where dim 0 is channels (R,G,B) and dims 1,2 are spatial
                    # count must have shape (1,)
                    stats[feature_name] = {
                        "mean": np.array([
                            [[np.mean(frames_array[:, :, :, 0])]],
                            [[np.mean(frames_array[:, :, :, 1])]],
                            [[np.mean(frames_array[:, :, :, 2])]]
                        ]).tolist(),
                        "std": np.array([
                            [[np.std(frames_array[:, :, :, 0])]],
                            [[np.std(frames_array[:, :, :, 1])]],
                            [[np.std(frames_array[:, :, :, 2])]]
                        ]).tolist(),
                        "min": np.array([
                            [[np.min(frames_array[:, :, :, 0])]],
                            [[np.min(frames_array[:, :, :, 1])]],
                            [[np.min(frames_array[:, :, :, 2])]]
                        ]).tolist(),
                        "max": np.array([
                            [[np.max(frames_array[:, :, :, 0])]],
                            [[np.max(frames_array[:, :, :, 1])]],
                            [[np.max(frames_array[:, :, :, 2])]]
                        ]).tolist(),
                        "count": [len(frames)]  # Must be list with one element for LeRobot
                    }

                    # Free memory after processing each video stream
                    del frames, frames_array

    except Exception as e:
        print(f"  Warning: Failed to compute statistics: {e}")

    return stats


def compute_duration_statistics(episode_durations: List[float]) -> Dict:
    """
    Compute duration statistics from episode durations.

    Args:
        episode_durations: List of episode durations in seconds

    Returns:
        Dictionary with duration statistics (min, max, mean, std, count)
    """
    if not episode_durations:
        return {}

    try:
        import numpy as np

        durations_array = np.array(episode_durations)
        # Wrap all values in arrays for LeRobot compatibility
        return {
            "min": [float(np.min(durations_array))],
            "max": [float(np.max(durations_array))],
            "mean": [float(np.mean(durations_array))],
            "std": [float(np.std(durations_array))],
            "count": [int(len(durations_array))]
        }

    except Exception as e:
        print(f"  Warning: Failed to compute duration statistics: {e}")

    return {}


def create_episode_parquet(dataset_folder: Path, episode_frames: List[int],
                           output_file: Path, arm_names: List[str],
                           parquet_settings: Dict, episode_index: int,
                           task_index: int) -> Tuple[float, float]:
    """
    Create parquet file for one episode.

    Args:
        dataset_folder: Path to dataset folder
        episode_frames: List of frame IDs in this episode
        output_file: Path to output parquet file
        arm_names: List of arm names to include
        parquet_settings: Parquet compression settings
        episode_index: Index of this episode (required by LeRobot)
        task_index: Index of the task/phase (required by LeRobot)

    Returns:
        Tuple of (start_timestamp, end_timestamp) in seconds

    Memory considerations for large-scale data:
        - Memory usage scales linearly with episode length (number of frames)
        - Each frame row contains ~50-100 float values depending on arm configuration
        - For typical episodes (100-1000 frames), memory usage is ~1-10 MB
        - For very long episodes (10000+ frames), consider splitting into multiple episodes
        - The parquet compression significantly reduces disk usage vs memory footprint
    """
    if not HAS_PYARROW:
        raise ImportError("pyarrow is required for parquet file creation")

    kinematic_folder = dataset_folder / 'kinematic'
    time_syn_folder = dataset_folder / 'time_syn'

    # Collect all data
    rows = []

    # Get start timestamp from first frame
    first_time_sync = read_time_sync_data(time_syn_folder, episode_frames[0])
    if first_time_sync:
        start_timestamp_dict = first_time_sync.get('image_left_stamp', {})
        start_timestamp = start_timestamp_dict.get('sec', 0) + start_timestamp_dict.get('nsec', 0) / 1e9
    else:
        start_timestamp = 0.0

    end_timestamp = start_timestamp
    frame_idx_counter = 0  # Track frame index within episode for LeRobot

    for frame_id in episode_frames:
        # Read kinematic data for all arms
        kinematic_data_all = {}
        for arm_name in arm_names:
            is_psm = arm_name.startswith('PSM')
            kin_data = read_kinematic_data(kinematic_folder, frame_id, arm_name)
            if kin_data:
                kinematic_data_all[arm_name] = kin_data

        # Read time sync data
        time_sync_data = read_time_sync_data(time_syn_folder, frame_id)

        if not time_sync_data:
            continue  # Skip frames without time sync data

        # Build row dictionary
        row = {}

        # Extract observation and action for each arm
        for arm_name, kin_data in kinematic_data_all.items():
            is_psm = arm_name.startswith('PSM')
            obs_state = extract_observation_state(kin_data, arm_name.lower(), is_psm)
            action_state = extract_action_state(kin_data, arm_name.lower(), is_psm)
            row.update(obs_state)
            row.update(action_state)

        # Extract metadata
        psm_names = [name for name in arm_names if name.startswith('PSM')]
        metadata = extract_metadata(time_sync_data, frame_id, episode_frames[0], psm_names)

        # Calculate actual timestamp relative to episode start
        curr_timestamp_dict = time_sync_data.get('image_left_stamp', {})
        curr_timestamp = curr_timestamp_dict.get('sec', 0) + curr_timestamp_dict.get('nsec', 0) / 1e9
        metadata['timestamp'] = np.array([curr_timestamp - start_timestamp])
        end_timestamp = curr_timestamp

        # Add episode_index, frame_index, and task_index (required by LeRobot)
        # Must be integers, not strings
        metadata['episode_index'] = np.array([int(episode_index)])
        metadata['frame_index'] = np.array([int(frame_idx_counter)])  # Frame index within episode
        metadata['task_index'] = np.array([int(task_index)])

        row.update(metadata)
        rows.append(row)
        frame_idx_counter += 1  # Increment frame counter for next iteration

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Write to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    compression = parquet_settings.get('compression', 'snappy')
    df.to_parquet(output_file, compression=compression, index=False)

    return start_timestamp, end_timestamp


def convert_images_to_video(image_folder: Path, output_video: Path,
                            frame_ids: List[int], fps: int,
                            video_encoding: Dict) -> bool:
    """
    Convert PNG images to MP4 video using FFmpeg for high-quality encoding.

    Args:
        image_folder: Path to folder containing PNG images
        output_video: Path to output MP4 file
        frame_ids: List of frame IDs to include (in order)
        fps: Frames per second
        video_encoding: Video encoding settings (codec, crf, preset, pixel_format)

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import tempfile

    # Check if images exist
    image_files = [image_folder / f"{fid}.png" for fid in frame_ids]
    existing_files = [f for f in image_files if f.exists()]

    if len(existing_files) == 0:
        print(f"  Warning: No images found in {image_folder}")
        return False

    # Create output directory
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_list_file = Path(f.name)
        for img_file in existing_files:
            # FFmpeg concat demuxer expects duration for each file
            # Duration = 1/fps for each frame
            f.write(f"file '{img_file.absolute()}'\n")
            f.write(f"duration {1.0/fps}\n")
        # Add last image again without duration (FFmpeg requirement)
        if existing_files:
            f.write(f"file '{existing_files[-1].absolute()}'\n")

    try:
        # Get encoding settings with defaults
        codec = video_encoding.get('codec', 'libx264')
        crf = video_encoding.get('crf', 17)
        preset = video_encoding.get('preset', 'slow')
        pixel_format = video_encoding.get('pixel_format', 'yuv420p')

        # Build ffmpeg command for high-quality encoding
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', str(temp_list_file),
            '-c:v', codec,
            '-crf', str(crf),
            '-preset', preset,
            '-pix_fmt', pixel_format,
            '-r', str(fps),  # Output frame rate
            str(output_video)
        ]

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Clean up temp file
        temp_list_file.unlink()

        return True

    except subprocess.CalledProcessError as e:
        print(f"  Warning: FFmpeg encoding failed: {e.stderr.decode()}")
        # Clean up temp file
        if temp_list_file.exists():
            temp_list_file.unlink()
        return False
    except FileNotFoundError:
        print(f"  Warning: FFmpeg not found. Please install ffmpeg for high-quality video encoding.")
        # Fallback to opencv if available
        if HAS_CV2:
            return _convert_images_to_video_opencv(image_folder, output_video,
                                                   existing_files, fps)
        # Clean up temp file
        if temp_list_file.exists():
            temp_list_file.unlink()
        return False


def _convert_images_to_video_opencv(image_folder: Path, output_video: Path,
                                    existing_files: List[Path], fps: int) -> bool:
    """
    Fallback method using OpenCV for video encoding (lower quality).

    Args:
        image_folder: Path to folder containing PNG images
        output_video: Path to output MP4 file
        existing_files: List of existing image file paths
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    if not HAS_CV2:
        print(f"  Warning: opencv-python not available, skipping video: {output_video.name}")
        return False

    # Read first image to get dimensions
    first_img = cv2.imread(str(existing_files[0]))
    if first_img is None:
        print(f"  Warning: Failed to read first image: {existing_files[0]}")
        return False

    height, width = first_img.shape[:2]

    # Setup video writer with best available codec
    # Try H.264 codec first (better quality than mp4v)
    fourcc_options = [
        cv2.VideoWriter_fourcc(*'avc1'),  # H.264
        cv2.VideoWriter_fourcc(*'H264'),  # H.264 alternative
        cv2.VideoWriter_fourcc(*'mp4v')   # MPEG-4 fallback
    ]

    video_writer = None
    for fourcc in fourcc_options:
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if video_writer.isOpened():
            break
        video_writer.release()
        video_writer = None

    if video_writer is None:
        print(f"  Warning: Failed to create video writer for {output_video}")
        return False

    # Write frames
    for img_file in existing_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            video_writer.write(img)

    video_writer.release()

    return True


def generate_features_schema(available_arms: List[str], available_videos: Dict[str, Path],
                             video_resolution: Tuple[int, int]) -> Dict:
    """
    Generate features schema for info.json based on available data.

    Args:
        available_arms: List of arm names (e.g., ['ECM', 'PSM1', 'PSM2'])
        available_videos: Dictionary mapping video names to folders
        video_resolution: Video resolution (width, height)

    Returns:
        Dictionary with features schema
    """
    features = {}
    width, height = video_resolution

    # Add video features
    for video_name in available_videos.keys():
        features[f"observation.images.{video_name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
            "info": f"Video observation from {video_name} camera in RGB format"
        }

    # Add state and action features for each arm
    for arm_name in available_arms:
        arm_lower = arm_name.lower()
        is_psm = arm_name.startswith('PSM')
        num_joints = 6 if is_psm else 4

        # Joint state features
        joint_names = []
        if is_psm:
            joint_names = ['yaw', 'pitch', 'insertion', 'roll', 'wrist_pitch', 'wrist_yaw']
        else:  # ECM
            joint_names = ['yaw', 'pitch', 'insertion', 'roll']

        features[f"observation.state.{arm_lower}.joint_position"] = {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
            "info": "Joint positions in rad (revolute) or m (prismatic)"
        }

        features[f"observation.state.{arm_lower}.joint_velocity"] = {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
            "info": "Joint velocities in rad/s or m/s"
        }

        features[f"observation.state.{arm_lower}.joint_effort"] = {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
            "info": "Joint efforts in Nm or N"
        }

        # Gripper for PSM only
        if is_psm:
            features[f"observation.state.{arm_lower}.gripper"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["angle"],
                "info": "Gripper angle in radians"
            }

        # Cartesian pose features - position
        position_info = "Position in meters (camera frame for PSM, world frame for ECM)"
        features[f"observation.cartesian_state.{arm_lower}.pose.position.x"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["x"],
            "info": position_info
        }
        features[f"observation.cartesian_state.{arm_lower}.pose.position.y"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["y"],
            "info": position_info
        }
        features[f"observation.cartesian_state.{arm_lower}.pose.position.z"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["z"],
            "info": position_info
        }

        # Cartesian orientation (quaternion) - all components share same info
        orientation_info = "Orientation as quaternion (x, y, z, w)"
        for component in ['x', 'y', 'z', 'w']:
            features[f"observation.cartesian_state.{arm_lower}.pose.orientation.{component}"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": [component],
                "info": orientation_info
            }

        # Cartesian twist (velocity) - linear components share same info
        linear_velocity_info = "Linear velocity in m/s"
        for axis in ['x', 'y', 'z']:
            features[f"observation.cartesian_state.{arm_lower}.twist.linear.{axis}"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": [axis],
                "info": linear_velocity_info
            }

        # Angular velocity components share same info
        angular_velocity_info = "Angular velocity in rad/s"
        for axis in ['x', 'y', 'z']:
            features[f"observation.cartesian_state.{arm_lower}.twist.angular.{axis}"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": [axis],
                "info": angular_velocity_info
            }

        # Local cartesian pose (relative to RCM) - position components share same info
        local_position_info = "Position relative to RCM (Remote Center of Motion) in meters"
        features[f"observation.cartesian_state.{arm_lower}.local_pose.position.x"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["x"],
            "info": local_position_info
        }
        features[f"observation.cartesian_state.{arm_lower}.local_pose.position.y"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["y"],
            "info": local_position_info
        }
        features[f"observation.cartesian_state.{arm_lower}.local_pose.position.z"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["z"],
            "info": local_position_info
        }

        # Local orientation (quaternion) - all components share same info
        local_orientation_info = "Orientation relative to RCM as quaternion (x, y, z, w)"
        for component in ['x', 'y', 'z', 'w']:
            features[f"observation.cartesian_state.{arm_lower}.local_pose.orientation.{component}"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": [component],
                "info": local_orientation_info
            }

        # Action features (similar structure to observation.state)
        features[f"action.{arm_lower}.joint_position"] = {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
            "info": "Joint position setpoints in rad or m"
        }

        features[f"action.{arm_lower}.joint_velocity"] = {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
            "info": "Joint velocity setpoints in rad/s or m/s"
        }

        features[f"action.{arm_lower}.joint_effort"] = {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
            "info": "Joint effort setpoints in Nm or N"
        }

        if is_psm:
            features[f"action.{arm_lower}.gripper"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["angle"],
                "info": "Gripper angle setpoint in radians"
            }

    # Add metadata features
    features["frame_id"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["frame_id"],
        "info": "Original frame ID from data collection"
    }

    features["timestamp"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["timestamp"],
        "info": "Time in seconds since episode start"
    }

    features["observation.meta.ros_time_sec"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["sec"],
        "info": "ROS timestamp seconds part"
    }

    features["observation.meta.ros_time_nsec"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["nsec"],
        "info": "ROS timestamp nanoseconds part"
    }

    features["observation.meta.time_tolerance_ms"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["tolerance_ms"],
        "info": "Maximum time difference between sensor timestamps in milliseconds"
    }

    # Add robot rate for each PSM
    for arm_name in available_arms:
        if arm_name.startswith('PSM'):
            arm_lower = arm_name.lower()
            features[f"observation.meta.{arm_lower}.robot_rate"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["frequency_hz"],
                "info": f"{arm_name} control loop frequency in Hz"
            }

    return features


def create_lerobot_metadata(output_folder: Path, phase_name: str, case_index: int,
                            episodes_info: List[Dict], episodes_stats_info: List[Dict],
                            meta_data: Dict,
                            phase_desc: Dict, step_desc: Dict, user_info: Dict,
                            robot_config: Dict, fps: int,
                            available_arms: List[str], available_videos: Dict[str, Path],
                            video_resolution: Tuple[int, int],
                            video_encoding: Dict[str, Any] = None,
                            dataset_config: Dict[str, Any] = None,
                            case_time_stats: Dict[str, Any] = None,
                            phase_time_stats: Dict[str, Any] = None,
                            is_append_mode: bool = True) -> None:
    """
    Create LeRobot v2.1 metadata files (info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl).

    Args:
        output_folder: Root output folder
        phase_name: Name of the phase/task
        case_index: Numeric case index (e.g., 0, 1, 2)
        episodes_info: List of episode info dictionaries
        meta_data: Dataset metadata from meta_data.json
        phase_desc: Phase idx-to-text mapping
        step_desc: Step idx-to-text mapping
        user_info: User ID to description mapping
        robot_config: Robot configuration from config
        fps: Frames per second
        available_arms: List of available arm names
        available_videos: Dictionary of available video streams
        video_resolution: Video resolution (width, height)
        video_encoding: Video encoding configuration (codec, crf, preset, etc.)
        dataset_config: Dataset configuration (repo_id, etc.)
        is_append_mode: If True, append to existing JSONL files (unknown chunk).
                        If False, remove existing entries for this chunk and rewrite (known chunk).
    """
    if video_encoding is None:
        video_encoding = {}
    if dataset_config is None:
        dataset_config = {}
    # Meta folder is now directly under output_folder (no phase_name level)
    meta_folder = output_folder / 'meta'
    meta_folder.mkdir(parents=True, exist_ok=True)

    # Create or update info.json
    info_file = meta_folder / 'info.json'

    # Generate features schema
    features = generate_features_schema(available_arms, available_videos, video_resolution)

    # Prepare encoding information with video encoder details
    encoding_info = {
        "image": "video",
        "video_codec": video_encoding.get('codec', 'libx264'),
        "video_quality_crf": video_encoding.get('crf', 17),
        "video_preset": video_encoding.get('preset', 'slow'),
        "pixel_format": video_encoding.get('pixel_format', 'yuv420p'),
        "video_fps": fps
    }

    # Calculate paths relative to output folder (no phase_name level)
    # LeRobot v2.1 expects specific placeholder names: episode_chunk, episode_index, video_key
    # We use 'case-' prefix for our folder naming with 'episode_chunk' as the placeholder
    # Format: case-{episode_chunk:03d} produces case-000, case-001, etc.
    # Format: episode_{episode_index:06d} produces episode_000000, episode_000001, etc.
    # Note: LeRobot v2.1 uses 6-digit episode indices by default
    data_path = "data/case-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_path = "videos/case-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    # Calculate totals by scanning ALL existing case folders plus current episodes
    # This ensures info.json always reflects the complete dataset state
    # Data is now directly under output_folder/data/ (no phase_name level)
    data_folder = output_folder / 'data'
    total_episodes = 0
    total_frames = 0

    # chunks_size is fixed at 100 - this defines the maximum episodes per chunk
    # and is used in the episode_index calculation: episode_index = chunks_size * episode_chunk + local_ep_idx
    chunks_size = 100

    if data_folder.exists():
        for case_dir in data_folder.iterdir():
            if case_dir.is_dir() and case_dir.name.startswith('case-'):
                parquet_files = list(case_dir.glob('episode_*.parquet'))

                # Count frames from parquet files (read metadata only for efficiency)
                for pq_file in parquet_files:
                    try:
                        import pyarrow.parquet as pq
                        pq_meta = pq.read_metadata(pq_file)
                        total_frames += pq_meta.num_rows
                        total_episodes += 1
                    except Exception:
                        # Fallback: count this as 1 episode with unknown frames
                        total_episodes += 1

    # If no existing data, use current episodes_info
    if total_episodes == 0:
        total_episodes = len(episodes_info)
        total_frames = sum(ep['length'] for ep in episodes_info)

    info_data = {
        "codebase_version": "v2.1",
        "repo_id": dataset_config.get('repo_id', 'SMARTS_LCSR_JHU'),
        "robot_type": robot_config.get('type', 'dVRK-Si'),
        "fps": fps,
        "chunks_size": chunks_size,  # Max episodes in any case folder
        "use_video": True,
        "encoding": encoding_info,
        "data_path": data_path,
        "video_path": video_path,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": features
    }
    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=2)

    # Handle episodes.jsonl based on append/overwrite mode
    episodes_file = meta_folder / 'episodes.jsonl'
    if is_append_mode:
        # Append mode: just add new episodes
        with open(episodes_file, 'a') as f:
            for ep_info in episodes_info:
                f.write(json.dumps(ep_info) + '\n')
    else:
        # Overwrite mode: remove existing entries for this chunk and rewrite
        existing_episodes = []
        if episodes_file.exists():
            with open(episodes_file, 'r') as f:
                for line in f:
                    if line.strip():
                        ep = json.loads(line)
                        # Keep episodes from OTHER chunks
                        if ep.get('episode_chunk') != case_index:
                            existing_episodes.append(ep)
        # Write back: existing (other chunks) + new (current chunk)
        with open(episodes_file, 'w') as f:
            for ep in existing_episodes:
                f.write(json.dumps(ep) + '\n')
            for ep_info in episodes_info:
                f.write(json.dumps(ep_info) + '\n')

    # Create or update tasks.jsonl
    tasks_file = meta_folder / 'tasks.jsonl'
    # Read existing tasks to check if this task_index already exists
    existing_task_indices = set()
    if tasks_file.exists():
        with open(tasks_file, 'r') as f:
            for line in f:
                if line.strip():
                    task_data = json.loads(line)
                    existing_task_indices.add(task_data.get('task_index'))

    # Find task_index for this phase_name
    phase_idx = None
    for idx, name in phase_desc.items():
        if name == phase_name:
            phase_idx = idx
            break

    # Only add if this task_index is not already in the file
    # (same task_index = same task, no duplicate needed)
    if phase_idx and int(phase_idx) not in existing_task_indices:
        task_data = {
            "task_index": int(phase_idx),  # Must be integer
            "task": phase_name
        }
        with open(tasks_file, 'a') as f:
            f.write(json.dumps(task_data) + '\n')

    # Handle episodes_stats.jsonl based on append/overwrite mode
    # This file contains statistical analysis of each episode's data
    # JSONL format: one JSON object per line
    stats_file = meta_folder / 'episodes_stats.jsonl'

    # Prepare stats with case/phase time info
    stats_to_write = []
    for ep_stats in episodes_stats_info:
        ep_stats_copy = ep_stats.copy()
        if case_time_stats:
            ep_stats_copy['case_time_stats'] = case_time_stats
        if phase_time_stats:
            ep_stats_copy['phase_time_stats'] = phase_time_stats
        # Add episode_chunk for filtering in overwrite mode
        ep_stats_copy['episode_chunk'] = case_index
        stats_to_write.append(ep_stats_copy)

    if is_append_mode:
        # Append mode: just add new stats
        with open(stats_file, 'a') as f:
            for ep_stats in stats_to_write:
                f.write(json.dumps(ep_stats) + '\n')
    else:
        # Overwrite mode: remove existing entries for this chunk and rewrite
        existing_stats = []
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                for line in f:
                    if line.strip():
                        stat = json.loads(line)
                        # Keep stats from OTHER chunks
                        if stat.get('episode_chunk') != case_index:
                            existing_stats.append(stat)
        # Write back: existing (other chunks) + new (current chunk)
        with open(stats_file, 'w') as f:
            for stat in existing_stats:
                f.write(json.dumps(stat) + '\n')
            for ep_stats in stats_to_write:
                f.write(json.dumps(ep_stats) + '\n')


def copy_calibration_files(dataset_folder: Path, output_folder: Path,
                           case_index: int) -> None:
    """
    Copy camera and hand-eye calibration files to output.

    For hand-eye calibration, only copies the specific registration JSON files:
    - PSMx-registration-dVRK.json
    - PSMx-registration-open-cv.json
    Ignores any backup subfolders or other files.

    Args:
        dataset_folder: Source dataset folder
        output_folder: Root output folder
        case_index: Numeric case index
    """
    # Calibration files now directly under output_folder/calibrations/case-xxx/
    calib_output = output_folder / 'calibrations' / f'case-{case_index:03d}'
    calib_output.mkdir(parents=True, exist_ok=True)

    # Copy camera calibration (copy entire folder as-is)
    camera_calib_src = dataset_folder / 'camera_calibration'
    camera_calib_dst = calib_output / 'camera_calibration'
    if camera_calib_src.exists():
        shutil.copytree(camera_calib_src, camera_calib_dst, dirs_exist_ok=True)

    # Copy hand-eye calibration (only specific JSON files, ignore backup subfolders)
    handeye_calib_src = dataset_folder / 'hand_eye_calibration'
    handeye_calib_dst = calib_output / 'hand_eye_calibration'
    if handeye_calib_src.exists():
        handeye_calib_dst.mkdir(parents=True, exist_ok=True)

        # Only copy PSMx-registration-dVRK.json and PSMx-registration-open-cv.json files
        # Pattern: PSM followed by a digit (1, 2, or 3) then the specific suffix
        for src_file in handeye_calib_src.iterdir():
            # Skip directories (backup folders, etc.)
            if src_file.is_dir():
                continue

            # Only copy files matching the expected patterns
            filename = src_file.name
            if (filename.startswith('PSM') and
                (filename.endswith('-registration-dVRK.json') or
                 filename.endswith('-registration-open-cv.json'))):
                dst_file = handeye_calib_dst / filename
                shutil.copy2(src_file, dst_file)


def process_single_dataset(dataset_folder: Path, output_folder: Path,
                           config: ConversionConfig, phase_desc: Dict,
                           step_desc: Dict, user_info: Dict,
                           case_start_idx: int = 0,
                           phase_case_counters: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Process a single dataset folder and convert to LeRobot format.

    Args:
        dataset_folder: Path to source dataset folder
        output_folder: Path to root output folder
        config: Conversion configuration
        phase_desc: Phase idx-to-text mapping
        step_desc: Step idx-to-text mapping
        user_info: User ID to description mapping
        case_start_idx: Starting case index (-1 for auto-detect, >= 0 for explicit)
        phase_case_counters: Dict tracking next case index per phase (for recursive mode)
                             This dict is mutated to track state across calls.

    Returns:
        Dictionary with processing statistics and time information
    """
    print(f"\n{'='*70}")
    print(f"Processing dataset: {dataset_folder}")
    print(f"{'='*70}")

    # Load metadata
    meta_data = load_json_file(dataset_folder / 'meta_data.json')
    if not meta_data:
        print("  Error: Could not load meta_data.json")
        return {}

    # Get annotation folder
    annotation_folder = dataset_folder / 'annotation'

    # Read all phase annotations to find unique phases
    phase_folder = annotation_folder / 'phase'
    phase_files = sorted(phase_folder.glob('*.json'), key=lambda x: int(x.stem))

    # Get unique phase indices
    phase_indices = set()
    for phase_file in phase_files:
        data = load_json_file(phase_file)
        if data:
            phase_indices.add(data.get('phase'))

    # Statistics
    stats = {
        'phases': {},
        'total_episodes': 0,
        'total_frames': 0
    }

    # Process each phase
    for phase_idx in sorted(phase_indices):
        if phase_idx is None:
            continue

        phase_name = phase_desc.get(str(phase_idx), f"unknown_phase_{phase_idx}")
        print(f"\n  Processing phase: {phase_name} (idx: {phase_idx})")

        # Group frames into episodes
        episodes = group_frames_into_episodes(annotation_folder, str(phase_idx))

        if not episodes:
            print(f"    No episodes found for phase {phase_idx}")
            continue

        print(f"    Found {len(episodes)} episode(s)")

        # Determine case index (pure numeric, like episode_index)
        # Logic depends on mode and start_idx:
        #   - single mode with start_idx >= 0: use start_idx directly
        #   - single mode with start_idx == -1: auto-detect next available
        #   - recursive mode with start_idx >= 0: use global counter (starts from start_idx)
        #   - recursive mode with start_idx == -1: auto-detect next available
        # Note: case_index is now GLOBAL (not per-phase) since all data goes to one folder
        if config.mode == 'recursive' and case_start_idx >= 0 and phase_case_counters is not None:
            # Recursive mode with explicit start_idx: use global counter
            # Use '_global' key since case indices are now global across all phases
            if '_global' not in phase_case_counters:
                phase_case_counters['_global'] = case_start_idx
            case_index = phase_case_counters['_global']
            phase_case_counters['_global'] += 1  # Increment for next dataset
        elif case_start_idx >= 0:
            # Single mode with explicit start_idx
            case_index = case_start_idx
        else:
            # Auto-detect next available case index (global)
            case_index = get_next_case_index(output_folder)

        # Create case_id (formatted string, like episode_id)
        case_id = f"case-{case_index:03d}"
        print(f"    Case Index: {case_index}, Case ID: {case_id}")

        # Define output folders (no phase_name level - all combined in output_folder)
        data_output = output_folder / 'data' / case_id
        video_output = output_folder / 'videos' / case_id

        # Determine if this is overwrite mode (existing case being replaced)
        # Check if this case folder already exists with episode data
        case_already_exists = data_output.exists() and any(data_output.glob('episode_*.parquet'))

        # In recursive mode, never overwrite - stop if case already exists
        if config.mode == 'recursive' and case_already_exists:
            print(f"\n    Error: {case_id} already exists.")
            print(f"    Recursive mode does not allow overwriting. Stopping.")
            print(f"    Use single mode if you need to overwrite existing cases.")
            return {}

        # Overwrite mode only allowed in single mode
        is_overwrite_mode = config.mode == 'single' and case_start_idx >= 0 and case_already_exists

        if is_overwrite_mode:
            print(f"    Overwrite mode: Clearing existing data for {case_id}")
            # Clear existing parquet files in data folder
            if data_output.exists():
                for pq_file in data_output.glob('episode_*.parquet'):
                    pq_file.unlink()
            # Clear existing video files in video folder
            if video_output.exists():
                for video_subdir in video_output.iterdir():
                    if video_subdir.is_dir():
                        for mp4_file in video_subdir.glob('episode_*.mp4'):
                            mp4_file.unlink()

        # Create output folders
        data_output.mkdir(parents=True, exist_ok=True)
        video_output.mkdir(parents=True, exist_ok=True)

        # Create meta folder (at output_folder level, not per phase)
        meta_folder = output_folder / 'meta'
        meta_folder.mkdir(parents=True, exist_ok=True)

        # Get available arms
        kinematic_folder = dataset_folder / 'kinematic'
        available_arms = []
        for arm in ['ECM', 'PSM1', 'PSM2', 'PSM3']:
            if (kinematic_folder / arm).exists():
                available_arms.append(arm)

        # Get available video streams
        image_folder = dataset_folder / 'image'
        available_videos = {}
        for subfolder in image_folder.iterdir():
            if subfolder.is_dir():
                video_name = config.video_name_mapping.get(subfolder.name, subfolder.name)
                available_videos[video_name] = subfolder

        # Process each episode
        episodes_info = []
        episodes_stats_info = []
        phase_stats = {
            'episode_chunk': case_index,  # Use episode_chunk for LeRobot v2.1 compatibility
            'case_id': case_id,
            'episodes': [],
            'total_duration_s': 0.0
        }

        # chunks_size is fixed at 100 - used to calculate global episode_index
        # Formula: global_episode_index = chunks_size * episode_chunk + local_ep_idx
        chunks_size = 100

        for local_ep_idx, episode_frames in enumerate(tqdm(episodes, desc=f"    Episodes")):
            # Calculate global episode_index using the formula:
            # episode_index = chunks_size * episode_chunk + local_episode_index
            # This ensures unique episode indices across all chunks
            episode_index = chunks_size * case_index + local_ep_idx

            # LeRobot v2.1 uses 6-digit episode indices
            # Use global episode_index for filename convention
            episode_id = f"episode_{episode_index:06d}"

            # Create parquet file
            parquet_file = data_output / f"{episode_id}.parquet"
            start_ts, end_ts = create_episode_parquet(
                dataset_folder, episode_frames, parquet_file,
                available_arms, config.parquet_settings, episode_index, phase_idx
            )

            duration_s = end_ts - start_ts

            # Create videos for each camera
            for video_name, video_folder in available_videos.items():
                video_subfolder = video_output / f"observation.images.{video_name}"
                video_file = video_subfolder / f"{episode_id}.mp4"
                convert_images_to_video(
                    video_folder, video_file, episode_frames,
                    config.fps, config.video_encoding
                )

            # Get step description
            first_frame = episode_frames[0]
            step_idx = read_annotation_data(annotation_folder, first_frame, 'step')
            step_description = step_desc.get(str(step_idx), f"unknown_step_{step_idx}")

            # Get user information from meta_data
            user_id = meta_data.get('user_id', '')
            user_description = user_info.get(str(user_id), 'Unknown user')
            skill_level = meta_data.get('operator_skill_level', 'Unknown')
            case_type = meta_data.get('case_type', 'Unknown')
            tool_type = meta_data.get('tool', {})

            # Compute episode statistics (use configured max_video_samples)
            max_video_samples = config.statistics.get('max_video_samples', 30)
            episode_stats = compute_episode_statistics(
                parquet_file, available_videos, episode_frames, max_video_samples=max_video_samples
            )

            # Episode info for metadata (with all required fields)
            # Episode info for episodes.jsonl (WITHOUT stats)
            # Use 'episode_chunk' for LeRobot v2.1 compatibility (matches path template placeholder)
            # episode_index is the GLOBAL index: chunks_size * episode_chunk + local_ep_idx
            ep_info = {
                'episode_index': episode_index,  # Global episode index across all chunks
                'episode_id': episode_id,
                'episode_chunk': case_index,  # LeRobot expects 'episode_chunk' for path formatting
                'case_id': case_id,  # Keep case_id format for folder naming reference
                'task_index': int(phase_idx),  # Must be integer
                'task': phase_name,
                'instruction_text': step_description,
                'duration_s': duration_s,
                'length': len(episode_frames),
                'user_id': user_id,
                'user_description': user_description,
                'skill_level': skill_level,
                'data_collection_scenario': case_type,
                'tool_type': tool_type
            }
            episodes_info.append(ep_info)

            # Episode stats info for episodes_stats.jsonl (WITH stats)
            ep_stats_info = {
                'episode_index': episode_index,  # Global episode index
                'stats': episode_stats
            }
            episodes_stats_info.append(ep_stats_info)

            phase_stats['episodes'].append({
                'episode_index': episode_index,
                'episode_id': episode_id,
                'duration_s': duration_s,
                'length': len(episode_frames)
            })
            phase_stats['total_duration_s'] += duration_s

            stats['total_episodes'] += 1
            stats['total_frames'] += len(episode_frames)

        # Explicit garbage collection after processing all episodes for this phase
        # Helps manage memory when processing phases with many episodes
        gc.collect()

        # Compute case-level duration statistics (statistics of episode durations in this case)
        case_time_stats = None
        if config.statistics.get('compute_case_level', True):
            # Collect all episode durations from this case
            case_durations = [ep['duration_s'] for ep in phase_stats['episodes']]
            case_time_stats = compute_duration_statistics(case_durations)

        # Get video resolution from first available video
        video_resolution = (1920, 1080)  # Default
        if available_videos:
            first_video_folder = list(available_videos.values())[0]
            first_image_files = list(first_video_folder.glob('*.png'))
            if first_image_files and HAS_CV2:
                import cv2
                first_img = cv2.imread(str(first_image_files[0]))
                if first_img is not None:
                    video_resolution = (first_img.shape[1], first_img.shape[0])  # (width, height)

        # Compute phase-level duration statistics (statistics of episode durations across all cases with same task)
        phase_time_stats = None
        if config.statistics.get('compute_phase_level', True):
            # Read all existing episodes to compute task-level stats
            episodes_file = output_folder / 'meta' / 'episodes.jsonl'
            all_phase_durations = []

            # Add current case durations
            all_phase_durations.extend([ep['duration_s'] for ep in episodes_info])

            # Read durations from existing episodes with same task (if any)
            if episodes_file.exists():
                try:
                    with open(episodes_file, 'r') as f:
                        for line in f:
                            ep_data = json.loads(line)
                            # Only include episodes with the same task
                            if ep_data.get('task') == phase_name and 'duration_s' in ep_data:
                                all_phase_durations.append(ep_data['duration_s'])
                except Exception:
                    pass

            if all_phase_durations:
                phase_time_stats = compute_duration_statistics(all_phase_durations)

        # Use the is_overwrite_mode flag determined earlier
        # - Append mode (is_append_mode=True): When adding a NEW case (auto-detect or non-existing chunk)
        # - Overwrite mode (is_append_mode=False): When REPLACING an existing case (known chunk that exists)
        is_append_mode = not is_overwrite_mode

        # Create LeRobot metadata files
        create_lerobot_metadata(
            output_folder, phase_name, case_index, episodes_info, episodes_stats_info,
            meta_data, phase_desc, step_desc, user_info,
            config.robot, config.fps,
            available_arms, available_videos, video_resolution,
            config.video_encoding, config.dataset, case_time_stats, phase_time_stats,
            is_append_mode
        )

        # Copy calibration files
        copy_calibration_files(dataset_folder, output_folder, case_index)

        stats['phases'][phase_name] = phase_stats

    return stats


def convert_datasets(config: ConversionConfig) -> Dict:
    """
    Main conversion function.

    Args:
        config: Conversion configuration

    Returns:
        Dictionary with conversion statistics and time information
    """
    print("="*70)
    print("dVRK to LeRobot v2.1 Conversion")
    print("="*70)
    print(f"Mode: {config.mode}")
    print(f"Data folder: {config.data_folder}")
    print(f"Output folder: {config.output_folder}")
    print(f"FPS: {config.fps}")
    print("="*70)

    # Load description mappings
    task_desc_folder = Path(config.task_description_folder)
    phase_desc, step_desc, user_info = load_description_mappings(task_desc_folder)

    # Find dataset folders
    data_folder = Path(config.data_folder)
    if config.mode == 'single':
        if validate_dataset_folder(data_folder):
            # The folder itself is a valid dataset
            dataset_folders = [data_folder]
        else:
            # Check if there are dataset subfolders inside
            subfolders = find_dataset_folders(data_folder, max_depth=5)
            if len(subfolders) > 1:
                print(f"Error: Single mode selected, but found {len(subfolders)} datasets in {data_folder}")
                print("  Found datasets:")
                for sf in subfolders[:10]:  # Show first 10
                    print(f"    - {sf}")
                if len(subfolders) > 10:
                    print(f"    ... and {len(subfolders) - 10} more")
                print("\nPlease either:")
                print("  1. Specify a single dataset folder directly, or")
                print("  2. Use 'recursive' mode to process all datasets")
                return {}
            elif len(subfolders) == 1:
                # Found exactly one dataset subfolder, use it
                print(f"Note: Using dataset found at {subfolders[0]}")
                dataset_folders = subfolders
            else:
                print(f"Error: {data_folder} is not a valid dataset folder and contains no valid datasets")
                return {}
    else:  # recursive
        print(f"\nSearching for dataset folders in {data_folder}...")
        dataset_folders = find_dataset_folders(data_folder)
        print(f"Found {len(dataset_folders)} dataset folder(s)")

    if not dataset_folders:
        print("No dataset folders found")
        return {}

    # Process each dataset
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # In recursive mode with start_idx >= 0, validate that no existing cases would conflict
    # Case indices are now global (no phase_name level)
    if config.mode == 'recursive' and config.start_idx >= 0:
        print(f"\nRecursive mode with start_idx={config.start_idx}")
        print("Checking for existing case indices that would conflict...")

        # Scan data folder for existing case indices (now directly under output_folder/data/)
        conflicting_cases = []
        data_dir = output_folder / 'data'
        if data_dir.exists():
            for case_dir in data_dir.iterdir():
                if case_dir.is_dir() and case_dir.name.startswith('case-'):
                    try:
                        existing_idx = int(case_dir.name[5:])
                        if existing_idx >= config.start_idx:
                            conflicting_cases.append((case_dir.name, existing_idx))
                    except ValueError:
                        continue

        if conflicting_cases:
            print(f"\nError: Found {len(conflicting_cases)} existing case(s) with index >= {config.start_idx}")
            print("  Conflicting cases:")
            for case_name, idx in sorted(conflicting_cases, key=lambda x: x[1])[:15]:
                print(f"    - {case_name} (index: {idx})")
            if len(conflicting_cases) > 15:
                print(f"    ... and {len(conflicting_cases) - 15} more")
            print("\nPlease either:")
            print(f"  1. Use a higher start_idx (>= {max(c[1] for c in conflicting_cases) + 1}), or")
            print("  2. Use start_idx=-1 to auto-detect next available index, or")
            print("  3. Remove the conflicting case folders first")
            return {}

        print("  No conflicts found. Starting from case index", config.start_idx)

    all_stats = {
        'datasets': [],
        'phases': {},  # Aggregated statistics per phase
        'total_episodes': 0,
        'total_frames': 0,
        'total_duration_s': 0.0
    }

    # Track running case index per phase for recursive mode with start_idx >= 0
    # Key: phase_name, Value: next case index to use
    phase_case_counters = {}

    for dataset_folder in dataset_folders:
        stats = process_single_dataset(
            dataset_folder, output_folder, config,
            phase_desc, step_desc, user_info,
            config.start_idx,
            phase_case_counters  # Pass per-phase counters
        )

        if stats:
            # Calculate path relative to data_folder
            try:
                relative_path = str(Path(dataset_folder).relative_to(Path(config.data_folder)))
            except ValueError:
                # If not relative, use the folder name
                relative_path = Path(dataset_folder).name

            all_stats['datasets'].append({
                'path': relative_path,
                'stats': stats
            })
            all_stats['total_episodes'] += stats.get('total_episodes', 0)
            all_stats['total_frames'] += stats.get('total_frames', 0)

            # Aggregate statistics by phase
            for phase_name, phase_stats in stats.get('phases', {}).items():
                # Initialize phase entry if not exists
                if phase_name not in all_stats['phases']:
                    all_stats['phases'][phase_name] = {
                        'total_episodes': 0,
                        'total_frames': 0,
                        'total_duration_s': 0.0,
                        'cases': []  # List of case statistics for this phase
                    }

                # Aggregate phase-level totals
                phase_duration = phase_stats.get('total_duration_s', 0.0)
                all_stats['phases'][phase_name]['total_duration_s'] += phase_duration
                all_stats['total_duration_s'] += phase_duration

                # Count episodes and frames from this phase's episodes
                for episode in phase_stats.get('episodes', []):
                    all_stats['phases'][phase_name]['total_episodes'] += 1
                    all_stats['phases'][phase_name]['total_frames'] += episode.get('length', 0)

                # Add case/chunk info to phase (use episode_chunk for LeRobot v2.1 compatibility)
                all_stats['phases'][phase_name]['cases'].append({
                    'episode_chunk': phase_stats.get('episode_chunk'),
                    'case_id': phase_stats.get('case_id'),
                    'dataset_path': relative_path,
                    'duration_s': phase_duration,
                    'num_episodes': len(phase_stats.get('episodes', [])),
                    'episodes': phase_stats.get('episodes', [])
                })

        # Explicit garbage collection after each dataset to manage memory for large-scale processing
        # This helps prevent memory accumulation when processing many datasets sequentially
        gc.collect()

    # Save total time statistics if requested
    # Read from output_folder/meta/episodes.jsonl and aggregate by task/phase and case
    if config.get_total_time:
        # Reset stats and rebuild from all existing data
        complete_stats = {
            'datasets': all_stats.get('datasets', []),  # Keep the list of processed datasets
            'tasks': {},  # Aggregated by task (phase)
            'cases': [],  # All cases
            'total_episodes': 0,
            'total_frames': 0,
            'total_duration_s': 0.0
        }

        # Read episodes from the single episodes.jsonl (no longer per-phase folders)
        episodes_file = output_folder / 'meta' / 'episodes.jsonl'

        if episodes_file.exists():
            # Group episodes by case (episode_chunk) and task
            case_episodes = {}  # episode_chunk -> list of episodes
            try:
                with open(episodes_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            ep_data = json.loads(line)
                            chunk_idx = ep_data.get('episode_chunk', 0)
                            if chunk_idx not in case_episodes:
                                case_episodes[chunk_idx] = []
                            case_episodes[chunk_idx].append(ep_data)
            except Exception as e:
                logger.warning(f"Error reading episodes.jsonl: {e}")

            # Aggregate stats by case
            for chunk_idx, episodes in sorted(case_episodes.items()):
                case_duration = sum(ep.get('duration_s', 0.0) for ep in episodes)
                case_frames = sum(ep.get('length', 0) for ep in episodes)
                case_id = f"case-{chunk_idx:03d}"
                # Get task index from first episode (all episodes in a case have same task)
                # Use task_index instead of task description for consistency
                task_index = episodes[0].get('task_index', -1) if episodes else -1

                case_info = {
                    'episode_chunk': chunk_idx,
                    'case_id': case_id,
                    'task': task_index,  # Use task index instead of task description
                    'duration_s': case_duration,
                    'num_episodes': len(episodes),
                    'episodes': [
                        {
                            'episode_index': ep.get('episode_index'),
                            'episode_id': ep.get('episode_id'),
                            'duration_s': ep.get('duration_s', 0.0),
                            'length': ep.get('length', 0)
                        }
                        for ep in episodes
                    ]
                }
                complete_stats['cases'].append(case_info)

                # Aggregate by task index (use string key for JSON compatibility)
                task_key = str(task_index)
                if task_key not in complete_stats['tasks']:
                    complete_stats['tasks'][task_key] = {
                        'total_episodes': 0,
                        'total_frames': 0,
                        'total_duration_s': 0.0,
                        'num_cases': 0
                    }
                complete_stats['tasks'][task_key]['total_episodes'] += len(episodes)
                complete_stats['tasks'][task_key]['total_frames'] += case_frames
                complete_stats['tasks'][task_key]['total_duration_s'] += case_duration
                complete_stats['tasks'][task_key]['num_cases'] += 1

                # Update global totals
                complete_stats['total_episodes'] += len(episodes)
                complete_stats['total_frames'] += case_frames
                complete_stats['total_duration_s'] += case_duration

        # Calculate time_tolerance_ms statistics from all parquet files in output_folder
        # Uses Welford's online algorithm for memory-efficient streaming computation
        # This approach handles large-scale datasets without loading all values into memory
        data_folder = output_folder / 'data'

        # Welford's online algorithm state variables for streaming mean and variance
        # Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        n_samples = 0  # Total count of samples
        running_mean = 0.0  # Running mean
        running_m2 = 0.0  # Sum of squared differences from the mean (for variance)
        global_min = float('inf')
        global_max = float('-inf')

        if data_folder.exists() and HAS_PYARROW:
            # Find all parquet files in data folder (case-xxx/episode_xxxxxx.parquet)
            parquet_files = sorted(data_folder.glob('case-*/episode_*.parquet'))
            num_files = len(parquet_files)
            logger.info(f"Reading time_tolerance_ms from {num_files} parquet files (streaming mode)...")

            # Process files and log progress for large datasets
            log_interval = max(1, num_files // 10)  # Log every 10% progress

            for file_idx, pq_file in enumerate(parquet_files):
                try:
                    # Read only the specific column we need using pyarrow for efficiency
                    # This avoids loading entire parquet file into memory
                    parquet_table = pq.read_table(
                        pq_file,
                        columns=['observation.meta.time_tolerance_ms']
                    )

                    if 'observation.meta.time_tolerance_ms' in parquet_table.column_names:
                        # Convert to numpy array for efficient batch processing
                        tolerance_col = parquet_table['observation.meta.time_tolerance_ms'].to_numpy()

                        # Process values - handle both scalar and array-like storage
                        for val in tolerance_col:
                            # Handle case where value might be stored as array
                            if hasattr(val, '__iter__') and not isinstance(val, str):
                                values_to_process = np.asarray(val, dtype=np.float64)
                            else:
                                values_to_process = np.array([float(val)], dtype=np.float64)

                            # Batch update statistics using Welford's algorithm
                            # This is more efficient than processing one value at a time
                            for v_float in values_to_process:
                                n_samples += 1
                                delta = v_float - running_mean
                                running_mean += delta / n_samples
                                delta2 = v_float - running_mean
                                running_m2 += delta * delta2

                            # Batch update min/max (more efficient for arrays)
                            batch_min = np.min(values_to_process)
                            batch_max = np.max(values_to_process)
                            if batch_min < global_min:
                                global_min = batch_min
                            if batch_max > global_max:
                                global_max = batch_max

                except Exception as e:
                    logger.warning(f"Error reading time_tolerance_ms from {pq_file}: {e}")

                # Log progress for large datasets
                if (file_idx + 1) % log_interval == 0 or file_idx == num_files - 1:
                    logger.debug(f"Processed {file_idx + 1}/{num_files} parquet files...")

        elif not HAS_PYARROW:
            logger.warning("pyarrow not available, skipping time_tolerance_ms statistics")

        # Compute final statistics from Welford's algorithm state
        if n_samples > 0:
            # Variance = M2 / n (population variance), std = sqrt(variance)
            variance = running_m2 / n_samples if n_samples > 0 else 0.0
            std_dev = np.sqrt(variance)

            complete_stats['time_tolerance_ms'] = {
                'min': float(global_min),
                'max': float(global_max),
                'mean': float(running_mean),
                'std': float(std_dev),
                'num_samples': n_samples
            }
            logger.info(f"Time tolerance stats: min={global_min:.3f}ms, "
                       f"max={global_max:.3f}ms, "
                       f"mean={running_mean:.3f}ms, "
                       f"std={std_dev:.3f}ms, "
                       f"samples={n_samples}")
        else:
            logger.warning("No time_tolerance_ms data found in parquet files")
            complete_stats['time_tolerance_ms'] = {
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'num_samples': 0
            }

        total_time_file = output_folder / 'total_time.json'
        with open(total_time_file, 'w') as f:
            json.dump(complete_stats, f, indent=2)
        print(f"\nTotal time statistics saved to: {total_time_file}")

    print(f"\n{'='*70}")
    print("Conversion Complete!")
    print(f"  Total datasets processed: {len(all_stats['datasets'])}")
    print(f"  Total episodes: {all_stats['total_episodes']}")
    print(f"  Total frames: {all_stats['total_frames']}")
    print(f"  Total duration: {all_stats['total_duration_s']:.2f} seconds")
    print(f"{'='*70}")

    return all_stats


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="convert_openh_config", node=ConversionConfig)

# Set config path
project_root = Path(__file__).resolve().parent.parent
config_path = project_root / 'config'

if not config_path.exists():
    raise FileNotFoundError(f"Config directory not found: {config_path}")


@hydra.main(
    version_base=None,
    config_path=str(config_path),
    config_name="config_convert_openh"
)
def main(cfg: DictConfig):
    """
    Main entry point for the conversion script.

    Args:
        cfg: Hydra configuration object
    """
    # Convert DictConfig to ConversionConfig
    config = ConversionConfig(
        workspace=cfg.workspace,
        mode=cfg.mode,
        data_folder=cfg.data_folder,
        output_folder=cfg.output_folder,
        task_description_folder=cfg.task_description_folder,
        start_idx=cfg.start_idx,
        fps=cfg.fps,
        get_total_time=cfg.get_total_time,
        video_encoding=dict(cfg.get('video_encoding', {})),
        parquet_settings=dict(cfg.get('parquet_settings', {})),
        processing=dict(cfg.get('processing', {})),
        statistics=dict(cfg.get('statistics', {})),
        dataset=dict(cfg.get('dataset', {})),
        robot=dict(cfg.get('robot', {})),
        video_name_mapping=dict(cfg.get('video_name_mapping', {}))
    )

    # Run conversion
    try:
        stats = convert_datasets(config)

        if stats and len(stats.get('datasets', [])) > 0:
            print("\n✓ Conversion successful")
        else:
            print("\n✗ No datasets were converted")

    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
