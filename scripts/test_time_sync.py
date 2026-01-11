"""
Time Synchronization Testing Script for dVRK Multi-modal Data Collection

This script analyzes temporal alignment between multiple video streams by finding
the best matching frames within a search window. It processes datasets collected
with the dVRK system where cameras may have slight temporal misalignments.

Key Features:
- Loads timestamp data from timetable JSON files (nanosecond precision)
- Automatically determines reference video based on frame count
- Finds best matching frames using temporal search window
- Supports multiple strategies: "stereo" (left/right priority) or "all" cameras
- Outputs synchronization results as JSON files for each camera

Algorithm:
  For each non-reference video:
    For each frame k in the REFERENCE video:
      Search frames [k-n, k+n] in the selected (non-reference) video
      Find frame with minimum time difference
      Store the match index and time difference (in milliseconds)

  Note: The search is FROM reference TO selected videos, ensuring every
  reference frame has a match in all other videos.

Output Format:
  - For non-reference videos:
    * matched_sequence: mapping of REFERENCE frame indices to best matches in this video
      {reference_frame_idx: {"best_match": selected_video_frame_idx, "time_diff": time_diff_ms}}
    * time_diff: time difference between consecutive frames in this video (ms)
  - For reference video:
    * time_diff: time difference between consecutive frames (ms)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
import numpy as np


@dataclass
class TimeSyncConfig:
    """
    Configuration for time synchronization testing.

    All parameters are loaded from config_test_sync.yaml via Hydra.
    """
    workspace: str
    input_folder: str
    output_folder: str
    video_fps: int
    search_window: int
    strategy: str  # "stereo" or "all"
    default_reference_camera: str
    camera_names: List[str]
    verbose: bool


def load_timetable_json(timetable_path: Path) -> Dict[int, int]:
    """
    Load timestamp data from a timetable JSON file.

    Args:
        timetable_path: Path to <camera_name>_timetable.json file

    Returns:
        Dictionary mapping frame index to timestamp in nanoseconds
        Example: {0: 1214438695911726, 1: 1214438828430078, ...}

    Note:
        The JSON file format is an array of objects:
        [{"frame": 0, "timestamp_ns": 1214438695911726}, ...]

        This function converts it to a simple dict for fast lookup.
    """
    if not timetable_path.exists():
        raise FileNotFoundError(f"Timetable file not found: {timetable_path}")

    with open(timetable_path, 'r') as f:
        data = json.load(f)

    # Convert list of dicts to dict mapping frame -> timestamp_ns
    timetable = {}
    for entry in data:
        frame_idx = entry["frame"]
        timestamp_ns = entry["timestamp_ns"]
        timetable[frame_idx] = timestamp_ns

    return timetable


def count_frames_in_folder(image_folder: Path) -> int:
    """
    Count the number of frame files in an image folder.

    Args:
        image_folder: Path to image folder (e.g., regular/image/left/)

    Returns:
        Number of image files in the folder

    Note:
        Only counts files (not subdirectories). Assumes all files are frames.
    """
    if not image_folder.exists():
        return 0

    # Count all files in the folder (assuming all are image frames)
    frame_count = sum(1 for item in image_folder.iterdir() if item.is_file())

    return frame_count


def determine_reference_camera(
    frame_counts: Dict[str, int],
    strategy: str,
    default_reference: str = ""
) -> str:
    """
    Determine which camera to use as the reference video.

    Args:
        frame_counts: Dictionary mapping camera names to frame counts
        strategy: "stereo" or "all"
        default_reference: Optional camera name to use as reference (overrides auto-detection)

    Returns:
        Name of the reference camera

    Note:
        Selection logic:
        1. If default_reference is specified and exists, use it
        2. For "stereo" strategy:
           - Only consider "left" and "right" cameras
           - Use the one with FEWEST frames (this is the limiting factor)
           - If tied, use "left" as reference
        3. For "all" strategy:
           - Consider all cameras
           - Use the one with FEWEST frames (this is the limiting factor)
           - If multiple have min frames, use first alphabetically

        Rationale: Using the camera with fewest frames as reference ensures that
        all reference frames can be matched by the other cameras (which have more
        frames to search through). This prevents issues where the reference has
        frames beyond the range of other cameras.
    """
    # If default reference is specified and valid, use it
    if default_reference and default_reference in frame_counts:
        return default_reference

    # Apply strategy-specific logic
    if strategy == "stereo":
        # Only consider left and right cameras
        stereo_cameras = {k: v for k, v in frame_counts.items() if k in ["left", "right"]}

        if not stereo_cameras:
            raise ValueError("No stereo cameras (left/right) found in the dataset")

        # Find camera with FEWEST frames (minimum, not maximum)
        min_frames = min(stereo_cameras.values())
        candidates = [cam for cam, count in stereo_cameras.items() if count == min_frames]

        # If tied, prefer "left"
        if "left" in candidates:
            return "left"
        else:
            return candidates[0]

    else:  # strategy == "all"
        # Consider all cameras
        if not frame_counts:
            raise ValueError("No cameras found in the dataset")

        # Find camera with FEWEST frames (minimum, not maximum)
        min_frames = min(frame_counts.values())
        candidates = [cam for cam, count in frame_counts.items() if count == min_frames]

        # If multiple cameras have min frames, use first alphabetically
        return sorted(candidates)[0]


def find_best_match_in_window(
    query_timestamp: int,
    search_timetable: Dict[int, int],
    search_center: int,
    search_window: int,
    max_search_frame: int
) -> Tuple[int, float]:
    """
    Find the best matching frame in the search video within a search window.

    Args:
        query_timestamp: Timestamp (ns) of the query frame
        search_timetable: Timetable of video to search in (frame -> timestamp_ns)
        search_center: Center frame index for the search window (typically same as query frame idx)
        search_window: Search radius (searches [center-window, center+window])
        max_search_frame: Maximum valid frame index in search video

    Returns:
        Tuple of (best_match_frame_idx, time_diff_ms)
        - best_match_frame_idx: Index of search video frame with smallest time difference
        - time_diff_ms: Time difference in milliseconds (absolute value)

    Note:
        The search window is clamped to valid frame indices [0, max_search_frame].
        Time difference is computed as |query_timestamp - search_timestamp|.

        Tiebreaker logic when multiple frames have the same minimum time difference:
        1. Choose the frame closest to search_center (minimum index distance)
        2. If equally close (e.g., center-2 and center+2), prefer the backward one (center-2)

        This ensures stable, predictable matching behavior.

        Special case: If search_center is beyond the search video's range,
        we clamp it and search within valid bounds.
    """
    # Define search range (clamped to valid frame indices [0, max_search_frame])
    search_start = max(0, search_center - search_window)
    search_end = min(max_search_frame, search_center + search_window)

    # Initialize best_match_idx to a valid search frame index
    # If search_center is beyond search range, use the closest valid frame
    best_match_idx = min(search_center, max_search_frame)
    min_time_diff = float('inf')
    min_index_distance = float('inf')

    # Search through the window to find best match
    for search_frame_idx in range(search_start, search_end + 1):
        if search_frame_idx not in search_timetable:
            # Skip if frame doesn't exist in search timetable
            continue

        search_timestamp = search_timetable[search_frame_idx]

        # Compute absolute time difference in nanoseconds
        time_diff_ns = abs(query_timestamp - search_timestamp)

        # Compute index distance from search center (for tiebreaking)
        index_distance = abs(search_frame_idx - search_center)

        # Determine if this is a better match
        is_better = False

        if time_diff_ns < min_time_diff:
            # Better time match - this takes priority
            is_better = True
        elif time_diff_ns == min_time_diff:
            # Same time difference - use tiebreaker
            if index_distance < min_index_distance:
                # Closer to search_center
                is_better = True
            elif index_distance == min_index_distance:
                # Same distance from center - prefer backward (earlier frame)
                # This happens when comparing k-n and k+n (same distance)
                # We prefer k-n (the backward/earlier one)
                if search_frame_idx < best_match_idx:
                    is_better = True

        if is_better:
            min_time_diff = time_diff_ns
            best_match_idx = search_frame_idx
            min_index_distance = index_distance

    # If we didn't find any valid match in the search window (shouldn't happen),
    # fall back to the closest valid search frame
    if min_time_diff == float('inf'):
        # This can happen if search window is empty or all frames are missing from timetable
        best_match_idx = min(search_center, max_search_frame)
        if best_match_idx in search_timetable:
            search_timestamp = search_timetable[best_match_idx]
            min_time_diff = abs(query_timestamp - search_timestamp)
        else:
            # Timetable is missing this frame, search for nearest available frame
            for offset in range(max_search_frame + 1):
                for candidate in [best_match_idx - offset, best_match_idx + offset]:
                    if 0 <= candidate <= max_search_frame and candidate in search_timetable:
                        best_match_idx = candidate
                        search_timestamp = search_timetable[candidate]
                        min_time_diff = abs(query_timestamp - search_timestamp)
                        break
                if min_time_diff != float('inf'):
                    break

    # Convert time difference to milliseconds (1 ms = 1e6 ns)
    time_diff_ms = min_time_diff / 1e6

    return best_match_idx, time_diff_ms


def compute_reference_frame_diffs(timetable: Dict[int, int]) -> Dict[int, float]:
    """
    Compute time differences between consecutive frames for the reference video.

    Args:
        timetable: Timetable of reference video (frame -> timestamp_ns)

    Returns:
        Dictionary mapping frame index to time difference with next frame (in ms)
        Example: {0: 132.5, 1: 89.9, 2: 95.7, ...}
        Last frame will have time_diff = 0.0

    Note:
        This provides timing information for the reference video itself.
        Useful for understanding frame rate variations.
    """
    frame_diffs = {}

    # Sort frames by index
    sorted_frames = sorted(timetable.keys())

    # Compute differences between consecutive frames
    for i in range(len(sorted_frames) - 1):
        curr_frame = sorted_frames[i]
        next_frame = sorted_frames[i + 1]

        curr_time = timetable[curr_frame]
        next_time = timetable[next_frame]

        # Time difference in milliseconds
        time_diff_ms = (next_time - curr_time) / 1e6

        frame_diffs[curr_frame] = time_diff_ms

    # Last frame has no next frame, so time_diff = 0
    if sorted_frames:
        frame_diffs[sorted_frames[-1]] = 0.0

    return frame_diffs


def synchronize_camera_to_reference(
    camera_name: str,
    camera_timetable: Dict[int, int],
    reference_name: str,
    reference_timetable: Dict[int, int],
    search_window: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Synchronize one camera's frames to the reference camera using search window.

    Args:
        camera_name: Name of the camera to synchronize
        camera_timetable: Timetable for this camera (frame -> timestamp_ns)
        reference_name: Name of the reference camera
        reference_timetable: Timetable for reference camera
        search_window: Search radius for matching frames
        verbose: Print detailed progress information

    Returns:
        Dictionary containing:
        - reference_video_name: Name of reference camera
        - matched_sequence: Dict mapping reference_frame_idx -> {best_match, time_diff}
          where best_match is the index in this camera's video
        - time_diff: Dict mapping frame_idx -> time diff to next frame (ms)

    Note:
        For each frame k in the REFERENCE video:
        1. Get timestamp of reference frame k
        2. Search frames [k-window, k+window] in THIS camera's video
        3. Find this camera's frame with minimum time difference
        4. Store the match and time difference

        The output is indexed by reference frame, showing which frame in this
        camera matches each reference frame.
    """
    num_ref_frames = len(reference_timetable)
    max_camera_frame = max(camera_timetable.keys())

    matched_sequence = {}

    # Process each frame in the REFERENCE video
    frame_iterator = tqdm(
        sorted(reference_timetable.keys()),
        desc=f"Synchronizing {camera_name}",
        disable=not verbose
    )

    for ref_frame_idx in frame_iterator:
        ref_timestamp = reference_timetable[ref_frame_idx]

        # Find best match in THIS camera's video
        # Search around frame index ref_frame_idx in the camera video
        best_match_idx, time_diff_ms = find_best_match_in_window(
            query_timestamp=ref_timestamp,
            search_timetable=camera_timetable,
            search_center=ref_frame_idx,  # Search around same frame index in camera video
            search_window=search_window,
            max_search_frame=max_camera_frame
        )

        # Store the result - indexed by REFERENCE frame
        matched_sequence[ref_frame_idx] = {
            "best_match": best_match_idx,  # Index in THIS camera's video
            "time_diff": round(time_diff_ms, 6)  # Round to microsecond precision
        }

    # Compute match statistics
    time_diffs = [match["time_diff"] for match in matched_sequence.values()]
    avg_diff = np.mean(time_diffs)
    max_diff = np.max(time_diffs)

    # Compute time differences between consecutive frames in this camera
    frame_time_diffs = compute_reference_frame_diffs(camera_timetable)

    if verbose:
        print(f"  Camera: {camera_name}")
        print(f"  Reference frames processed: {num_ref_frames}")
        print(f"  Average match time diff: {avg_diff:.3f} ms")
        print(f"  Max match time diff: {max_diff:.3f} ms")

    return {
        "reference_video_name": reference_name,
        "matched_sequence": matched_sequence,
        "time_diff": frame_time_diffs
    }


def process_single_subset(
    subset_path: Path,
    config: TimeSyncConfig,
    output_base: Path
) -> bool:
    """
    Process a single data subset (e.g., data/suturing/1/).

    Args:
        subset_path: Path to the subset folder (contains regular/ and meta/)
        config: Time synchronization configuration
        output_base: Base output folder for results

    Returns:
        True if processing was successful, False otherwise

    Note:
        Processing steps:
        1. Load timetables for all cameras from meta/ folder
        2. Count frames in regular/image/<camera>/ folders
        3. Determine reference camera based on strategy
        4. For each non-reference camera, compute synchronization
        5. For reference camera, compute inter-frame time differences
        6. Save results as JSON files in output folder
    """
    # Define paths
    meta_path = subset_path / "meta"
    image_path = subset_path / "regular" / "image"

    # Validate paths exist
    if not meta_path.exists():
        print(f"  Warning: meta folder not found in {subset_path}")
        return False

    if not image_path.exists():
        print(f"  Warning: regular/image folder not found in {subset_path}")
        return False

    # Load timetables and count frames for each camera
    timetables = {}
    frame_counts = {}

    if config.verbose:
        print(f"\n  Loading timetables and counting frames...")

    for camera_name in config.camera_names:
        # Load timetable
        timetable_file = meta_path / f"{camera_name}_timetable.json"
        if not timetable_file.exists():
            if config.verbose:
                print(f"    Warning: Timetable not found for {camera_name}, skipping")
            continue

        timetable = load_timetable_json(timetable_file)
        timetables[camera_name] = timetable

        # Count frames in image folder
        camera_image_folder = image_path / camera_name
        frame_count = count_frames_in_folder(camera_image_folder)
        frame_counts[camera_name] = frame_count

        if config.verbose:
            print(f"    {camera_name}: {frame_count} frames, {len(timetable)} timestamps")

    # Check if we have any valid cameras
    if not timetables:
        print(f"  Error: No valid cameras found in {subset_path}")
        return False

    # Determine reference camera
    reference_camera = determine_reference_camera(
        frame_counts=frame_counts,
        strategy=config.strategy,
        default_reference=config.default_reference_camera
    )

    if config.verbose:
        print(f"\n  Reference camera: {reference_camera} ({frame_counts[reference_camera]} frames)")
        print(f"  Strategy: {config.strategy}")
        print(f"  Search window: ±{config.search_window} frames")

    # Create output folder (mirror the subset structure)
    # Extract the relative path from input_folder to subset_path
    try:
        input_folder_path = Path(config.input_folder)
        relative_subset_path = subset_path.relative_to(input_folder_path)
        output_folder = output_base / relative_subset_path
    except ValueError:
        # If paths are not relative, use subset name structure
        output_folder = output_base / subset_path.name

    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each camera
    if config.verbose:
        print(f"\n  Processing cameras...")

    for camera_name, camera_timetable in timetables.items():
        if camera_name == reference_camera:
            # For reference camera, compute inter-frame time differences
            frame_diffs = compute_reference_frame_diffs(camera_timetable)

            result = {
                "reference_video_name": reference_camera,
                "time_diff": frame_diffs
            }
        else:
            # For non-reference cameras, synchronize to reference
            result = synchronize_camera_to_reference(
                camera_name=camera_name,
                camera_timetable=camera_timetable,
                reference_name=reference_camera,
                reference_timetable=timetables[reference_camera],
                search_window=config.search_window,
                verbose=config.verbose
            )

        # Save result as JSON
        output_file = output_folder / f"{camera_name}_sync_test.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        if config.verbose:
            print(f"    Saved: {output_file}")

    return True


def find_all_subsets(input_folder: Path, max_depth: int = 3) -> List[Path]:
    """
    Find all data subsets in the input folder by searching for the required structure.

    Args:
        input_folder: Root input folder (e.g., data/)
        max_depth: Maximum depth to search for subsets (default: 3)

    Returns:
        List of paths to subset folders

    Note:
        This function recursively searches for ANY folder that matches the required
        structure, regardless of folder names or hierarchy depth (up to max_depth).

        **Subset Identification Criteria (structure-based, NOT name-based):**
        A folder is considered a valid subset if it contains:
        - "meta/" folder (for timetables)
        - "regular/" folder containing ALL three subfolders:
          * "image/" (camera frames)
          * "kinematic/" (robot kinematic data)
          * "time_syn/" (time synchronization data)

        **Examples of valid subset structures:**

        Example 1 - Direct under input_folder:
           input_folder/
             1/              <- VALID SUBSET (has meta/ and regular/ with 3 subfolders)
               meta/
               regular/
                 image/
                 kinematic/
                 time_syn/

        Example 2 - Nested under task folders:
           input_folder/
             suturing/       <- NOT a subset (missing required structure)
               1/            <- VALID SUBSET
                 meta/
                 regular/
                   image/
                   kinematic/
                   time_syn/
               2/            <- VALID SUBSET
                 meta/
                 regular/
                   image/
                   kinematic/
                   time_syn/

        Example 3 - Any arbitrary nesting:
           input_folder/
             foo/
               bar/
                 baz/        <- VALID SUBSET (if it has the required structure)
                   meta/
                   regular/
                     image/
                     kinematic/
                     time_syn/

        The search is purely structure-based - folder names don't matter, only
        the presence of the required subfolders. Any folder at any depth (up to
        max_depth) that has the required structure will be identified as a subset.
    """
    subsets = []

    def _search_subsets(current_path: Path, current_depth: int):
        """
        Recursively search for subsets up to max_depth.

        Args:
            current_path: Current directory being searched
            current_depth: Current recursion depth
        """
        if current_depth > max_depth:
            return

        # Check if current folder is a valid subset
        # Criteria: has "meta/" and "regular/" folders, and within "regular/"
        # must have "image/", "kinematic/", and "time_syn/" subfolders
        meta_path = current_path / "meta"
        regular_path = current_path / "regular"

        if meta_path.exists() and regular_path.exists():
            # Check for required subfolders within regular/
            image_path = regular_path / "image"
            kinematic_path = regular_path / "kinematic"
            time_syn_path = regular_path / "time_syn"

            # All three subfolders must exist
            if image_path.exists() and kinematic_path.exists() and time_syn_path.exists():
                subsets.append(current_path)
                # Don't recurse into valid subsets (they are leaf nodes)
                return

        # Recurse into subdirectories
        try:
            for item in current_path.iterdir():
                if item.is_dir():
                    _search_subsets(item, current_depth + 1)
        except PermissionError:
            # Skip directories we don't have permission to read
            pass

    # Start search from input folder at depth 0
    _search_subsets(input_folder, 0)

    # Sort subsets for consistent processing order
    subsets.sort()

    return subsets


def run_time_sync(config: TimeSyncConfig) -> int:
    """
    Main function to run time synchronization on all subsets.

    Args:
        config: Time synchronization configuration

    Returns:
        Number of subsets successfully processed

    Note:
        This orchestrates the entire synchronization workflow:
        1. Find all data subsets in input folder
        2. Create output folder structure
        3. Process each subset
        4. Report summary statistics
    """
    input_path = Path(config.input_folder)
    output_path = Path(config.output_folder)

    # Validate input folder exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all subsets
    print(f"Searching for data subsets in {input_path}...")
    subsets = find_all_subsets(input_path)

    if not subsets:
        print(f"No data subsets found in {input_path}")
        return 0

    print(f"Found {len(subsets)} subset(s) to process")

    # Process each subset
    processed_count = 0
    failed_subsets = []

    for i, subset_path in enumerate(subsets, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(subsets)}] Processing: {subset_path}")
        print(f"{'='*70}")

        success = process_single_subset(
            subset_path=subset_path,
            config=config,
            output_base=output_path
        )

        if success:
            processed_count += 1
        else:
            failed_subsets.append(subset_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Time synchronization complete!")
    print(f"Successfully processed: {processed_count}/{len(subsets)} subsets")

    if failed_subsets:
        print(f"\nFailed subsets:")
        for subset in failed_subsets:
            print(f"  {subset}")

    print(f"\nInput folder: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"{'='*70}")

    return processed_count


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="time_sync", node=TimeSyncConfig)

# Set config path
project_root = Path(__file__).resolve().parent.parent
p_config = project_root / 'config'

# Verify config path exists
if not p_config.exists():
    raise FileNotFoundError(f"Config directory not found: {p_config}")


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_test_sync"
)
def main(cfg: DictConfig):
    """
    Main entry point for the time synchronization script.

    Args:
        cfg: Hydra configuration object (DictConfig from OmegaConf)

    Note:
        This function is decorated with @hydra.main to enable Hydra-based
        configuration management. Configuration can be overridden from command line.
    """
    print("="*70)
    print("dVRK Multi-modal Data Time Synchronization Script")
    print("="*70)
    print(f"Workspace: {cfg.workspace}")
    print(f"Input folder: {cfg.input_folder}")
    print(f"Output folder: {cfg.output_folder}")
    print(f"Video FPS: {cfg.video_fps}")
    print(f"Search window: ±{cfg.search_window} frames")
    print(f"Strategy: {cfg.strategy}")
    print(f"Camera names: {cfg.camera_names}")
    print("="*70)

    # Convert DictConfig to TimeSyncConfig
    time_sync_config = TimeSyncConfig(
        workspace=cfg.workspace,
        input_folder=cfg.input_folder,
        output_folder=cfg.output_folder,
        video_fps=cfg.video_fps,
        search_window=cfg.search_window,
        strategy=cfg.strategy,
        default_reference_camera=cfg.default_reference_camera,
        camera_names=list(cfg.camera_names),
        verbose=cfg.verbose
    )

    # Run time synchronization
    try:
        processed = run_time_sync(time_sync_config)

        if processed > 0:
            print(f"\n✓ Successfully processed {processed} subset(s)")
        else:
            print(f"\n✗ No subsets were processed")

    except Exception as e:
        print(f"\n✗ Error during time synchronization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
