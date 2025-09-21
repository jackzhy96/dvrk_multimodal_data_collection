"""
Script to assemble sequential images into a video with timestamp-based or fixed frame rate timing.
Uses OpenCV for video processing and Hydra for configuration management.
"""

from dataclasses import dataclass
from typing import Union, List, Optional
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import os
import json

from dvrk_data_processing.utils.hydra_config import ImageToVideoConfig
from dvrk_data_processing.utils.utility import glob_sorted_frame, create_folder
from dvrk_data_processing.utils.hydra_config import PathConfig, ImageToVideoConfig


@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: ImageToVideoConfig
    video_name: str
    workspace: Union[str, Path]
    image_path: Union[str, Path]
    timestamp_path: Union[str, Path]
    save_folder: Union[str, Path]


def load_timestamps(timestamp_path: Path) -> np.ndarray:
    """
    Load timestamps from file or directory. Supports multiple formats:
    - Directory with individual JSON files containing timestamps

    Args:
        timestamp_path: Path to timestamp file or directory

    Returns:
        Array of timestamps in seconds
    """
    if timestamp_path.is_dir():
        # Handle directory with individual JSON timestamp files
        # Get all JSON files and sort them numerically
        json_files = glob_sorted_frame(timestamp_path)

        timestamps = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    timestamp_str = data.get('timestamp', '')

                    # Parse timestamp format: "seconds_nanoseconds"
                    if '_' in timestamp_str:
                        seconds_str, nanoseconds_str = timestamp_str.split('_')
                        seconds = float(seconds_str)
                        nanoseconds = float(nanoseconds_str)
                        # Convert to total seconds
                        total_seconds = seconds + (nanoseconds / 1e9)
                        timestamps.append(total_seconds)
                    else:
                        # Fallback: try to parse as float
                        timestamps.append(float(timestamp_str))

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse timestamp from {json_file}: {e}")
                continue

        timestamps = np.array(timestamps)
    else:
        raise ValueError(f"Timestamp path must be a file or directory: {timestamp_path}")

    return timestamps


def calculate_frame_delays(timestamps: np.ndarray, fps: float) -> np.ndarray:
    """
    Calculate frame delays from timestamps.

    Args:
        timestamps: Array of timestamps in seconds
        fps: Frame rate

    Returns:
        Array of frame delays in seconds
    """
    if len(timestamps) < 2:
        return np.array([1.0/fps])  # Default to 30fps if only one frame

    # Calculate differences between consecutive timestamps
    delays = np.diff(timestamps)

    # Handle edge cases where delays might be negative or zero
    delays = np.maximum(delays, 1e-6)  # Minimum delay of 1 microsecond

    # For the first frame, use the first calculated delay
    frame_delays = np.concatenate([[delays[0]], delays])

    return frame_delays


def create_video_from_images(config: AppCfg) -> None:
    """
    Create video from sequential images.
    config: ImageToVideoConfig object
    output: saved video file
    """

    image_path = Path(config.image_path) 
    output_path = Path(config.save_folder)

    # Validate input directory
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_path}")

    if not image_path.is_dir():
        raise ValueError(f"Image path is not a directory: {image_path}")

    # Create output directory if it doesn't exist
    if not output_path.exists():
        create_folder(output_path)

    # Get sorted image files
    image_files = glob_sorted_frame(image_path)

    print(f"Found {len(image_files)} total images")

    # Apply frame range selection based on start_frame and end_frame
    start_idx = 0 if config.preprocess.start_frame == -1 else config.preprocess.start_frame
    end_idx = len(image_files) if config.preprocess.end_frame == -1 else config.preprocess.end_frame + 1

    # Ensure indices are within valid range
    start_idx = max(0, min(start_idx, len(image_files) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(image_files)))

    # Slice the image files list
    image_files = image_files[start_idx:end_idx]

    print(f"Processing frames {start_idx} to {end_idx - 1} ({len(image_files)} images)")

    # Load timestamps if not using fixed rate
    timestamps = None
    if not config.preprocess.enable_fixed_rate:
        if config.timestamp_path is None:
            raise ValueError("timestamp_path is required when enable_fixed_rate is False")

        timestamp_path = Path(config.timestamp_path)
        if not timestamp_path.exists():
            raise FileNotFoundError(f"Timestamp file not found: {timestamp_path}")

        timestamps = load_timestamps(timestamp_path)
        print(f"Loaded {len(timestamps)} total timestamps")

        # Apply the same frame range selection to timestamps
        if len(timestamps) > 0:
            # Ensure we don't go out of bounds for timestamps
            timestamp_start_idx = max(0, min(start_idx, len(timestamps) - 1))
            timestamp_end_idx = max(timestamp_start_idx + 1, min(end_idx, len(timestamps)))
            timestamps = timestamps[timestamp_start_idx:timestamp_end_idx]
            print(f"Using timestamps {timestamp_start_idx} to {timestamp_end_idx - 1} ({len(timestamps)} timestamps)")

    if not image_files:
        raise ValueError("No image files found")

    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")

    original_height, original_width = first_image.shape[:2]

    # Determine output dimensions
    if config.preprocess.resize_config.enable_resize:
        width, height = config.preprocess.resize_config.new_size
    else:
        width, height = original_width, original_height

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*config.preprocess.codec)

    output_file = output_path / f"{config.video_name}.mp4"

    if config.preprocess.enable_fixed_rate:
        # Use fixed frame rate
        video_writer = cv2.VideoWriter(str(output_file), fourcc, config.preprocess.video_fixed_rate, (width, height))

        print(f"Creating video with fixed frame rate: {config.preprocess.video_fixed_rate} fps")
        for image_file in tqdm(image_files, desc="Processing images"):
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Warning: Could not read image {image_file}, skipping...")
                continue

            if config.preprocess.resize_config.enable_resize:
                image = cv2.resize(image, (width, height))

            video_writer.write(image)
    else:
        # Use timestamp-based timing
        if timestamps is None:
            raise ValueError("Timestamps required when enable_fixed_rate is False")

        if len(timestamps) != len(image_files):
            raise ValueError(f"Number of timestamps ({len(timestamps)}) doesn't match number of images ({len(image_files)})")

        # Calculate frame delays
        frame_delays = calculate_frame_delays(timestamps, config.preprocess.video_fixed_rate)

        # Calculate average fps for video writer initialization
        total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        avg_fps = len(image_files) / total_duration if total_duration > 0 else config.preprocess.video_fixed_rate
        avg_fps = max(0.5, min(avg_fps, 120.0))  # Clamp between 1 and 120 fps
        print(f"Average fps: {avg_fps:.2f}")

        video_writer = cv2.VideoWriter(str(output_file), fourcc, avg_fps, (width, height))

        print(f"Creating video with timestamp-based timing (avg fps: {avg_fps:.2f})")

        for i, (image_file, delay) in enumerate(tqdm(zip(image_files, frame_delays), desc="Processing images", total=len(image_files))):
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Warning: Could not read image {image_file}, skipping...")
                continue

            if config.preprocess.resize_config.enable_resize:
                image = cv2.resize(image, (width, height))

            # Calculate how many times to repeat this frame based on delay
            # This is a simplification - for more accurate timing, consider using variable frame rates
            repeat_count = max(1, int(delay * avg_fps))

            for _ in range(repeat_count):
                video_writer.write(image)

    video_writer.release()
    print(f"Video saved to: {output_file}")


cs = ConfigStore.instance()
cs.store(name="image_to_video", node=AppCfg)

# Set config path
p_config = Path.cwd().parents[0] / 'config'


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_i2v_jack"
)
def main(cfg: AppCfg) -> None:
    """
    Main function to convert images to video using hydra configuration.

    Args:
        cfg: ImageToVideoConfig object
    """
    # Create video
    create_video_from_images(cfg)


if __name__ == '__main__':
    main()
    print('Image to video conversion completed!')