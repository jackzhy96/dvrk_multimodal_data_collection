"""
Script to extract frames from a video file and optionally save frame timestamps.
Uses OpenCV for video processing and Hydra for configuration management.
This is the reverse process of converting images to video.
"""

from dataclasses import dataclass
from typing import Union, List, Optional
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json

from dvrk_data_processing.utils.hydra_config import VideoToImageConfig, PathConfig
from dvrk_data_processing.utils.utility import glob_sorted_frame, create_folder

@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: VideoToImageConfig
    workspace: Union[str, Path]
    video_name: str
    video_path: Union[str, Path]
    output_path: Union[str, Path]
    image_folder: str
    timestamp_folder: str

def save_timestamp_as_json(timestamp: float, frame_number: int, output_dir: Path) -> None:
    """
    Save timestamp as individual JSON file in the format expected by image-to-video script.

    Args:
        timestamp: Timestamp in seconds
        frame_number: Frame number for the filename
        output_dir: Directory to save the JSON file
    """
    # Convert timestamp to seconds_nanoseconds format
    seconds = int(timestamp)
    nanoseconds = int((timestamp - seconds) * 1e9)
    timestamp_str = f"{seconds}_{nanoseconds}"

    # Create JSON data in the expected format
    json_data = {"timestamp": timestamp_str}

    # Save to JSON file with frame number as filename
    json_file = output_dir / f"{frame_number}.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=3)


def extract_frames_from_video(config: AppCfg) -> None:
    """
    Extract frames from video and optionally save timestamps.

    config: VideoToImageConfig object

    output: saved image files
    """
    # Convert paths to Path objects
    video_path = Path(config.video_path)
    output_path = Path(config.output_path)

    if not output_path.exists():
        create_folder(output_path)

    output_image_path = output_path / config.image_folder
    if not output_image_path.exists():
        create_folder(output_image_path)

    # Parse timestamp output path
    if config.timestamp_folder is not None:
        output_timestamp_path = output_path / config.video_name / config.timestamp_folder
    else:
        output_timestamp_path = output_path / config.video_name /'time_syn'

    # Create timestamp directory if extracting timestamps
    if config.preprocess.enable_timestamp:
        if not output_timestamp_path.exists():
            create_folder(output_timestamp_path)


    # Validate input video file
    validate_video_file(video_path)

    print(f"Input video: {video_path}")
    print(f"Output directory: {output_path}")
    print(f"File extension: {config.preprocess.file_extension}")
    print(f"Extract timestamps: {config.preprocess.enable_timestamp}")

    if config.preprocess.enable_timestamp:
        print(f"Timestamp output: {output_timestamp_path}")

    video_capture = cv2.VideoCapture(str(video_path))

    if not video_capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video properties:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")

    # Initialize timestamp storage and create timestamp directory if needed
    timestamps = []
    frame_count = 0


    print(f"Extracting frames to: {output_image_path}")
    if config.preprocess.enable_timestamp:
        print(f"Extracting timestamps to: {output_timestamp_path}")

    # Process video frame by frame
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            # Calculate timestamp in seconds
            timestamp = frame_count / fps if fps > 0 else frame_count

            # Save timestamp as individual JSON file if extracting timestamps
            if config.preprocess.enable_timestamp:
                save_timestamp_as_json(timestamp, frame_count, output_timestamp_path)
                timestamps.append(timestamp)

            # Generate output filename with frame number (matching the reference data format)
            filename = f"{frame_count}.{config.preprocess.file_extension}"
            output_file= output_image_path / filename

            # Save frame
            success = cv2.imwrite(str(output_file), frame)
            if not success:
                print(f"Warning: Failed to save frame {frame_count}")

            frame_count += 1
            pbar.update(1)

    video_capture.release()

    print(f"Extracted {frame_count} frames")

    if config.preprocess.enable_timestamp and len(timestamps) > 0:
        print(f"Timestamps saved as individual JSON files (compatible with image-to-video script)")


def validate_video_file(video_path: Path) -> None:
    """
    Validate that the video file exists and can be opened.

    Args:
        video_path: Path to video file

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file cannot be opened
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    # Try to open the video to validate it
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        video_capture.release()
        raise ValueError(f"Cannot open video file (possibly corrupted or unsupported format): {video_path}")

    video_capture.release()


cs = ConfigStore.instance()
cs.store(name="video_to_image_custom", node=AppCfg)

# Set config path
# p_config = Path.cwd() / 'config'
p_config = Path.cwd().parents[0] / 'config'


@hydra.main(
    version_base=None,
    config_path=str(p_config),
    config_name="config_v2i_jack"
)
def main(cfg: AppCfg) -> None:
    """
    Main function to extract frames from video using hydra configuration.

    Args:
        cfg: AppCfg object
    """

    # Extract frames
    extract_frames_from_video(cfg)


if __name__ == '__main__':
    main()
    print('Video to image extraction completed!')