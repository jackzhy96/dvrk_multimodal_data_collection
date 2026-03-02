"""
Script to extract frames from a video file and optionally save frame timestamps.
Uses FFmpeg as the primary extractor (faster, more accurate) with OpenCV as a fallback.
Configuration managed via Hydra.

Extraction modes:
  - extract_fps == -1  → extract every frame from the video
  - extract_fps > 0    → extract at the specified fps (e.g., 10 → one frame every 0.1s)
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
import subprocess
import shutil
import re
import time

from dvrk_data_processing.utils.hydra_config import VideoToImageConfig, PathConfig
from dvrk_data_processing.utils.utility import glob_sorted_frame, create_folder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _check_ffmpeg_available() -> bool:
    """Return True if the ffmpeg binary is on PATH and executable."""
    return shutil.which("ffmpeg") is not None


def _get_video_fps_ffprobe(video_path: Path) -> Optional[float]:
    """
    Use ffprobe to get the video's frame rate.
    Returns None if ffprobe is unavailable or fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        # r_frame_rate comes back as "num/den" e.g. "30/1"
        fps_str = result.stdout.decode().strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        return float(fps_str)
    except Exception:
        return None


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


# ---------------------------------------------------------------------------
# FFmpeg-based extraction (primary)
# ---------------------------------------------------------------------------

def _parse_ffmpeg_progress(line: str) -> Optional[int]:
    """
    Extract the current frame number from an FFmpeg stderr progress line.

    FFmpeg periodically prints lines like:
      frame=  100 fps= 25 q=28.0 size=  256kB time=00:00:04.00 ...
    Returns the frame number if found, else None.
    """
    m = re.search(r'frame=\s*(\d+)', line)
    return int(m.group(1)) if m else None


def _extract_frames_ffmpeg(
    video_path: Path,
    output_image_path: Path,
    file_extension: str,
    extract_fps: float,
    total_frames: int = 0,
) -> int:
    """
    Extract frames from a video using FFmpeg.
    Shows a real-time progress bar by streaming FFmpeg's stderr.

    Args:
        video_path:        Input video file.
        output_image_path: Directory where extracted frames will be saved.
        file_extension:    Image format (png, jpg, etc.).
        extract_fps:       Target fps for extraction (-1 = all frames).
        total_frames:      Expected total frames (for progress bar; 0 = unknown).

    Returns:
        Number of frames extracted.
    """
    # Build the output pattern — FFmpeg %d gives 0-based numbering
    output_pattern = str(output_image_path / f"%d.{file_extension}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
    ]

    if extract_fps > 0:
        # Extract at a specific fps using the fps video filter
        cmd += ["-vf", f"fps={extract_fps}"]
    else:
        # Extract every frame; -vsync 0 avoids frame duplication/dropping
        cmd += ["-vsync", "0"]

    # Set quality for output images
    if file_extension.lower() in ("png",):
        cmd += ["-compression_level", "3"]  # moderate PNG compression
    elif file_extension.lower() in ("jpg", "jpeg"):
        cmd += ["-q:v", "2"]  # high quality JPEG

    cmd.append(output_pattern)

    # --- Run FFmpeg with real-time progress tracking ---
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Use total_frames for the bar if known; otherwise indeterminate
    bar_total = total_frames if total_frames > 0 else None
    pbar = tqdm(total=bar_total, desc="[FFmpeg] Extracting frames", unit="frame")
    stderr_lines: List[str] = []

    # FFmpeg writes progress to stderr; lines terminated by \r not \n
    buf = ""
    while True:
        ch = proc.stderr.read(1)
        if not ch:
            break
        ch = ch.decode("utf-8", errors="replace")

        if ch in ('\r', '\n'):
            stderr_lines.append(buf)
            frame_num = _parse_ffmpeg_progress(buf)
            if frame_num is not None:
                pbar.n = frame_num
                pbar.refresh()
            buf = ""
        else:
            buf += ch

    proc.wait()
    pbar.close()

    if proc.returncode != 0:
        # Print last few stderr lines for diagnosis
        error_msg = "\n".join(
            line.strip() for line in stderr_lines[-10:] if line.strip()
        )
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stderr=error_msg.encode()
        )

    # Count how many frames were actually written
    extracted = list(output_image_path.glob(f"*.{file_extension}"))
    return len(extracted)


# ---------------------------------------------------------------------------
# OpenCV-based extraction (fallback)
# ---------------------------------------------------------------------------

def _extract_frames_opencv(
    video_path: Path,
    output_image_path: Path,
    file_extension: str,
    extract_fps: float,
) -> int:
    """
    Extract frames from a video using OpenCV (fallback when FFmpeg is unavailable).

    Args:
        video_path:        Input video file.
        output_image_path: Directory where extracted frames will be saved.
        file_extension:    Image format (png, jpg, etc.).
        extract_fps:       Target fps for extraction (-1 = all frames).

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute frame step: how many source frames to skip per extracted frame
    if extract_fps > 0 and video_fps > 0:
        frame_step = video_fps / extract_fps
    else:
        frame_step = 1.0  # extract every frame

    frame_count = 0        # index into the source video
    extracted_count = 0    # number of frames actually saved
    next_extract_at = 0.0  # next source-frame index to extract

    with tqdm(total=total_frames, desc="[OpenCV] Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count >= next_extract_at:
                filename = f"{extracted_count}.{file_extension}"
                output_file = output_image_path / filename
                cv2.imwrite(str(output_file), frame)
                extracted_count += 1
                next_extract_at += frame_step

            frame_count += 1
            pbar.update(1)

    cap.release()
    return extracted_count


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_frames_from_video(config: AppCfg) -> None:
    """
    Extract frames from a video file and optionally save timestamps.

    Tries FFmpeg first for speed and accuracy; falls back to OpenCV.

    Args:
        config: Hydra application config.
    """
    video_path = Path(config.video_path)
    output_path = Path(config.output_path)

    # --- Prepare output directories ---
    if not output_path.exists():
        create_folder(output_path)

    output_image_path = output_path / config.image_folder
    if not output_image_path.exists():
        create_folder(output_image_path)

    # Timestamp output directory
    if config.timestamp_folder is not None:
        output_timestamp_path = output_path / config.video_name / config.timestamp_folder
    else:
        output_timestamp_path = output_path / config.video_name / 'time_syn'

    if config.preprocess.enable_timestamp:
        if not output_timestamp_path.exists():
            create_folder(output_timestamp_path)

    # --- Validate input ---
    validate_video_file(video_path)

    # --- Gather video metadata ---
    # Try ffprobe first for accurate fps, fall back to OpenCV
    fps = _get_video_fps_ffprobe(video_path)
    if fps is None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    duration = total_frames / fps if fps > 0 else 0.0
    extract_fps = config.preprocess.extract_fps

    print(f"Input video: {video_path}")
    print(f"Output directory: {output_path}")
    print(f"File extension: {config.preprocess.file_extension}")
    print(f"Extract timestamps: {config.preprocess.enable_timestamp}")
    print(f"Video properties:  fps={fps:.2f}  frames={total_frames}  duration={duration:.2f}s")
    if extract_fps > 0:
        print(f"Target extraction fps: {extract_fps}")
    else:
        print("Extracting all frames")

    if config.preprocess.enable_timestamp:
        print(f"Timestamp output: {output_timestamp_path}")

    # --- Extract frames ---
    use_ffmpeg = _check_ffmpeg_available()
    extracted_count = 0
    t_start = time.time()

    if use_ffmpeg:
        print("Extracting with FFmpeg")
        try:
            extracted_count = _extract_frames_ffmpeg(
                video_path, output_image_path,
                config.preprocess.file_extension, extract_fps,
                total_frames=total_frames,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: FFmpeg extraction failed: {e.stderr.decode()}")
            print("Falling back to OpenCV")
            use_ffmpeg = False  # trigger fallback below

    if not use_ffmpeg:
        print("Extracting with OpenCV (fallback)")
        extracted_count = _extract_frames_opencv(
            video_path, output_image_path,
            config.preprocess.file_extension, extract_fps,
        )

    elapsed = time.time() - t_start
    rate = extracted_count / elapsed if elapsed > 0 else 0
    print(f"Extracted {extracted_count} frames in {elapsed:.1f}s  ({rate:.1f} frames/s)")

    # --- Save per-frame timestamps ---
    if config.preprocess.enable_timestamp and extracted_count > 0:
        # Determine the effective fps for timestamp computation
        if extract_fps > 0:
            effective_fps = extract_fps
        else:
            effective_fps = fps if fps > 0 else 30.0  # safe default

        for frame_idx in range(extracted_count):
            timestamp = frame_idx / effective_fps
            save_timestamp_as_json(timestamp, frame_idx, output_timestamp_path)

        print(f"Timestamps saved as individual JSON files (compatible with image-to-video script)")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

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
