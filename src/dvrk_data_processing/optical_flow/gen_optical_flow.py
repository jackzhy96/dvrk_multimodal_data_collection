from dataclasses import dataclass
from typing import Union, List, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import cv2
import time
import re
import logging
from tqdm import tqdm

# Import local utilities following the established pattern
from dvrk_data_processing.utils.hydra_config import (
    PathConfig, OpticalFlowConfig, OpticalFlowFilterConfig, OpticalFlowAlgorithmConfig
)
from dvrk_data_processing.utils.utility import create_folder, clear_folder, get_sorted_names, glob_sorted_frame


@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: OpticalFlowConfig
    workspace: str
    camera_names: List[str]


def preprocess_image_pair(img1: np.ndarray, img2: np.ndarray,
                          filter_config: OpticalFlowFilterConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image pair before optical flow calculation.
    This includes noise reduction and smoothing operations to improve flow accuracy.

    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        filter_config: Configuration for filtering operations

    Returns:
        tuple: (processed_img1, processed_img2)
    """
    # Step 1: Reduce speckle noise using bilateral filtering
    # Bilateral filter preserves edges while smoothing noise - crucial for surgical scenes
    # with fine instruments and textured backgrounds
    img1_filtered = cv2.bilateralFilter(
        img1,
        filter_config.bilateral_d,
        filter_config.bilateral_sigma_color,
        filter_config.bilateral_sigma_space
    )
    img2_filtered = cv2.bilateralFilter(
        img2,
        filter_config.bilateral_d,
        filter_config.bilateral_sigma_color,
        filter_config.bilateral_sigma_space
    )

    # Step 2: Light Gaussian smoothing for additional noise reduction
    # This helps create smoother optical flow fields while maintaining motion boundaries
    kernel_size = tuple(filter_config.gaussian_kernel_size)
    img1_smooth = cv2.GaussianBlur(img1_filtered, kernel_size, filter_config.gaussian_sigma)
    img2_smooth = cv2.GaussianBlur(img2_filtered, kernel_size, filter_config.gaussian_sigma)

    return img1_smooth, img2_smooth


def calculate_optical_flow(img1_path: Path, img2_path: Path,
                           algorithm_config: OpticalFlowAlgorithmConfig,
                           filter_config: OpticalFlowFilterConfig,
                           enable_preprocessing: bool) -> np.ndarray:
    """
    Calculate dense optical flow between two consecutive frames.

    This function implements the Farneback optical flow algorithm with configurable parameters.
    The algorithm is well-suited for surgical scenes as it provides dense flow fields
    and handles smooth motions effectively.

    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        algorithm_config: Optical flow algorithm parameters
        filter_config: Image preprocessing filter parameters
        enable_preprocessing: Whether to apply image preprocessing

    Returns:
        np.ndarray: Dense optical flow field (H, W, 2) containing (u, v) components

    Raises:
        FileNotFoundError: If input images don't exist
        ValueError: If images can't be read or have mismatched dimensions
    """
    # Validate input files exist
    if not img1_path.exists():
        raise FileNotFoundError(f"First image not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Second image not found: {img2_path}")

    # Load images - using cv2.IMREAD_COLOR first, then convert to grayscale
    # This ensures consistent handling of different image formats
    img1_color = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2_color = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)

    if img1_color is None or img2_color is None:
        raise ValueError(f"Could not read images: {img1_path}, {img2_path}")

    # Convert to grayscale for optical flow computation
    # Optical flow algorithms typically work with intensity information
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Check dimensions match
    if img1_gray.shape != img2_gray.shape:
        raise ValueError(f"Image dimensions mismatch: {img1_gray.shape} vs {img2_gray.shape}")

    # Apply preprocessing if enabled
    # Preprocessing is crucial for surgical scenes with fine details and potential noise
    if enable_preprocessing:
        img1_processed, img2_processed = preprocess_image_pair(img1_gray, img2_gray, filter_config)
    else:
        img1_processed, img2_processed = img1_gray, img2_gray

    # Calculate dense optical flow using Farneback method
    # This method is robust and provides good results for surgical scene analysis
    flow = cv2.calcOpticalFlowFarneback(
        img1_processed, img2_processed,
        None,  # No previous flow estimate - start from zero
        algorithm_config.pyramid_scale,  # Image pyramid scale factor (0.5 typical)
        algorithm_config.pyramid_levels,  # Number of pyramid levels for coarse-to-fine estimation
        algorithm_config.window_size,  # Averaging window size (larger = smoother)
        algorithm_config.iterations,  # Number of iterations at each pyramid level
        algorithm_config.poly_n,  # Size of neighborhood for polynomial expansion
        algorithm_config.poly_sigma,  # Standard deviation for polynomial expansion
        algorithm_config.flags  # Additional algorithm flags
    )

    return flow


def visualize_optical_flow(flow: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """
    Create a color-coded visualization of optical flow field.

    The visualization uses HSV color space where:
    - Hue represents flow direction (angle)
    - Saturation is fixed at maximum
    - Value represents flow magnitude (speed)

    This visualization is particularly useful for analyzing instrument motion patterns
    in surgical scenes.

    Args:
        flow: Dense optical flow field (H, W, 2)
        original_img: Original image for reference dimensions

    Returns:
        np.ndarray: Color-coded flow visualization in BGR format
    """
    # Calculate flow magnitude and angle using polar coordinates
    # This conversion helps visualize both speed and direction of motion
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV representation for intuitive color-coding
    hsv = np.zeros_like(original_img)
    hsv[..., 1] = 255  # Maximum saturation for vivid colors

    # Map flow angle to hue (0-180 degrees for OpenCV HSV)
    # Different colors represent different motion directions
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Map flow magnitude to value (brightness)
    # Brighter areas indicate faster motion
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to BGR for saving and display
    flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_visualization


def save_optical_flow_results(flow: np.ndarray, visualization: np.ndarray,
                              output_dir: Path, frame_name: str, camera_name: str,
                              flow_format: str, enable_visualization: bool):
    """
    Save optical flow results to disk in the specified format.

    Args:
        flow: Computed optical flow field
        visualization: Flow visualization image
        output_dir: Directory to save results
        frame_name: Base name for output files
        camera_name: Camera identifier (left/right)
        flow_format: Output format ('npy' or 'flo')
        enable_visualization: Whether to save visualization images

    Returns:
        dict: Timing information for save operations
    """
    timing = {}

    # Save raw optical flow data
    save_start = time.time()
    flow_filename = f"{camera_name}_{frame_name}.{flow_format}"
    flow_filepath = output_dir / flow_filename

    if flow_format.lower() == "npy":
        # Save as NumPy binary format - efficient and preserves full precision
        np.save(str(flow_filepath), flow)
    elif flow_format.lower() == "flo":
        # Save in Middlebury .flo format for compatibility with external tools
        with open(str(flow_filepath), 'wb') as f:
            # Write magic number for .flo format identification
            np.array([202021.25], dtype=np.float32).tofile(f)
            # Write dimensions (width, height)
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            # Write flow data
            flow.astype(np.float32).tofile(f)
    else:
        raise ValueError(f"Unsupported flow format: {flow_format}")

    timing['flow_save'] = time.time() - save_start

    # Save visualization if requested
    if enable_visualization:
        vis_start = time.time()
        vis_dir = output_dir / "visualization"
        create_folder(vis_dir)

        vis_filename = f"{camera_name}_{frame_name}.png"
        vis_filepath = vis_dir / vis_filename
        cv2.imwrite(str(vis_filepath), visualization)

        timing['vis_save'] = time.time() - vis_start

    return timing


def get_camera_image_sequences(input_folder: Path, camera_names: List[str]) -> dict:
    """
    Discover and organize image sequences for each camera.

    This function scans the input directory structure and creates sorted lists
    of image files for each camera. The sorting is based on numeric frame numbers
    to ensure proper temporal ordering.

    Args:
        input_folder: Base input directory containing camera subdirectories
        camera_names: List of camera identifiers to process

    Returns:
        dict: Dictionary mapping camera names to sorted lists of image paths

    Raises:
        FileNotFoundError: If input folder or camera directories don't exist
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    camera_sequences = {}

    for camera_name in camera_names:
        camera_dir = input_folder / 'image' / camera_name

        if not camera_dir.exists():
            logging.warning(f"Camera directory not found: {camera_dir}")
            camera_sequences[camera_name] = []
            continue

        # Find all image files in the camera directory
        # Support common image formats used in the dataset

        image_files_sorted = glob_sorted_frame(camera_dir)

        if not image_files_sorted:
            logging.warning(f"No images found in camera directory: {camera_dir}")
            camera_sequences[camera_name] = []
            continue

        # Sort images by frame number for proper temporal ordering
        # This is crucial for optical flow calculation which requires consecutive frames
        camera_sequences[camera_name] = image_files_sorted

        logging.info(f"Found {len(image_files_sorted)} images for {camera_name} camera")

    return camera_sequences


def process_camera_optical_flow(camera_name: str, image_sequence: List[Path],
                                output_dir: Path, optical_flow_config: OpticalFlowConfig) -> dict:
    """
    Process optical flow for a single camera's image sequence.

    This function calculates optical flow between consecutive frames and handles
    all saving operations. It processes the sequence in temporal order and provides
    comprehensive progress tracking.

    Args:
        camera_name: Identifier for the camera being processed
        image_sequence: Sorted list of image file paths
        output_dir: Directory for saving results
        optical_flow_config: Configuration parameters for optical flow processing

    Returns:
        dict: Processing statistics and timing information
    """
    if len(image_sequence) < 2:
        logging.warning(f"Insufficient images for {camera_name} camera: {len(image_sequence)}")
        return {"processed_count": 0, "error_count": 0, "timing": {}}

    # Initialize tracking variables
    processed_count = 0
    error_count = 0
    timing_stats = {"flow_calculation": [], "visualization": [], "saving": []}

    logging.info(f"Processing {camera_name} camera optical flow...")

    # Process consecutive frame pairs
    # Each optical flow represents motion from frame[i] to frame[i+1]
    for i in tqdm(range(len(image_sequence) - 1), desc=f"{camera_name} camera"):
        frame1_path = image_sequence[i]
        frame2_path = image_sequence[i + 1]

        # Extract frame identifiers for output naming
        frame1_num = int(frame1_path.stem)
        frame2_num = int(frame2_path.stem)

        if frame1_num == -1 or frame2_num == -1:
            logging.warning(f"Could not extract frame numbers: {frame1_path.name}, {frame2_path.name}")
            error_count += 1
            continue

        # Generate descriptive output name indicating the frame pair
        frame_pair_name = f"flow_{frame1_num:03d}_{frame2_num:03d}"

        try:
            # Calculate optical flow between consecutive frames
            flow_start = time.time()
            flow = calculate_optical_flow(
                frame1_path, frame2_path,
                optical_flow_config.algorithm_config,
                optical_flow_config.filter_config,
                optical_flow_config.enable_preprocessing
            )
            flow_time = time.time() - flow_start
            timing_stats["flow_calculation"].append(flow_time)

            # Generate visualization if requested
            visualization = None
            if optical_flow_config.enable_visualization:
                vis_start = time.time()
                # Load original image for visualization reference
                original_img = cv2.imread(str(frame1_path), cv2.IMREAD_COLOR)
                visualization = visualize_optical_flow(flow, original_img)
                vis_time = time.time() - vis_start
                timing_stats["visualization"].append(vis_time)

            # Save results to disk
            save_timing = save_optical_flow_results(
                flow, visualization, output_dir, frame_pair_name, camera_name,
                optical_flow_config.flow_format, optical_flow_config.enable_visualization
            )

            # Accumulate timing statistics
            if 'flow_save' in save_timing:
                timing_stats["saving"].append(save_timing['flow_save'])

            processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process {camera_name} frames {frame1_num}-{frame2_num}: {e}")
            error_count += 1
            continue

    # Calculate summary statistics
    total_possible = len(image_sequence) - 1
    success_rate = (processed_count / total_possible * 100) if total_possible > 0 else 0

    return {
        "processed_count": processed_count,
        "error_count": error_count,
        "total_possible": total_possible,
        "success_rate": success_rate,
        "timing": timing_stats
    }


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="optical_flow", node=AppCfg)

# Set config path relative to the project structure
config_path = Path(__file__).resolve().parents[3] / 'config'


@hydra.main(
    version_base=None,
    config_path=str(config_path),
    config_name="config_of_jack"  # Default config name
)
def main(cfg: AppCfg):
    """
    Main processing function that orchestrates the optical flow calculation pipeline.

    This follows the established pattern used in other processing scripts:
    1. Parse and validate configuration
    2. Set up input/output directories
    3. Discover image sequences for each camera
    4. Process optical flows for both cameras
    5. Save results and log comprehensive statistics
    """
    # Extract configuration parameters following the established pattern
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)
    camera_names = cfg.camera_names  # Now comes from main config following the pattern

    # Validate input directory exists
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Initialize output directory, clearing if requested
    if cfg.preprocess.folder_initialize:
        if output_folder.exists():
            clear_folder(output_folder)

    create_folder(output_folder)

    # Log configuration information for debugging and reproducibility
    logging.info("=" * 60)
    logging.info("OPTICAL FLOW PROCESSING")
    logging.info("=" * 60)
    logging.info(f"Input directory: {input_folder}")
    logging.info(f"Output directory: {output_folder}")
    logging.info(f"Cameras to process: {camera_names}")
    logging.info(f"Flow format: {cfg.preprocess.flow_format}")
    logging.info(f"Visualization: {'Enabled' if cfg.preprocess.enable_visualization else 'Disabled'}")
    logging.info(f"Preprocessing: {'Enabled' if cfg.preprocess.enable_preprocessing else 'Disabled'}")

    # Log algorithm parameters for reproducibility
    algo_cfg = cfg.preprocess.algorithm_config
    logging.info(f"Algorithm parameters:")
    logging.info(f"  - Pyramid scale: {algo_cfg.pyramid_scale}")
    logging.info(f"  - Pyramid levels: {algo_cfg.pyramid_levels}")
    logging.info(f"  - Window size: {algo_cfg.window_size}")
    logging.info(f"  - Iterations: {algo_cfg.iterations}")
    logging.info("=" * 60)

    # Discover image sequences for each camera
    logging.info("Discovering image sequences...")
    camera_sequences = get_camera_image_sequences(input_folder, camera_names)

    # Validate that we have sequences to process
    total_images = sum(len(seq) for seq in camera_sequences.values())
    if total_images == 0:
        logging.error("No image sequences found to process")
        return

    # Process optical flows for each camera
    logging.info(f"Processing optical flows for {len(camera_names)} cameras...")

    # Initialize overall timing and statistics tracking
    total_start_time = time.time()
    processing_results = {}

    # Process each camera independently
    # This allows for different numbers of images per camera and independent error handling
    for camera_name in camera_names:
        image_sequence = camera_sequences.get(camera_name, [])

        if not image_sequence:
            logging.warning(f"No images found for {camera_name} camera, skipping")
            continue

        if len(image_sequence) < 2:
            logging.warning(f"Insufficient images for {camera_name} camera: {len(image_sequence)}")
            continue

        # Process this camera's optical flow sequence
        camera_results = process_camera_optical_flow(
            camera_name, image_sequence, output_folder, cfg.preprocess
        )
        processing_results[camera_name] = camera_results

    # Calculate and log comprehensive summary statistics
    total_time = time.time() - total_start_time

    logging.info("")
    logging.info("=" * 60)
    logging.info("OPTICAL FLOW PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total processing time: {total_time:.2f}s")
    logging.info(f"Results saved to: {output_folder}")

    # Log detailed statistics for each camera
    total_flows_generated = 0
    total_errors = 0

    for camera_name, results in processing_results.items():
        flows_count = results["processed_count"]
        error_count = results["error_count"]
        success_rate = results["success_rate"]

        total_flows_generated += flows_count
        total_errors += error_count

        logging.info(f"{camera_name.upper()} camera results:")
        logging.info(f"  - Optical flows generated: {flows_count}")
        logging.info(f"  - Processing errors: {error_count}")
        logging.info(f"  - Success rate: {success_rate:.1f}%")

        # Log timing statistics if available
        timing = results.get("timing", {})
        if timing.get("flow_calculation"):
            avg_flow_time = sum(timing["flow_calculation"]) / len(timing["flow_calculation"])
            logging.info(f"  - Average flow calculation time: {avg_flow_time:.3f}s")
        if timing.get("saving"):
            avg_save_time = sum(timing["saving"]) / len(timing["saving"])
            logging.info(f"  - Average save time: {avg_save_time:.3f}s")

    logging.info(f"SUMMARY: Generated {total_flows_generated} optical flows with {total_errors} errors")

    if cfg.preprocess.enable_visualization:
        vis_dir = output_folder / "visualization"
        logging.info(f"Visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
    print('Optical Flow Processing Complete!')