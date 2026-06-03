from dataclasses import dataclass
from typing import Union, List, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import sys
import torch
import cv2
import time
import re
import json
import logging
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf
from dvrk_data_processing.utils.hydra_config import PathConfig, DepthEstimationConfig
from dvrk_data_processing.utils.utility import create_folder, clear_folder, get_sorted_names, glob_sorted_frame
# Depth helpers — pulled into a small dependency-light module so unit tests
# don't need the full FoundationStereo / tqdm / torch stack just to validate
# the numeric conversion.
from dvrk_data_processing.depth_estimation.depth_utils import (
    load_stereo_depth_params, disparity_to_depth_m, colorize_depth_m,
)


def setup_foundation_stereo_imports():
    '''
    Set up FoundationStereo dependencies for depth estimation.
    '''
    # Get the project root directory (three levels up from this script)
    project_root = Path(__file__).resolve().parents[3]  # Goes up from src/dvrk_data_processing/depth_estimation/
    foundation_stereo_path = project_root / 'FoundationStereo'

    # Add FoundationStereo to Python path if not already present
    foundation_stereo_str = str(foundation_stereo_path)
    if foundation_stereo_str not in sys.path:
        sys.path.append(foundation_stereo_str)

    # Import required modules - using try/except for graceful error handling
    try:
        from omegaconf import OmegaConf
        from core.utils.utils import InputPadder
        from Utils import vis_disparity, set_logging_format, set_seed
        from core.foundation_stereo import FoundationStereo
        return True, (OmegaConf, InputPadder, vis_disparity, set_logging_format, set_seed, FoundationStereo)
    except ImportError as e:
        logging.error(f"Failed to import FoundationStereo dependencies: {e}")
        return False, None


@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: DepthEstimationConfig
    workspace: str
    camera_names: List[str]


def load_and_prepare_model(model_path: Path, config_overrides: dict = None):
    """
    Load the FoundationStereo model with proper configuration.

    Args:
        model_path: Path to the pretrained model checkpoint
        config_overrides: Dictionary of configuration overrides

    Returns:
        tuple: (model, config) - loaded model and its configuration
    """
    # Import dependencies
    success, modules = setup_foundation_stereo_imports()
    if not success:
        raise ImportError("Failed to import FoundationStereo dependencies")

    OmegaConf_mod, InputPadder, vis_disparity, set_logging_format, set_seed, FoundationStereo = modules

    # Set up logging and random seed for reproducible results
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)  # Disable gradients for inference

    # Load model configuration
    config_path = model_path.parent / 'cfg.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")

    cfg = OmegaConf_mod.load(str(config_path))

    # Set default values if missing
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'

    # Apply configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            cfg[key] = value

    logging.info(f"Model configuration: {cfg}")
    logging.info(f"Loading pretrained model from {model_path}")

    # Initialize and load model
    model = FoundationStereo(cfg)

    # Load checkpoint
    checkpoint = torch.load(str(model_path))
    logging.info(f"Checkpoint info - global_step: {checkpoint.get('global_step', 'unknown')}, "
                f"epoch: {checkpoint.get('epoch', 'unknown')}")

    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    return model, cfg


def process_stereo_pair(model, left_img_path: Path, right_img_path: Path,
                       scale: float, hierarchical: bool, valid_iters: int,
                       padder_divis_by: int = 32,
                       hierarchical_small_ratio: float = 0.5):
    """
    Process a single stereo image pair to generate depth estimation.

    Args:
        model: Loaded FoundationStereo model
        left_img_path: Path to left camera image
        right_img_path: Path to right camera image
        scale: Image scaling factor (must be <= 1)
        hierarchical: Whether to use hierarchical inference
        valid_iters: Number of inference iterations

    Returns:
        tuple: (disparity_map, processing_times) - disparity map and timing info
    """
    success, modules = setup_foundation_stereo_imports()
    if not success:
        raise ImportError("Failed to import FoundationStereo dependencies")

    OmegaConf_mod, InputPadder, vis_disparity, set_logging_format, set_seed, FoundationStereo = modules

    # Check if both images exist
    if not left_img_path.exists():
        raise FileNotFoundError(f"Left image not found: {left_img_path}")
    if not right_img_path.exists():
        raise FileNotFoundError(f"Right image not found: {right_img_path}")

    timing = {}

    # Load and preprocess images
    load_start = time.time()
    img_left = cv2.imread(str(left_img_path), cv2.IMREAD_COLOR)
    img_right = cv2.imread(str(right_img_path), cv2.IMREAD_COLOR)
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    timing['load'] = time.time() - load_start

    # Apply scaling if specified
    resize_start = time.time()
    assert scale <= 1, "Scale must be <= 1"
    if scale < 1:
        img_left = cv2.resize(img_left, fx=scale, fy=scale, dsize=None)
        img_right = cv2.resize(img_right, fx=scale, fy=scale, dsize=None)
    timing['resize'] = time.time() - resize_start

    # Store original for visualization
    img_left_original = img_left.copy()
    H, W = img_left.shape[:2]

    # Convert to tensors and prepare for GPU processing
    tensor_start = time.time()
    img_left_tensor = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    img_right_tensor = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)

    # Pad images to ensure compatible dimensions for the model
    padder = InputPadder(img_left_tensor.shape, divis_by=padder_divis_by, force_square=False)
    img_left_padded, img_right_padded = padder.pad(img_left_tensor, img_right_tensor)
    timing['tensor_prep'] = time.time() - tensor_start

    # Model inference - core depth estimation computation
    inference_start = time.time()
    with torch.cuda.amp.autocast(True):
        if not hierarchical:
            disparity = model.forward(img_left_padded, img_right_padded,
                                    iters=valid_iters, test_mode=True)
        else:
            # Use hierarchical inference for high-resolution images
            disparity = model.run_hierachical(img_left_padded, img_right_padded,
                                            iters=valid_iters, test_mode=True,
                                            small_ratio=hierarchical_small_ratio)
    timing['inference'] = time.time() - inference_start

    # Post-process results
    post_start = time.time()
    disparity = padder.unpad(disparity.float())
    disparity = disparity.data.cpu().numpy().reshape(H, W)
    timing['post_process'] = time.time() - post_start

    # Clean up GPU memory to prevent accumulation
    cleanup_start = time.time()
    torch.cuda.empty_cache()
    timing['cleanup'] = time.time() - cleanup_start

    return disparity, img_left_original, timing


def save_depth_results(disparity: np.ndarray, original_img: np.ndarray,
                      output_dir: Path, frame_number: int,
                      save_depth: bool, save_visualization: bool,
                      depth_m: np.ndarray = None,
                      depth_viz_range_m: Tuple[float, float] = (0.02, 0.5),
                      depth_viz_cmap: str = "turbo"):
    """
    Save depth estimation results to disk.

    Args:
        disparity:          Computed disparity map (pixels, post-scale resolution)
        original_img:       Original left camera image for visualization
        output_dir:         Directory to save results (processed_dir/depth_estimation/)
        frame_number:       Frame number for file naming
        save_depth:         Whether to save raw disparity .npy (legacy flag name —
                            controls disparity/<i>.npy only; the depth output
                            has its own knob, ``compute_depth``)
        save_visualization: Whether to save the colored .png files (disparity_image/,
                            combined_image/, depth_image/)
        depth_m:            Optional precomputed depth-in-meters array (float32, NaN-masked).
                            When provided, written to depth/<i>.npy and colorized to
                            depth_image/<i>.png (the latter gated by save_visualization).
                            When None the depth outputs are skipped.
        depth_viz_range_m:  (min_m, max_m) for the depth_image/ colormap.
    """
    # Import vis_disparity from the modules
    success, modules = setup_foundation_stereo_imports()
    if not success:
        raise ImportError("Failed to import FoundationStereo dependencies")

    OmegaConf_mod, InputPadder, vis_disparity, set_logging_format, set_seed, FoundationStereo = modules

    # Create frame-specific output directory
    save_start = time.time()

    # Disparity output (legacy — name kept for backwards compatibility).
    disparity_output_dir = output_dir / "disparity"
    if not disparity_output_dir.exists():
        create_folder(disparity_output_dir)

    img_output_dir = output_dir / "disparity_image"
    vis_img_output_dir = output_dir / "combined_image"

    if not img_output_dir.exists():
        create_folder(img_output_dir)
    if not vis_img_output_dir.exists():
        create_folder(vis_img_output_dir)

    # Depth output directories. Only created when depth_m is passed.
    if depth_m is not None:
        depth_npy_dir = output_dir / "depth"
        depth_png_dir = output_dir / "depth_image"
        if not depth_npy_dir.exists():
            create_folder(depth_npy_dir)
        if save_visualization and not depth_png_dir.exists():
            create_folder(depth_png_dir)

    # Save raw disparity data if requested
    if save_depth:
        disparity_file = disparity_output_dir / f"{frame_number}.npy"
        np.save(str(disparity_file), disparity.astype(np.float32))

    # Save depth in meters (float32 .npy with NaN sentinels) regardless
    # of save_depth — the depth-write toggle is whether `depth_m` was passed.
    if depth_m is not None:
        depth_file = depth_npy_dir / f"{frame_number}.npy"
        np.save(str(depth_file), depth_m.astype(np.float32))

    # Save visualization if requested
    if save_visualization:
        # Generate disparity visualization using the utility function
        disparity_vis = vis_disparity(disparity)
        disparity_vis_bgr = cv2.cvtColor(disparity_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
        depth_vis_file = img_output_dir / f"{frame_number}.png"
        cv2.imwrite(str(depth_vis_file), disparity_vis_bgr)

        # Create side-by-side comparison with original image
        combined_vis = np.concatenate([original_img, disparity_vis], axis=1)
        combined_vis_bgr = cv2.cvtColor(combined_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
        vis_file = vis_img_output_dir / f"{frame_number}.png"
        cv2.imwrite(str(vis_file), combined_vis_bgr)

        # depth_image/<i>.png — colorized depth viz (turbo by default).
        if depth_m is not None:
            depth_vis = colorize_depth_m(depth_m, depth_viz_range_m, cmap=depth_viz_cmap)
            depth_png_file = depth_png_dir / f"{frame_number}.png"
            cv2.imwrite(str(depth_png_file), depth_vis)

    save_time = time.time() - save_start
    return save_time


def get_stereo_image_pairs(input_folder: Path, camera_names: List[str],
                          start_frame: int, end_frame: int):
    """
    Get paired stereo images within the specified frame range.
    If start_frame and end_frame are set to default values (-1, -1),
    processes all available frames in the left camera folder.

    Args:
        input_folder: Base input folder containing camera subdirectories
        camera_names: List of camera names (e.g., ['left', 'right'])
        start_frame: First frame to process (-1 for auto-detect from available images)
        end_frame: Last frame to process (inclusive, -1 for auto-detect from available images)

    Returns:
        list: List of tuples (left_path, right_path, frame_number)
    """
    if len(camera_names) != 2:
        raise ValueError("Stereo processing requires exactly 2 cameras")

    # Construct paths to left and right image directories
    left_dir = input_folder / 'image' / camera_names[0]
    right_dir = input_folder / 'image' / camera_names[1]

    if not left_dir.exists():
        raise FileNotFoundError(f"Left camera directory not found: {left_dir}")
    if not right_dir.exists():
        raise FileNotFoundError(f"Right camera directory not found: {right_dir}")

    # Get all left images and sort by frame number
    left_images_sorted = glob_sorted_frame(left_dir)

    if not left_images_sorted:
        raise FileNotFoundError(f"No images found in left camera directory: {left_dir}")

    # Auto-detect frame range if default values are used
    # This allows processing all available images by default
    if start_frame == -1 or end_frame == -1:
        available_frame_numbers = [int(path.stem) for path in left_images_sorted]

        if not available_frame_numbers:
            raise ValueError("Could not extract frame numbers from image filenames")

        actual_start = min(available_frame_numbers) if start_frame == -1 else start_frame
        actual_end = max(available_frame_numbers) if end_frame == -1 else end_frame

        logging.info(f"Auto-detected frame range: {actual_start} to {actual_end} "
                    f"(total {len(available_frame_numbers)} frames available)")
    else:
        actual_start = start_frame
        actual_end = end_frame
        logging.info(f"Using specified frame range: {actual_start} to {actual_end}")

    # Filter by frame range and find corresponding right images
    stereo_pairs = []
    processed_count = 0
    missing_right_count = 0

    for left_path in left_images_sorted:
        frame_num = int(left_path.stem)

        if actual_start <= frame_num <= actual_end:
            # Find corresponding right image
            right_path = right_dir / left_path.name
            if right_path.exists():
                stereo_pairs.append((left_path, right_path, frame_num))
                processed_count += 1
            else:
                logging.warning(f"Missing right image for frame {frame_num}: {right_path}")
                missing_right_count += 1

    # Log summary statistics for better visibility
    logging.info(f"Stereo pair discovery summary:")
    logging.info(f"  - Found {processed_count} complete stereo pairs")
    logging.info(f"  - Missing {missing_right_count} right images")
    logging.info(f"  - Frame range: {actual_start} to {actual_end}")
    logging.info(f"  - Total images to process: {len(stereo_pairs)}")

    if not stereo_pairs:
        raise ValueError(f"No valid stereo pairs found in frame range {actual_start}-{actual_end}")

    return stereo_pairs


# Configure Hydra
cs = ConfigStore.instance()
cs.store(name="depth_estimation", node=AppCfg)

# Set config path relative to the project structure
config_path = Path(__file__).resolve().parents[3] / 'config'

@hydra.main(
    version_base=None,
    config_path=str(config_path),
    config_name="config_de_jack"  # Default config name - will be created
    # config_name="config_de_jack_ubc"  # Default config name - will be created
)
def main(cfg: AppCfg):
    """
    Main processing function that orchestrates the depth estimation pipeline.

    This follows the established pattern used in other processing scripts:
    1. Parse and validate configuration
    2. Set up input/output directories
    3. Load and prepare the model
    4. Process images in batches
    5. Save results and log statistics
    """
    # Extract configuration parameters following the established pattern
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)
    model_path = Path(cfg.preprocess.pretrained_model_path)
    camera_names = cfg.camera_names

    # Depth-in-meters knobs: opt-in depth-in-meters output. Reading with getattr so
    # legacy configs that don't set these keys keep working (default behavior:
    # compute depth, NaN-mask at eps=1e-3, viz range [0.02, 0.5] m, turbo cmap).
    compute_depth = bool(getattr(cfg.preprocess, 'compute_depth', True))
    depth_eps = float(getattr(cfg.preprocess, 'depth_eps', 1.0e-3))
    depth_viz_range_m = tuple(getattr(cfg.preprocess, 'depth_viz_range_m', [0.02, 0.5]))
    # Colormap name resolved by depth_utils._resolve_cv2_colormap. Default
    # "turbo" matches modern stereo viz conventions (Open3D, RealSense, etc.).
    depth_viz_cmap = str(getattr(cfg.preprocess, 'depth_viz_cmap', 'turbo'))
    stereo_calib_filename = str(getattr(cfg.preprocess, 'stereo_calib_filename',
                                        'stereo_calib_params.json'))

    # The stereo calibration lives under intermediate_dir/camera_calibration/
    # (where stage 1 wrote the scaled left.yaml / right.yaml and copied the
    # original stereo_calib_params.json). We compute fx/baseline once here and
    # reuse across all frames.
    fx_left = None
    baseline_m = None
    if compute_depth:
        camera_calibration_path = input_folder / 'camera_calibration'
        try:
            fx_left, baseline_m = load_stereo_depth_params(
                camera_calibration_path, stereo_calib_filename
            )
            logging.info(
                f"Depth conversion enabled: fx_left = {fx_left:.3f} px, "
                f"baseline = {baseline_m * 1000:.4f} mm (depth_eps = {depth_eps:g})"
            )
        except FileNotFoundError as e:
            # Don't kill the run if calibration is missing — just turn off
            # the depth output and log a loud warning. The disparity output
            # still works.
            logging.error(f"Cannot enable depth-in-meters: {e}")
            logging.warning("Continuing with disparity-only outputs.")
            compute_depth = False

    # Validate input directories exist
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")

    # Initialize output directory, clearing if requested
    if cfg.preprocess.folder_initialize:
        if output_folder.exists():
            clear_folder(output_folder)
        else:
            print(f"Output folder does not exist - {output_folder}")

    if not output_folder.exists():
        create_folder(output_folder)

    # Set up model configuration overrides from Hydra config
    model_config_overrides = {
        'scale': cfg.preprocess.scale,
        'hiera': int(cfg.preprocess.hierarchical_inference),
        'valid_iters': cfg.preprocess.valid_iters,
        'save_depth': int(cfg.preprocess.save_depth),
        'start_frame': cfg.preprocess.start_frame,
        'end_frame': cfg.preprocess.end_frame
    }

    # Load and prepare the FoundationStereo model
    logging.info("Loading FoundationStereo model...")
    model, model_cfg = load_and_prepare_model(model_path, model_config_overrides)

    # Get stereo image pairs to process
    logging.info("Discovering stereo image pairs...")

    # Provide user-friendly feedback about processing mode
    if cfg.preprocess.start_frame == -1 and cfg.preprocess.end_frame == -1:
        logging.info("Processing mode: ALL AVAILABLE FRAMES (auto-detected range)")
    elif cfg.preprocess.start_frame == -1:
        logging.info(f"Processing mode: AUTO-START to frame {cfg.preprocess.end_frame}")
    elif cfg.preprocess.end_frame == -1:
        logging.info(f"Processing mode: Frame {cfg.preprocess.start_frame} to AUTO-END")
    else:
        logging.info(f"Processing mode: SPECIFIED RANGE (frames {cfg.preprocess.start_frame}-{cfg.preprocess.end_frame})")

    stereo_pairs = get_stereo_image_pairs(
        input_folder,
        camera_names,
        cfg.preprocess.start_frame,
        cfg.preprocess.end_frame
    )

    if not stereo_pairs:
        logging.warning("No stereo pairs found to process")
        return

    # Process each stereo pair
    logging.info(f"Processing {len(stereo_pairs)} stereo pairs...")

    # Initialize timing statistics for performance monitoring
    total_start_time = time.time()
    frame_times = []
    timing_breakdown = {
        'load': [], 'resize': [], 'tensor_prep': [],
        'inference': [], 'post_process': [], 'save': [], 'cleanup': []
    }

    # Process frames with progress bar
    for left_path, right_path, frame_num in tqdm(stereo_pairs, desc="Processing frames"):
        frame_start_time = time.time()

        try:
            # Process the stereo pair
            disparity, original_img, frame_timing = process_stereo_pair(
                model, left_path, right_path,
                cfg.preprocess.scale,
                cfg.preprocess.hierarchical_inference,
                cfg.preprocess.valid_iters,
                padder_divis_by=int(getattr(cfg.preprocess, 'padder_divis_by', 32)),
                hierarchical_small_ratio=float(getattr(cfg.preprocess, 'hierarchical_small_ratio', 0.5)),
            )

            # Convert disparity → depth in meters (NaN-masked). Note
            # that when preprocess.scale < 1 the saved disparity is at the
            # reduced resolution but so is fx_left (scaled at stage 1) — we
            # would need to re-scale fx by the same factor for the conversion
            # to be exact. We multiply fx by `scale` here so the math works
            # at any scale.
            depth_m = None
            if compute_depth:
                fx_scaled = fx_left * float(cfg.preprocess.scale)
                depth_m = disparity_to_depth_m(
                    disparity_px=disparity,
                    fx_left=fx_scaled,
                    baseline_m=baseline_m,
                    eps=depth_eps,
                )

            # Save results
            save_time = save_depth_results(
                disparity, original_img, output_folder, frame_num,
                cfg.preprocess.save_depth,
                cfg.preprocess.save_visualization,
                depth_m=depth_m,
                depth_viz_range_m=depth_viz_range_m,
                depth_viz_cmap=depth_viz_cmap,
            )
            frame_timing['save'] = save_time

            # Record timing statistics
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)

            # Accumulate detailed timing breakdown
            for timing_key in timing_breakdown:
                if timing_key in frame_timing:
                    timing_breakdown[timing_key].append(frame_timing[timing_key])

            # Log progress for current frame
            logging.info(f"Frame {frame_num} processed in {frame_time:.2f}s "
                        f"(load: {frame_timing.get('load', 0):.2f}s, "
                        f"resize: {frame_timing.get('resize', 0):.2f}s, "
                        f"tensor: {frame_timing.get('tensor_prep', 0):.2f}s, "
                        f"inference: {frame_timing.get('inference', 0):.2f}s, "
                        f"post: {frame_timing.get('post_process', 0):.2f}s, "
                        f"save: {frame_timing.get('save', 0):.2f}s, "
                        f"cleanup: {frame_timing.get('cleanup', 0):.2f}s)")

        except Exception as e:
            logging.error(f"Failed to process frame {frame_num}: {e}")
            continue

    # Log overall processing statistics
    total_time = time.time() - total_start_time
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0

    logging.info("")
    logging.info("="*50)
    logging.info("DEPTH ESTIMATION PROCESSING COMPLETE")
    logging.info("="*50)
    logging.info(f"Results saved to: {output_folder}")
    logging.info(f"Total processing time: {total_time:.2f}s for {len(frame_times)} frames")
    logging.info(f"Average time per frame: {avg_frame_time:.2f}s ({1/avg_frame_time:.2f} FPS)")

    if frame_times:
        logging.info(f"Frame time range: {min(frame_times):.2f}s - {max(frame_times):.2f}s")

        # Log average timing breakdown
        logging.info("Average timing breakdown:")
        for timing_key, times in timing_breakdown.items():
            if times:
                avg_time = sum(times) / len(times)
                logging.info(f"  {timing_key}: {avg_time:.3f}s")


if __name__ == '__main__':
    main()
    print('Depth Estimation Processing Complete!')

    # from hydra import compose, initialize
    # with initialize(version_base=None, config_path='../../../config'):
    #     cfg = compose(config_name="config_de_jack")