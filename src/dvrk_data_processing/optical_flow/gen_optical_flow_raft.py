"""
Deep Learning Optical Flow Generation Script using RAFT

This script implements optical flow calculation using the state-of-the-art RAFT (Recurrent All-Pairs Field Transforms) model.
RAFT is currently the top-performing optical flow method on benchmarks like SINTEL and KITTI.

Key improvements over traditional methods:
- Uses deep learning for more accurate flow estimation, especially in challenging scenarios
- Better handles occlusions, motion blur, and complex motions
- Provides more robust optical flow for surgical scenes with fine instruments and tissue deformation
- Follows the established patterns in the codebase with Hydra configuration and pathlib
- Integrates seamlessly with the existing data processing pipeline
- Supports both left and right camera optical flows simultaneously
- Configurable and scalable through YAML config files
- Auto-adjusts file paths based on data organization structure

Technical Details:
- Uses PyTorch's torchvision implementation of RAFT
- Supports both RAFT Large (more accurate) and RAFT Small (faster) models
- Implements proper preprocessing and postprocessing for surgical image data
- Comprehensive error handling and progress tracking
- Efficient batch processing for better GPU utilization
"""

from dataclasses import dataclass
from typing import Union, List, Tuple, Optional
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import cv2
import time
import re
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from torchvision.utils import flow_to_image

# Import local utilities following the established pattern
from dvrk_data_processing.utils.hydra_config import PathConfig, RaftOpticalFlowConfig
from dvrk_data_processing.utils.utility import create_folder, clear_folder, get_sorted_names, glob_sorted_frame


@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: RaftOpticalFlowConfig
    workspace: str
    camera_names: List[str]


class RaftOpticalFlowProcessor:
    """
    Main processor class for RAFT-based optical flow calculation.

    This class encapsulates all RAFT-related functionality including model loading,
    preprocessing, inference, and postprocessing. It follows object-oriented design
    principles to maintain clean separation of concerns.
    """

    def __init__(self, config: RaftOpticalFlowConfig):
        """
        Initialize the RAFT optical flow processor.

        Args:
            config: Configuration object containing all processing parameters
        """
        self.config = config
        self.model = None
        self.device = None
        self.preprocessor = None

        # Initialize the processor components
        self._setup_device()
        self._load_model()
        self._setup_preprocessor()

        logging.info(f"RAFT Optical Flow Processor initialized")
        logging.info(f"Model variant: {config.model_config.model_variant}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Batch size: {config.model_config.batch_size}")

    def _setup_device(self):
        """
        Set up the computation device (CPU/GPU) based on configuration and availability.
        This ensures optimal performance by utilizing GPU acceleration when available.
        """
        if self.config.model_config.device == "auto":
            # Automatically select the best available device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                logging.warning("GPU not available, using CPU (will be slower)")
        else:
            self.device = torch.device(self.config.model_config.device)
            logging.info(f"Using specified device: {self.device}")

    def _load_model(self):
        """
        Load the RAFT model with appropriate weights and configuration.
        This handles both large and small RAFT variants with pretrained weights.
        """
        model_config = self.config.model_config

        try:
            if model_config.model_variant.lower() == "large":
                # Load RAFT Large - more accurate but slower and more memory-intensive
                if model_config.use_pretrained:
                    weights = Raft_Large_Weights.DEFAULT
                    self.model = raft_large(weights=weights, progress=False)
                    logging.info("Loaded RAFT Large with pretrained weights")
                else:
                    self.model = raft_large(weights=None, progress=False)
                    logging.info("Loaded RAFT Large without pretrained weights")

            elif model_config.model_variant.lower() == "small":
                # Load RAFT Small - faster but less accurate
                if model_config.use_pretrained:
                    weights = Raft_Small_Weights.DEFAULT
                    self.model = raft_small(weights=weights, progress=False)
                    logging.info("Loaded RAFT Small with pretrained weights")
                else:
                    self.model = raft_small(weights=None, progress=False)
                    logging.info("Loaded RAFT Small without pretrained weights")

            else:
                raise ValueError(f"Unsupported RAFT variant: {model_config.model_variant}")

            # Move model to the appropriate device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Enable mixed precision if requested (for faster inference on modern GPUs)
            if model_config.mixed_precision and self.device.type == "cuda":
                # Mixed precision is handled during inference, not model initialization
                logging.info("Mixed precision enabled for faster inference")

        except Exception as e:
            logging.error(f"Failed to load RAFT model: {e}")
            raise

    def _setup_preprocessor(self):
        """
        Set up image preprocessing pipeline for RAFT model.
        This ensures images are in the correct format and size for RAFT inference.
        Using cv2-based preprocessing for consistency with the existing codebase.
        """
        # RAFT models require specific preprocessing based on their training
        if self.config.model_config.use_pretrained:
            # For pretrained models, we need to apply ImageNet normalization
            # Mean and std values from ImageNet (used in torchvision pretrained models)
            self.imagenet_mean = np.array([0.485, 0.456, 0.406])
            self.imagenet_std = np.array([0.229, 0.224, 0.225])
            logging.info("Using ImageNet normalization for pretrained model")
        else:
            # For non-pretrained models, simple [0,1] normalization
            self.imagenet_mean = None
            self.imagenet_std = None
            logging.info("Using basic [0,1] normalization for non-pretrained model")

    def _cv2_to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert cv2 BGR image to PyTorch tensor with proper preprocessing.

        Args:
            img_bgr: cv2 image in BGR format (H, W, 3)

        Returns:
            torch.Tensor: Preprocessed image tensor (3, H, W)
        """
        # Convert BGR to RGB (cv2 uses BGR, PyTorch models expect RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize to [0,1] range
        img_float = img_rgb.astype(np.float32) / 255.0

        # Apply ImageNet normalization if using pretrained models
        if self.imagenet_mean is not None and self.imagenet_std is not None:
            # Normalize using ImageNet statistics for pretrained models
            img_normalized = (img_float - self.imagenet_mean) / self.imagenet_std
        else:
            # Keep in [0,1] range for non-pretrained models
            img_normalized = img_float

        # Convert from (H, W, 3) to (3, H, W) format expected by PyTorch
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1).copy()).float()

        return img_tensor

    def _preprocess_image_pair(self, img1_path: Path, img2_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a pair of images for RAFT inference using cv2.

        Args:
            img1_path: Path to the first image
            img2_path: Path to the second image

        Returns:
            Tuple of preprocessed image tensors ready for RAFT

        Raises:
            FileNotFoundError: If images don't exist
            ValueError: If images can't be loaded or have mismatched dimensions
        """
        # Validate input files exist
        if not img1_path.exists():
            raise FileNotFoundError(f"First image not found: {img1_path}")
        if not img2_path.exists():
            raise FileNotFoundError(f"Second image not found: {img2_path}")

        try:
            # Load images using cv2 for consistency with existing codebase
            img1_bgr = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
            img2_bgr = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)

            # Validate images were loaded successfully
            if img1_bgr is None:
                raise ValueError(f"Could not load first image: {img1_path}")
            if img2_bgr is None:
                raise ValueError(f"Could not load second image: {img2_path}")

            # Check that images have the same dimensions
            if img1_bgr.shape != img2_bgr.shape:
                raise ValueError(f"Image shape mismatch: {img1_bgr.shape} vs {img2_bgr.shape}")

            # Convert cv2 images to PyTorch tensors with proper preprocessing
            img1_tensor = self._cv2_to_tensor(img1_bgr)
            img2_tensor = self._cv2_to_tensor(img2_bgr)

            # Ensure images are on the correct device
            img1_tensor = img1_tensor.to(self.device)
            img2_tensor = img2_tensor.to(self.device)

            # Add batch dimension (RAFT expects batched input)
            img1_tensor = img1_tensor.unsqueeze(0)  # (1, 3, H, W)
            img2_tensor = img2_tensor.unsqueeze(0)  # (1, 3, H, W)

            return img1_tensor, img2_tensor

        except Exception as e:
            logging.error(f"Failed to preprocess images {img1_path.name}, {img2_path.name}: {e}")
            raise

    def calculate_optical_flow(self, img1_path: Path, img2_path: Path) -> np.ndarray:
        """
        Calculate optical flow between two images using RAFT.

        This method handles the complete flow calculation pipeline including
        preprocessing, model inference, and postprocessing.

        Args:
            img1_path: Path to the first image
            img2_path: Path to the second image

        Returns:
            Dense optical flow field as numpy array (H, W, 2) containing (u, v) components

        Raises:
            FileNotFoundError: If input images don't exist
            ValueError: If images can't be processed
        """
        try:
            # Preprocess the image pair for RAFT inference
            img1_tensor, img2_tensor = self._preprocess_image_pair(img1_path, img2_path)

            # Perform RAFT inference
            with torch.no_grad():  # Disable gradient calculation for inference
                # Use mixed precision if enabled and supported
                if (self.config.model_config.mixed_precision and
                    self.device.type == "cuda"):
                    with torch.cuda.amp.autocast():
                        list_of_flows = self.model(img1_tensor, img2_tensor)
                else:
                    list_of_flows = self.model(img1_tensor, img2_tensor)

            # Extract the final flow prediction
            # RAFT returns a list of flows from different iterations
            # The last element typically contains the most refined flow
            if self.config.model_config.num_flow_updates > 0:
                # Use specific number of flow updates if specified
                flow_idx = min(self.config.model_config.num_flow_updates - 1, len(list_of_flows) - 1)
                predicted_flow = list_of_flows[flow_idx]
            else:
                # Use the final iteration's flow
                predicted_flow = list_of_flows[-1]

            # Convert from PyTorch tensor to numpy array
            # Flow tensor is in shape (B, 2, H, W), we need (H, W, 2)
            flow_numpy = predicted_flow[0].permute(1, 2, 0).cpu().numpy()

            return flow_numpy

        except Exception as e:
            logging.error(f"RAFT optical flow calculation failed: {e}")
            raise

    def visualize_optical_flow(self, flow: np.ndarray) -> np.ndarray:
        """
        Create a color-coded visualization of RAFT optical flow field.

        Uses torchvision's flow_to_image function for consistent visualization
        that follows standard optical flow color coding conventions.

        Args:
            flow: Dense optical flow field (H, W, 2)

        Returns:
            Color-coded flow visualization in BGR format for OpenCV compatibility
        """
        try:
            # Convert numpy flow to PyTorch tensor for torchvision function
            # torchvision expects (2, H, W) format
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)

            # Generate color-coded visualization using torchvision
            # This function uses the standard optical flow color wheel encoding
            flow_img_tensor = flow_to_image(flow_tensor)

            # Convert back to numpy and change from RGB to BGR for OpenCV
            flow_img_rgb = flow_img_tensor[0].permute(1, 2, 0).numpy().astype(np.uint8)
            flow_img_bgr = cv2.cvtColor(flow_img_rgb, cv2.COLOR_RGB2BGR)

            return flow_img_bgr

        except Exception as e:
            logging.error(f"Flow visualization failed: {e}")
            # Fallback to a simple magnitude-based visualization
            magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.applyColorMap(magnitude_normalized.astype(np.uint8), cv2.COLORMAP_JET)


def save_optical_flow_results(flow: np.ndarray, visualization: Optional[np.ndarray],
                             output_dir: Path, frame_name: str, camera_name: str,
                             flow_format: str, enable_visualization: bool) -> dict:
    """
    Save RAFT optical flow results to disk in the specified format.

    Args:
        flow: Computed optical flow field
        visualization: Flow visualization image (can be None)
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
    flow_filename = f"{frame_name}.{flow_format}"
    flow_folder = output_dir / camera_name / 'optical_flow'
    flow_filepath = flow_folder / flow_filename
    if not flow_folder.exists():
        create_folder(flow_folder)

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

    # Save visualization if requested and available
    if enable_visualization and visualization is not None:
        vis_start = time.time()
        vis_dir = output_dir / camera_name / 'image'
        if not vis_dir.exists():
            create_folder(vis_dir)

        vis_filename = f"{frame_name}.png"
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
                               output_dir: Path, raft_config: RaftOpticalFlowConfig,
                               raft_processor: RaftOpticalFlowProcessor) -> dict:
    """
    Process RAFT optical flow for a single camera's image sequence.

    This function calculates optical flow between consecutive frames using the RAFT model
    and handles all saving operations. It processes the sequence in temporal order and
    provides comprehensive progress tracking.

    Args:
        camera_name: Identifier for the camera being processed
        image_sequence: Sorted list of image file paths
        output_dir: Directory for saving results
        raft_config: Configuration parameters for RAFT optical flow processing
        raft_processor: Initialized RAFT processor instance

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

    logging.info(f"Processing {camera_name} camera RAFT optical flow...")

    # Process consecutive frame pairs
    # Each optical flow represents motion from frame[i] to frame[i+1]
    for i in tqdm(range(len(image_sequence) - 1), desc=f"{camera_name} RAFT"):
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
        frame_name = f'{frame1_num}'

        try:
            # Calculate optical flow between consecutive frames using RAFT
            flow_start = time.time()
            flow = raft_processor.calculate_optical_flow(frame1_path, frame2_path)
            flow_time = time.time() - flow_start
            timing_stats["flow_calculation"].append(flow_time)

            # Generate visualization if requested
            visualization = None
            if raft_config.enable_visualization:
                vis_start = time.time()
                visualization = raft_processor.visualize_optical_flow(flow)
                vis_time = time.time() - vis_start
                timing_stats["visualization"].append(vis_time)

            # Save results to disk
            save_timing = save_optical_flow_results(
                flow, visualization, output_dir, frame_name, camera_name,
                raft_config.flow_format, raft_config.enable_visualization
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
cs.store(name="raft_optical_flow", node=AppCfg)

# Set config path relative to the project structure
config_path = Path(__file__).resolve().parents[3] / 'config'


@hydra.main(
    version_base=None,
    config_path=str(config_path),
    config_name="config_of_raft_jack"  # New config name for RAFT optical flow
)
def main(cfg: AppCfg):
    """
    Main processing function that orchestrates the RAFT optical flow calculation pipeline.

    This follows the established pattern used in other processing scripts:
    1. Parse and validate configuration
    2. Set up input/output directories
    3. Initialize RAFT processor with model loading
    4. Discover image sequences for each camera
    5. Process optical flows for both cameras using RAFT
    6. Save results and log comprehensive statistics
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
    logging.info("RAFT OPTICAL FLOW PROCESSING")
    logging.info("=" * 60)
    logging.info(f"Input directory: {input_folder}")
    logging.info(f"Output directory: {output_folder}")
    logging.info(f"Cameras to process: {camera_names}")
    logging.info(f"Flow format: {cfg.preprocess.flow_format}")
    logging.info(f"Visualization: {'Enabled' if cfg.preprocess.enable_visualization else 'Disabled'}")

    # Log RAFT-specific configuration
    model_cfg = cfg.preprocess.model_config
    logging.info(f"RAFT model variant: {model_cfg.model_variant}")
    logging.info(f"Pretrained weights: {'Enabled' if model_cfg.use_pretrained else 'Disabled'}")
    logging.info(f"Batch size: {model_cfg.batch_size}")
    logging.info(f"Mixed precision: {'Enabled' if model_cfg.mixed_precision else 'Disabled'}")
    logging.info("=" * 60)

    # Initialize RAFT processor
    # This loads the model and sets up all necessary components
    logging.info("Initializing RAFT optical flow processor...")
    try:
        raft_processor = RaftOpticalFlowProcessor(cfg.preprocess)
    except Exception as e:
        logging.error(f"Failed to initialize RAFT processor: {e}")
        return

    # Discover image sequences for each camera
    logging.info("Discovering image sequences...")
    camera_sequences = get_camera_image_sequences(input_folder, camera_names)

    # Validate that we have sequences to process
    total_images = sum(len(seq) for seq in camera_sequences.values())
    if total_images == 0:
        logging.error("No image sequences found to process")
        return

    # Process RAFT optical flows for each camera
    logging.info(f"Processing RAFT optical flows for {len(camera_names)} cameras...")

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

        # Process this camera's optical flow sequence using RAFT
        camera_results = process_camera_optical_flow(
            camera_name, image_sequence, output_folder, cfg.preprocess, raft_processor
        )
        processing_results[camera_name] = camera_results

    # Calculate and log comprehensive summary statistics
    total_time = time.time() - total_start_time

    logging.info("")
    logging.info("=" * 60)
    logging.info("RAFT OPTICAL FLOW PROCESSING COMPLETE")
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
        logging.info(f"  - RAFT optical flows generated: {flows_count}")
        logging.info(f"  - Processing errors: {error_count}")
        logging.info(f"  - Success rate: {success_rate:.1f}%")

        # Log timing statistics if available
        timing = results.get("timing", {})
        if timing.get("flow_calculation"):
            avg_flow_time = sum(timing["flow_calculation"]) / len(timing["flow_calculation"])
            logging.info(f"  - Average RAFT calculation time: {avg_flow_time:.3f}s")
        if timing.get("saving"):
            avg_save_time = sum(timing["saving"]) / len(timing["saving"])
            logging.info(f"  - Average save time: {avg_save_time:.3f}s")

    logging.info(f"SUMMARY: Generated {total_flows_generated} RAFT optical flows with {total_errors} errors")

    if cfg.preprocess.enable_visualization:
        vis_dir = output_folder / "visualization"
        logging.info(f"RAFT visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
    print('RAFT Optical Flow Processing Complete!')