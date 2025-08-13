from dataclasses import dataclass
from pathlib import Path
from typing import Union, List


@dataclass
class PathConfig:
    data_dir: Union[Path, str]
    data_name: str
    data_index: str
    raw_dir: Union[Path, str]
    intermediate_dir: Union[Path, str]
    processed_dir: Union[Path, str]


@dataclass
class KinmaticMapWeightConfig:
    sigma_x: float
    sigma_y: float
    advanced_weight: bool
    tol_dist: float


@dataclass
class KinematicMapConfig:
    stage: str
    img_size: List[int]
    arm_name: List[str]
    input_folder: str
    output_folder: str
    weight_config: KinmaticMapWeightConfig
    fps_img: float
    fps_kin: float
    enable_overlay: bool
    folder_initialize: bool = False


@dataclass
class ResizeConfig:
    original_size: List[int]
    new_size: List[int]
    enable_resize: bool


@dataclass
class ResizeRectifyConfig:
    stage: str
    img_size: List[int]
    resize_config: ResizeConfig
    input_folder: str
    output_folder: str
    enable_rectify: bool
    folder_initialize: bool = False


@dataclass
class DepthEstimationConfig:
    stage: str
    input_folder: str
    output_folder: str
    pretrained_model_path: str
    scale: float
    hierarchical_inference: bool
    valid_iters: int
    save_depth: bool
    save_visualization: bool
    start_frame: int
    end_frame: int
    folder_initialize: bool = False


@dataclass
class OpticalFlowFilterConfig:
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    gaussian_kernel_size: List[int]
    gaussian_sigma: float


@dataclass
class OpticalFlowAlgorithmConfig:
    pyramid_scale: float
    pyramid_levels: int
    window_size: int
    iterations: int
    poly_n: int
    poly_sigma: float
    flags: int


@dataclass
class OpticalFlowConfig:
    stage: str
    input_folder: str
    output_folder: str
    camera_names: List[str]
    flow_format: str
    enable_visualization: bool
    enable_preprocessing: bool
    filter_config: OpticalFlowFilterConfig
    algorithm_config: OpticalFlowAlgorithmConfig
    folder_initialize: bool = False


@dataclass
class RaftModelConfig:
    model_variant: str = "large"  # "large" or "small" - large is more accurate, small is faster
    use_pretrained: bool = True  # Use pretrained weights (highly recommended)
    device: str = "auto"  # "auto", "cuda", "cpu" - auto will use GPU if available
    batch_size: int = 1  # Number of frame pairs to process simultaneously
    mixed_precision: bool = True  # Use mixed precision for faster inference and lower memory usage
    num_flow_updates: int = -1  # Number of RAFT iterations (-1 uses all iterations)


# @dataclass
# class RaftPreprocessConfig:
#     resize_mode: str = "pad"  # "pad", "resize", "crop" - how to handle size requirements
#     target_size: Union[List[int], None] = None  # Target size for resizing (None = use original size)
#     normalize: bool = True  # Apply ImageNet normalization (required for pretrained models)
#     ensure_divisible: int = 8  # Ensure dimensions are divisible by this value (RAFT requirement)


@dataclass
class RaftOpticalFlowConfig:
    stage: str
    input_folder: str
    output_folder: str
    camera_names: List[str]
    model_config: RaftModelConfig
    # preprocess_config: RaftPreprocessConfig
    flow_format: str = "npy"  # "npy" or "flo" - output format for optical flow
    enable_visualization: bool = True  # Generate color-coded flow visualizations
    save_confidence: bool = False  # Save confidence/uncertainty maps (if supported)
    folder_initialize: bool = False



if __name__ == "__main__":
    pass
