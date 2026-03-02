from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List


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
    enable_overlay: bool
    # Per-PSM weight overrides — if None, falls back to the global weight_config
    weight_config_PSM1: Optional[KinmaticMapWeightConfig] = None
    weight_config_PSM2: Optional[KinmaticMapWeightConfig] = None
    weight_config_PSM3: Optional[KinmaticMapWeightConfig] = None
    fps_kin: float = -1.0  # legacy field; scripts now use per-frame measured_frequency instead
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
class RaftFlowToImageConfig:
    clip: float
    eps: float
    eps_frame: float
    enable_denoise: bool=True

@dataclass
class RaftOpticalFlowConfig:
    stage: str
    input_folder: str
    output_folder: str
    camera_names: List[str]
    model_config: RaftModelConfig
    # preprocess_config: RaftPreprocessConfig
    flow_to_image_config: RaftFlowToImageConfig
    flow_format: str = "npy"  # "npy" or "flo" - output format for optical flow
    enable_visualization: bool = True  # Generate color-coded flow visualizations
    save_confidence: bool = False  # Save confidence/uncertainty maps (if supported)
    folder_initialize: bool = False

@dataclass
class ResizeVideoConfig:
    enable_resize: bool
    new_size: List[int]  # [width, height]


@dataclass
class FFmpegConfig:
    """FFmpeg encoding settings for high-quality video output."""
    codec: str = "libx264"         # FFmpeg codec (libx264, libx265, etc.)
    crf: int = 17                  # Constant Rate Factor (0-51, lower = better quality)
    preset: str = "slow"           # Encoding preset (ultrafast/fast/medium/slow/veryslow)
    pixel_format: str = "yuv420p"  # Pixel format for broad playback compatibility


@dataclass
class ImageToVideoConfig:
    stage: str
    video_fixed_rate: float
    enable_fixed_rate: bool
    resize_config: ResizeVideoConfig
    ffmpeg_config: FFmpegConfig = None  # FFmpeg encoding settings (uses defaults if None)
    start_frame: int = -1  # -1 means start from first frame (0)
    end_frame: int = -1  # -1 means process until last frame
    folder_initialize: bool = False

    def __post_init__(self):
        # Initialize default FFmpegConfig if not provided
        if self.ffmpeg_config is None:
            self.ffmpeg_config = FFmpegConfig()


@dataclass
class VideoToImageConfig:
    stage: str
    enable_timestamp: bool
    file_extension: str = "png"  # output image file extension
    extract_fps: float = -1  # target extraction fps; -1 = extract all frames
    folder_initialize: bool = False


@dataclass
class UserSkillLevel:
    """User skill level information for data organization"""
    dVRK: int = -1  # skill level with dVRK system
    clinical: int = -1  # clinical/surgical skill level


@dataclass
class UserInfo:
    """User information for data organization metadata"""
    user_id: Union[int, str] = ""  # user identifier (can be empty)
    user_skill_level: UserSkillLevel = None  # skill level ratings
    user_description: str = ""  # additional user description (can be empty)

    def __post_init__(self):
        # Initialize default UserSkillLevel if not provided
        if self.user_skill_level is None:
            self.user_skill_level = UserSkillLevel()


@dataclass
class DataOrganizationConfig:
    """
    Configuration for data organization/reorganization script.
    Reorganizes raw data from nested folder structures into a flat, indexed structure.
    """
    stage: str  # processing stage name
    input_folder: str  # path to unorganized raw data (may contain multiple subfolders)
    output_folder: str  # path for reorganized data (will be created if needed)
    copy_image_name: List[str]  # list of camera names to copy (e.g., ["left", "right", "side"])
    enable_kinematic_copy: bool  # whether to copy kinematic folders
    enable_timestamp_copy: bool  # whether to copy time_syn folders
    enable_label_copy: bool  # whether to copy annotation folders
    start_idx: int  # starting index for output folders (use -1 to auto-continue from last)
    user_info: UserInfo = None  # user metadata for the dataset
    folder_initialize: bool = False  # whether to initialize/clear output folder

    def __post_init__(self):
        # Initialize default UserInfo if not provided
        if self.user_info is None:
            self.user_info = UserInfo()


if __name__ == "__main__":
    pass
