from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, List


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
class CalibratedKinematicConfig:
    """
    calibrated_kinematic block. Single knob: whether to emit the
    per-frame JSON files. Default off (must be opted into via YAML) so the
    legacy heatmap-only behavior stays the default when an older config is
    loaded without the new key.
    """
    enable: bool = False


@dataclass
class DrawframeConfigSchema:
    """
    Drawframe block. Keep this matching the YAML knobs:
      enable, axis_length_m, line_thickness_px, origin_marker_radius_px,
      colors_bgr (sub-block), cameras (list).
    `colors_bgr` is left as `Any` so OmegaConf doesn't fight us over the
    BGR triple format (DictConfig of three int-lists).
    """
    enable: bool = False
    axis_length_m: float = 0.010
    line_thickness_px: int = 2
    origin_marker_radius_px: int = 3
    colors_bgr: Any = None
    cameras: List[str] = field(default_factory=lambda: ["left", "right"])


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
    # New calibrated_kinematic / drawframe feature blocks. Defaulted to a no-op so old
    # configs that don't define these keys keep working (the resolver in
    # the entry point checks `enable` first).
    calibrated_kinematic: CalibratedKinematicConfig = field(default_factory=CalibratedKinematicConfig)
    drawframe: DrawframeConfigSchema = field(default_factory=DrawframeConfigSchema)


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
    # Depth-in-meters augmentation. Defaulted to enabled so any
    # caller using the new YAML schema automatically gets the depth/<i>.npy
    # output without further opt-in.
    compute_depth: bool = True
    # Disparity values at or below this threshold (pixels) → depth NaN. The
    # spec explicitly forbids clamping to a large positive number; NaN is the
    # "model failed here" signal. 1e-3 keeps a healthy guard band away from
    # both float epsilon and any realistic textured-region disparity.
    depth_eps: float = 1.0e-3
    # Colormap range (meters) for the depth_image/<i>.png visualization. The
    # default [0.02 m, 0.5 m] window matches the working volume of the
    # sample stereo rig; pixels outside this clip to the colormap endpoints.
    depth_viz_range_m: List[float] = None
    # Name of the cv2 colormap used to colorize depth_image/. Defaults to
    # "turbo" (Google's perceptually-uniform replacement for jet — modern
    # standard for stereo/depth visualization). Supported names live in
    # depth_utils._CV2_COLORMAP_BY_NAME (turbo, jet, inferno, magma, plasma,
    # viridis, parula, hot, bone, rainbow).
    depth_viz_cmap: str = "turbo"
    # Path to the stereo calibration JSON inside intermediate_dir. The depth
    # converter reads `baseline_m` from it. Treated as relative when not
    # starting with '/' — resolved against the camera_calibration folder
    # alongside left.yaml / right.yaml.
    stereo_calib_filename: str = "stereo_calib_params.json"
    # FoundationStereo InputPadder divisor: how many pixels the model needs
    # H and W to be divisible by. The pretrained 23-51-11 / 11-33-40 weights
    # require 32; smaller values risk shape mismatches. Don't change unless
    # you also know the model's stride.
    padder_divis_by: int = 32
    # `run_hierachical` (sic — that's the upstream spelling) downsamples by
    # this factor for the coarse pass when hierarchical_inference=true. The
    # FoundationStereo default is 0.5; tighten the ratio (e.g. 0.25) only if
    # the coarse pass is unstable on your data.
    hierarchical_small_ratio: float = 0.5

    def __post_init__(self):
        # OmegaConf passes default mutable args carefully; this still has the
        # standard Python gotcha so we instantiate the list here.
        if self.depth_viz_range_m is None:
            self.depth_viz_range_m = [0.02, 0.5]


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
