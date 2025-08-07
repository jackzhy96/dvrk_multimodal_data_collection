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


if __name__ == "__main__":
    pass
