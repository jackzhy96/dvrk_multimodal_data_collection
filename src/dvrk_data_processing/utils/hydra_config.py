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
class OpticalFlowConfig:
    stage: str
    folder_initialize: bool = False


@dataclass
class DepthEstimationConfig:
    stage: str
    folder_initialize: bool = False

if __name__ == "__main__":
    pass
