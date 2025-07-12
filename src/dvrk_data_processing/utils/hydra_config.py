from dataclasses import dataclass
from pathlib import Path
from typing import Union, List


@dataclass
class PathConfig:
    data_dir: Union[Path, str]
    data_name: str
    data_index: str
    raw_data_dir: Union[Path, str]
    intermediate_dir: Union[Path, str]
    processed_dir: Union[Path, str]


@dataclass
class KinmaticMapWeightConfig:
    sigma_x: float
    sigma_y: float
    advance_weight: bool
    tol_dist: float


@dataclass
class KinematicMapConfig:
    stage: str
    img_size: List[int]
    arm_name: List[str]
    input_subfolder: str
    output_subfolder: str
    weight_config: KinmaticMapWeightConfig
    fps_img: float
    fps_kin: float
    enable_overlay: bool
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
