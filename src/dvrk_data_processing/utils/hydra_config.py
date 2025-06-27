from dataclasses import dataclass
from pathlib import Path
from typing import Union, List


@dataclass
class PathConfig:
    data_dir: Union[Path, str]
    data_name: str
    raw_data_dir: Union[Path, str]
    intermediate_dir: Union[Path, str]
    processed_dir: Union[Path, str]


@dataclass
class KinematicMapConfig:
    stage: str
    img_size: List[int]
    arm_name: List[str]
    input_subfolder: str
    output_subfolder: str
    fps: float
    sigma_x: float
    sigma_y: float
    camera_calibration_path: Union[Path, str]
    weight_adv: bool
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
