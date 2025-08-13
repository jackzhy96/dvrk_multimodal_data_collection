from __future__ import annotations
import json
from dataclasses import dataclass, fields, is_dataclass, asdict
from pathlib import Path
from typing import Union, List, TypeVar, get_type_hints, Type
import yaml
import numpy as np


@dataclass
class CameraParameters:
    data: List[float]
    cols: int
    rows: int


@dataclass
class CameraInfo:
    R_stereo: CameraParameters
    T_stereo: CameraParameters
    camera_matrix: CameraParameters
    distortion_coefficients: CameraParameters
    camera_name: str
    distortion_model: str
    image_height: int
    image_width: int


@dataclass
class MonoCameraInfo:
    camera_matrix: CameraParameters
    distortion_coefficients: CameraParameters
    camera_name: str
    distortion_model: str
    image_height: int
    image_width: int


@dataclass
class CameraInfoProcessed:
    K: np.ndarray
    D: np.ndarray
    R_c: np.ndarray
    t_c: np.ndarray
    image_height: int
    image_width: int


@dataclass
class MonoCameraInfoProcessed:
    K: np.ndarray
    D: np.ndarray
    image_height: int
    image_width: int


@dataclass
class HeaderConfig:
    sec: int
    nsec: int


@dataclass
class JawConfig:
    position: List[float]


@dataclass
class JawLoadConfig:
    measured_data: JawConfig
    setpoint_data: JawConfig


@dataclass
class ArmMeasuredCpConfig:
    position: List[float]
    orientation: List[float]
    velocity: List[float]


@dataclass
class PSMMeasuredCpConfig:
    position: List[float]
    orientation: List[float]
    velocity: List[float]


@dataclass
class ECMMeasuredCpConfig:
    position: List[float]
    orientation: List[float]


@dataclass
class ArmLocalCpConfig:
    position: List[float]
    orientation: List[float]


@dataclass
class ArmSetpointCpConfig:
    position: List[float]
    orientation: List[float]


@dataclass
class ArmJsConfig:
    position: List[float]
    velocity: List[float]
    effort: List[float]

@dataclass
class MeasuredLoadConfig:
    cp: ArmMeasuredCpConfig
    js: ArmJsConfig


@dataclass
class PSMMeasuredLoadConfig:
    cp: PSMMeasuredCpConfig
    js: ArmJsConfig


@dataclass
class ECMMeasuredLoadConfig:
    cp: ECMMeasuredCpConfig
    js: ArmJsConfig

@dataclass
class DesiredLoadConfig:
    cp: ArmSetpointCpConfig
    js: ArmJsConfig

@dataclass
class ArmLoadConfig:
    local_cp: ArmLocalCpConfig
    measured_data: MeasuredLoadConfig
    setpoint_data: DesiredLoadConfig


@dataclass
class PSMLoadConfig:
    local_cp: ArmLocalCpConfig
    measured_data: PSMMeasuredLoadConfig
    setpoint_data: DesiredLoadConfig


@dataclass
class ECMLoadConfig:
    local_cp: ArmLocalCpConfig
    measured_data: ECMMeasuredLoadConfig
    setpoint_data: DesiredLoadConfig


@dataclass
class KinematicInfo:
    arm: ArmLoadConfig
    header: HeaderConfig
    jaw: JawLoadConfig


@dataclass
class PSMInfo:
    arm: PSMLoadConfig
    header: HeaderConfig
    jaw: JawLoadConfig
    measured_frequency: float


@dataclass
class ECMInfo:
    arm: ECMLoadConfig
    header: HeaderConfig


@dataclass
class CPInfo:
    arm_name: str
    R: np.ndarray
    t: np.ndarray
    w: np.ndarray
    v: np.ndarray
    R_local: np.ndarray
    t_local: np.ndarray


@dataclass
class PSMCPInfo:
    arm_name: str
    R: np.ndarray
    t: np.ndarray
    w: np.ndarray
    v: np.ndarray
    R_local: np.ndarray
    t_local: np.ndarray
    measured_frequency: float


@dataclass
class ECMCPInfo:
    arm_name: str
    R: np.ndarray
    t: np.ndarray
    R_local: np.ndarray
    t_local: np.ndarray


@dataclass
class HandEyeLoadConfig:
    name: str
    measured_cp: List[list]


@dataclass
class HandEyeInfo:
    name: str
    transformation_matrix: np.ndarray


datacls = TypeVar("datacls")
def datacls_from_dict(data_class: Type[datacls], raw: dict) -> datacls:
    '''
    Convert a loaded dict into an instance of the given data class recursively.
    data_class: the data class to convert to
    raw: the loaded dict
    output: an instance of the given data class
    '''
    hints = get_type_hints(data_class)
    kwargs = {}
    for f in fields(data_class):
        value = raw.get(f.name)
        if value is None:
            continue
        f_type = hints[f.name]
        if is_dataclass(f_type):
            kwargs[f.name] = datacls_from_dict(f_type, value)
        else:
            # plain types
            kwargs[f.name] = value
    return data_class(**kwargs)


if __name__ == "__main__":
    data_path = Path.cwd().parents[2] / 'data'
    # print(data_path)
    camera_path = data_path / 'camera_calibration' / 'left.yaml'
    kinematic_path = data_path / 'data_20250701' / '1' / 'regular' / 'kinematic' / 'PSM1' / '0.json'
    with open(camera_path, 'r') as f:
        data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, data)

    with open(kinematic_path, 'r') as f:
        data_kin = json.load(f)
    kin_info = datacls_from_dict(KinematicInfo, data_kin)

    cam_dict = asdict(cam_info) # convert back
    pass
