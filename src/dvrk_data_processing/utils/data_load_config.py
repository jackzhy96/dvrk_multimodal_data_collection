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
    velocity: List[float] = None  # present in both old/new formats, optional for safety


@dataclass
class MeasuredCvConfig:
    """Measured Cartesian velocity — separate linear and angular components."""
    linear: List[float]
    angular: List[float]


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
    """Generic measured data config — matches new JSON field names."""
    measured_cp: ArmMeasuredCpConfig
    measured_js: ArmJsConfig
    local_measured_cp: ArmLocalCpConfig = None  # local coordinates (base frame)
    measured_cv: MeasuredCvConfig = None         # Cartesian velocity (linear + angular)


@dataclass
class PSMMeasuredLoadConfig:
    """PSM measured data — fields match JSON keys: measured_cp, measured_js, etc."""
    measured_cp: PSMMeasuredCpConfig
    measured_js: ArmJsConfig
    local_measured_cp: ArmLocalCpConfig = None  # local coordinates (base frame)
    measured_cv: MeasuredCvConfig = None         # Cartesian velocity (linear + angular)


@dataclass
class ECMMeasuredLoadConfig:
    """ECM measured data — fields match JSON keys: measured_cp, measured_js, etc."""
    measured_cp: ECMMeasuredCpConfig
    measured_js: ArmJsConfig
    local_measured_cp: ArmLocalCpConfig = None  # local coordinates (base frame)
    measured_cv: MeasuredCvConfig = None         # Cartesian velocity (linear + angular)


@dataclass
class DesiredLoadConfig:
    """Setpoint/desired data — setpoint_cp is optional (absent in new ECM format)."""
    setpoint_js: ArmJsConfig
    setpoint_cp: ArmSetpointCpConfig = None  # not present in new ECM format


@dataclass
class ArmLoadConfig:
    """Generic arm config — matches the normalized new JSON structure."""
    measured_data: MeasuredLoadConfig
    setpoint_data: DesiredLoadConfig
    jaw: JawLoadConfig = None           # PSM only; inside arm in new format
    measured_frequency: float = None    # PSM only; inside arm in new format


@dataclass
class PSMLoadConfig:
    """PSM arm config — local_cp moved into measured_data as local_measured_cp."""
    measured_data: PSMMeasuredLoadConfig
    setpoint_data: DesiredLoadConfig
    jaw: JawLoadConfig = None           # inside arm in new format, moved here from top level in old format
    measured_frequency: float = None    # inside arm in new format, moved here from top level in old format


@dataclass
class ECMLoadConfig:
    """ECM arm config — local_cp moved into measured_data as local_measured_cp."""
    measured_data: ECMMeasuredLoadConfig
    setpoint_data: DesiredLoadConfig


@dataclass
class KinematicInfo:
    """Generic kinematic info — wraps the arm config."""
    arm: ArmLoadConfig


@dataclass
class PSMInfo:
    """PSM kinematic info — jaw and measured_frequency are inside arm (PSMLoadConfig)."""
    arm: PSMLoadConfig


@dataclass
class ECMInfo:
    """ECM kinematic info — no jaw or measured_frequency."""
    arm: ECMLoadConfig


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
    camera_path = data_path / 'camera_calibration' / 'left.yaml'
    # Example: load kinematic data from test_data (both old and new formats are list-wrapped)
    kinematic_path = data_path / 'test_data' / 'old' / 'raw' / '0' / 'kinematic' / 'PSM1' / '0.json'
    with open(camera_path, 'r') as f:
        data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, data)

    with open(kinematic_path, 'r') as f:
        data_kin = json.load(f)
    # Unwrap list if needed (both old and new formats are list-wrapped)
    if isinstance(data_kin, list) and len(data_kin) > 0:
        data_kin = data_kin[0]
    # Normalize old format: move jaw and measured_frequency from top level into arm
    if 'jaw' in data_kin and 'arm' in data_kin:
        data_kin['arm']['jaw'] = data_kin.pop('jaw')
    if 'measured_frequency' in data_kin and 'arm' in data_kin:
        data_kin['arm']['measured_frequency'] = data_kin.pop('measured_frequency')
    kin_info = datacls_from_dict(PSMInfo, data_kin)

    cam_dict = asdict(cam_info)  # convert back
    pass
