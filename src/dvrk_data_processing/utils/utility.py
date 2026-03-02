from typing import Union, List, Tuple, Dict
from pathlib import Path
import logging
import numpy as np
import yaml
import json
import cv2
import shutil
from dvrk_data_processing.utils.data_load_config import (CameraInfo, PSMInfo, ECMInfo, datacls_from_dict, HandEyeInfo,
                                                         HandEyeLoadConfig, CameraInfoProcessed, MonoCameraInfoProcessed,
                                                         PSMCPInfo, ECMCPInfo)
from scipy.spatial.transform import Rotation as R


def convert_pathlib_type(folder:Union[Path,str])->Path:
    '''
    Converts a pathlib Path to a Path object.
    folder: your selected path
    output: your selected path in Pathlib Path object
    '''
    if isinstance(folder, Path):
        new_path = folder
    elif isinstance(folder, str):
        new_path = Path(folder)
    else:
        raise TypeError('Folder must be of type Path or str')
    return new_path


def clear_folder(folder: Union[Path,str])->None:
    '''
    Clear the folder contents.
    folder: path of folder to clear
    '''
    new_path = convert_pathlib_type(folder)
    input(f'You are about to remove all files in {new_path}, press ENTER to continue')
    for item_obj in new_path.iterdir():
        if item_obj.is_dir():
            shutil.rmtree(item_obj, ignore_errors=True)
            print(f'Removing subfolder: {item_obj}')
        else:
            item_obj.unlink(missing_ok=True)
            print(f'Removing file: {item_obj}')


def create_folder(folder: Union[Path,str])->None:
    '''
    Create a new folder.
    folder: path of new folder
    '''
    new_path = convert_pathlib_type(folder)
    new_path.mkdir(parents=True, exist_ok=True)
    print(f'Created new path: {new_path}')


def load_camera_mtx_yaml(path: Union[Path, str])->np.ndarray:
    '''
    load the 3x3 camera projection matrix from the yaml file
    path: path to the yaml file
    output: 3x3 camera projection matrix
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, test_data)
    camera_mtx = np.array(cam_info.camera_matrix.data).reshape(3,3)
    return camera_mtx


def load_stereo_proj_mtx(path: Union[Path,str])->List[np.ndarray]:
    '''
    Load the 3x3 camera matrix of the stereo camera from the calibration folder
    path: path to camera calibration folder
    output: 3x3 camera projection matrix for both left and right cameras
    '''
    camera_calibration_path = convert_pathlib_type(path)
    num_files = sum(1 for p in camera_calibration_path.iterdir() if p.is_file())
    project_mtx = []
    if num_files == 2:
        file_names = ['left', 'right']
        for file_name in file_names:
            file_path = camera_calibration_path / f'{file_name}.yaml'
            K_cam = load_camera_mtx_yaml(file_path)
            project_mtx.append(K_cam)
    elif num_files == 1:
        file_name = 'camera'
        file_path = camera_calibration_path / f'{file_name}.yaml'
        K_cam = load_camera_mtx_yaml(file_path)
        project_mtx.append(K_cam)
    else:
        raise ValueError('Camera calibration folder have too many / no calibration files')
    return project_mtx


def load_raw_stereo_camera_param_yaml(path: Union[Path, str])->CameraInfo:
    '''
    load the camera parameters from the yaml file without any processing (for stereo camera)
    path: path to the yaml file
    output: loaded data class instance of CameraInfo
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, test_data)
    return cam_info


def load_stereo_camera_param_yaml(path: Union[Path, str])->CameraInfoProcessed:
    '''
    load the camera parameters from the yaml file (for stereo camera)
    path: path to the yaml file
    output: loaded data class instance of CameraInfoProcessed
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, test_data)
    K = np.array(cam_info.camera_matrix.data).reshape(3,3)
    D = np.array(cam_info.distortion_coefficients.data).reshape(-1,1)
    if path.stem == 'left':
        R_c = np.eye(3)
        t_c = np.zeros((3,1))
    else:
        R_c = np.array(cam_info.R_stereo.data).reshape(3,3)
        t_c = np.array(cam_info.T_stereo.data).reshape(-1,1)
    img_width = cam_info.image_width
    img_height = cam_info.image_height
    dict_cam_param = {'K': K, 'D': D,'R_c': R_c, 't_c': t_c, 'image_width': img_width, 'image_height': img_height}
    camera_params = datacls_from_dict(CameraInfoProcessed, dict_cam_param)
    return camera_params


def load_mono_camera_param_yaml(path: Union[Path, str])->MonoCameraInfoProcessed:
    '''
    load the camera parameters from the yaml file (for mono camera)
    path: path to the yaml file
    output: loaded data class instance of MonoCameraInfoProcessed
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, test_data)
    K = np.array(cam_info.camera_matrix.data).reshape(3,3)
    D = np.array(cam_info.distortion_coefficients.data).reshape(-1,1)
    img_width = cam_info.image_width
    img_height = cam_info.image_height
    dict_cam_param = {'K': K, 'D': D,'image_width': img_width, 'image_height': img_height}
    camera_params = datacls_from_dict(MonoCameraInfoProcessed, dict_cam_param)
    return camera_params


def _normalize_kinematic_json(data: Union[list, dict]) -> dict:
    '''
    Normalize raw kinematic JSON to a canonical dict structure.
    Handles two issues:
      1. List wrapping: both old and new formats wrap data in a list [{ ... }]
      2. Old format placement: jaw and measured_frequency are at the top level
         in the old format but inside 'arm' in the new format — move them into arm.
    data: raw JSON data (list or dict)
    output: normalized dict with structure matching the new format
    '''
    # Unwrap list if needed (both old and new formats are list-wrapped)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Kinematic JSON list is empty")
        data = data[0]

    # Detect old format: jaw and measured_frequency at top level (siblings of arm)
    # In new format, they are inside arm already
    if 'arm' in data:
        if 'jaw' in data:
            data['arm']['jaw'] = data.pop('jaw')
        if 'measured_frequency' in data:
            data['arm']['measured_frequency'] = data.pop('measured_frequency')

    return data


def load_json_cp(path: Union[Path, str], arm_name: str) -> Union[PSMCPInfo, ECMCPInfo, None]:
    '''
    Load the json file including both measured_cp, measured_cv (only for PSM), local_measured_cp
    and measured_js data. Supports both old and new kinematic JSON formats:
      - Old format: list-wrapped, jaw/measured_frequency at top level
      - New format: list-wrapped, jaw/measured_frequency inside arm
    path: path to the json file
    arm_name: name of the arm (ECM, PSM1, PSM2, PSM3)
    output: loaded data class instance of PSMCPInfo or ECMCPInfo
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, "r") as f:
        raw_data = json.load(f)

    # Normalize to canonical dict structure (unwrap list, move old-format fields)
    data = _normalize_kinematic_json(raw_data)

    arm_info = dict()
    if arm_name.upper() == 'ECM':
        kin_info = datacls_from_dict(ECMInfo, data)
        rot_world = R.from_quat(kin_info.arm.measured_data.measured_cp.orientation)
        arm_info["R"] = rot_world.as_matrix()
        arm_info["t"] = np.array(kin_info.arm.measured_data.measured_cp.position)
    elif 'PSM' in arm_name.upper():
        kin_info = datacls_from_dict(PSMInfo, data)
        # -------- measured_cp (world coordinates) --------
        rot_world = R.from_quat(kin_info.arm.measured_data.measured_cp.orientation)
        arm_info["R"] = rot_world.as_matrix()
        arm_info["t"] = np.array(kin_info.arm.measured_data.measured_cp.position)
        arm_info["w"] = np.array(kin_info.arm.measured_data.measured_cp.velocity)[0:3]
        arm_info["v"] = np.array(kin_info.arm.measured_data.measured_cp.velocity)[3:6]
        arm_info['measured_frequency'] = kin_info.arm.measured_frequency
    else:
        raise ValueError(f"Unknown arm name: {arm_name}")
    # -------- local_measured_cp (base coordinates) --------
    rot_local = R.from_quat(kin_info.arm.measured_data.local_measured_cp.orientation)
    arm_info["R_local"] = rot_local.as_matrix()
    arm_info["t_local"] = np.array(kin_info.arm.measured_data.local_measured_cp.position)
    # -------- meta --------
    arm_info["arm_name"] = arm_name
    if arm_name.upper() == 'ECM':
        cp_info = datacls_from_dict(ECMCPInfo, arm_info)
    elif 'PSM' in arm_name.upper():
        cp_info = datacls_from_dict(PSMCPInfo, arm_info)
    else:
        raise ValueError(f"Unknown arm name: {arm_name}")
    return cp_info


def glob_sorted_frame(path: Union[Path, str])->List[Path]:
    '''
    glob the sorted file names of the frames
    path: path to the folder containing the frames
    output: list of sorted file names
    '''
    data_path = convert_pathlib_type(path)
    cp_file_list = sorted(data_path.glob('*'), key=lambda p: int(p.stem))
    return cp_file_list


def get_sorted_names(path: Union[Path, str])->List[str]:
    '''
    Get the sorted file names in a given folder
    path: path to the folder containing the files
    output: list of sorted file names
    '''
    data_path = convert_pathlib_type(path)
    file_path = glob_sorted_frame(data_path)
    file_names = [p.name for p in file_path]
    return file_names


def copy_folder(src: Union[Path, str], dst: Union[Path, str])->None:
    '''
    Copy the folder from src to dst
    src: source folder path
    dst: destination folder path
    '''
    src_path = convert_pathlib_type(src)
    dst_path = convert_pathlib_type(dst)
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    print(f'Copied folder from {src_path} to {dst_path}')


def skew(vec: np.ndarray)->np.ndarray:
    '''
    vec: The vector to be extended to a 3x3 skew symmetric matrix
    output: The extended 3x3 skew symmetric matrix
    '''
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def load_handeye_json(path: Union[Path, str]) -> HandEyeInfo:
    '''
    Load the hand-eye calibration matrix from the json file
    path: path to the json file
    output: loaded data class instance of HandEyeInfo
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    handeye_json = datacls_from_dict(HandEyeLoadConfig, data)
    handeye_dict = dict()
    tranform_matrix = np.array(handeye_json.measured_cp)
    if tranform_matrix.shape != (4, 4):
        raise ValueError(f"Incorrect Hand-Eye Calibration File {file_path}, the transformation matrix is not a 4*4 matrix")
    handeye_dict['name'] = handeye_json.name
    handeye_dict['transformation_matrix'] = tranform_matrix
    handeye_info = datacls_from_dict(HandEyeInfo, handeye_dict)
    return handeye_info


def load_handeye_dict(calib_folder: Union[Path, str],
                      arm_names: List[str]) -> Dict[str, np.ndarray]:
    '''
    Load the hand-eye calibration matrix from the json files
    calib_folder: path to the folder containing the json files
    arm_names: list of arm names
    output: dictionary of hand-eye calibration matrices, key: arm name, value: 4*4 transformation matrix
    file name example: <arm name>-registration-dVRK.json
    '''
    calib_path = convert_pathlib_type(calib_folder)
    calibration_dict = dict()
    for arm in arm_names:
        he_file = calib_path / f"{arm}-registration-dVRK.json"
        if not he_file.exists():
            raise FileNotFoundError(f"Cannot find Hand-Eye file: {he_file}")
        handeye_info = load_handeye_json(he_file)
        calibration_dict[handeye_info.name] = handeye_info.transformation_matrix
    return calibration_dict


def load_ecm_transformation_matrix(path: Union[Path, str]) -> np.ndarray:
    '''
    Load the ECM transformation matrix from local_measured_cp topic in the given json file.
    Uses load_json_cp() which handles both old and new kinematic JSON formats.
    path: path to the json file
    output: 4*4 transformation matrix
    '''
    file_path = convert_pathlib_type(path)
    ecm_info = load_json_cp(file_path, "ECM")
    t_ecm = ecm_info.t_local
    R_ecm = ecm_info.R_local
    T_we = np.eye(4)
    T_we[:3, :3] = R_ecm
    T_we[:3,  3] = t_ecm
    return T_we


###############################################################################
# Tool-tip offset utilities
###############################################################################

def parse_tool_tip_offsets(raw_offsets: Union[None, dict],
                           arm_names: List[str]) -> Dict[str, np.ndarray]:
    '''
    Parse per-PSM tool-tip offset config into a dict of 4x4 matrices.
    Each entry can be:
      - a flat list of 16 floats → reshaped to 4x4
      - None / missing → identity 4x4
    raw_offsets: the tool_tip_offset dict from the Hydra config (keyed by arm name)
    arm_names: list of arm names to parse offsets for (e.g. ['PSM1', 'PSM2'])
    output: dict mapping arm_name → 4x4 np.ndarray transformation matrix
    '''
    offsets = {}
    for arm in arm_names:
        if raw_offsets is None:
            # No offsets configured at all — use identity for every arm
            offsets[arm] = np.eye(4)
            continue

        # Retrieve this arm's offset (may be None, missing, or a list)
        arm_offset = raw_offsets.get(arm, None)
        if arm_offset is None or (isinstance(arm_offset, (list, tuple)) and len(arm_offset) == 0):
            # None or empty list → identity
            offsets[arm] = np.eye(4)
        else:
            arr = np.array(arm_offset, dtype=np.float64)
            if arr.size != 16:
                raise ValueError(
                    f"tool_tip_offset for {arm} must have exactly 16 elements "
                    f"(4x4 matrix), got {arr.size}"
                )
            offsets[arm] = arr.reshape(4, 4)
    return offsets


def resolve_per_arm_weight_configs(cfg_preprocess, arm_names: List[str]) -> Dict[str, Dict]:
    '''
    Resolve per-arm weight configs for kinematic heatmap generation.
    For each arm, use its dedicated weight_config_<ARM> if present,
    otherwise fall back to the global weight_config.
    This follows the same pattern as parse_tool_tip_offsets().
    cfg_preprocess: the preprocess section of the Hydra config (KinematicMapConfig)
    arm_names: list of arm names to resolve configs for (e.g. ['PSM1', 'PSM2'])
    output: dict mapping arm_name → {sigma_x, sigma_y, advanced_weight, tol_dist}
    '''
    # Global defaults from weight_config
    global_wcfg = cfg_preprocess.weight_config
    global_dict = {
        'sigma_x': float(global_wcfg.sigma_x),
        'sigma_y': float(global_wcfg.sigma_y),
        'advanced_weight': bool(global_wcfg.advanced_weight),
        'tol_dist': float(global_wcfg.tol_dist),
    }

    weight_configs = {}
    for arm in arm_names:
        # Look up the per-PSM override attribute (e.g. weight_config_PSM1)
        per_arm_attr = f"weight_config_{arm}"
        per_arm_wcfg = getattr(cfg_preprocess, per_arm_attr, None)

        if per_arm_wcfg is None:
            # No per-arm override — use global defaults
            weight_configs[arm] = global_dict.copy()
        else:
            # Per-arm override exists — extract its values
            weight_configs[arm] = {
                'sigma_x': float(per_arm_wcfg.sigma_x),
                'sigma_y': float(per_arm_wcfg.sigma_y),
                'advanced_weight': bool(per_arm_wcfg.advanced_weight),
                'tol_dist': float(per_arm_wcfg.tol_dist),
            }
    return weight_configs


def apply_tool_tip_offset(R_psm: np.ndarray, t_psm: np.ndarray,
                           T_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Apply a 4x4 tool-tip offset to a PSM pose.
    Computes T_actual = T_psm @ T_offset, then extracts the new R and t.
    This "extends" the PSM's rotation and translation to account for the
    physical offset between the PSM's reported end-effector and the actual tool tip.
    R_psm: 3x3 rotation matrix of the PSM
    t_psm: 3-element translation vector of the PSM
    T_offset: 4x4 tool-tip offset transformation matrix
    output: (R_new, t_new) — the transformed rotation and translation
    '''
    # Build the PSM's 4x4 homogeneous transformation
    T_psm = np.eye(4)
    T_psm[:3, :3] = R_psm
    T_psm[:3, 3] = t_psm

    # Apply offset: the tool tip is at T_psm @ T_offset
    T_actual = T_psm @ T_offset

    R_new = T_actual[:3, :3]
    t_new = T_actual[:3, 3]
    return R_new, t_new


###############################################################################
# Camera projection and kinematic heatmap utilities
# (shared by gen_kinematic_heatmap_dVRK.py and gen_kinematic_heatmap_handeye.py)
###############################################################################

def pixel_coord_check(pixel_coord: np.ndarray, img_w: int, img_h: int) -> None:
    '''
    Check if the pixel coordinates are within the image scope.
    Logs a warning for any coordinate that falls outside the visible area.
    pixel_coord: 2D pixel coordinates (u,v), dimension is Nx2
    img_w: width of the image
    img_h: height of the image
    '''
    for i in range(len(pixel_coord)):
        u = pixel_coord[i][0]
        v = pixel_coord[i][1]
        if not (0 <= u < img_w and 0 <= v < img_h):
            logging.warning(
                f'pair {i} pixel coordinate ({u:.1f}, {v:.1f}) '
                f'is out of range (image {img_w}x{img_h})'
            )


def cam_project_3d_to_2d(coord_3d: np.ndarray, cam_param: CameraInfoProcessed,
                          cam_offset: Union[None, np.ndarray]) -> np.ndarray:
    '''
    Project 3D point to 2D pixel coordinates, with optional camera offset rotation.
    coord_3d: 3D point in the camera coordinate system, shape (3,) or (N,3)
    cam_param: camera parameters (K, D, R_c, t_c, dimensions)
    cam_offset: optional 3x3 rotation matrix for camera coordinate frame correction
    output: 2D pixel coordinates (u,v), dimension Nx2
    '''
    if cam_offset is None:
        cam_offset = np.eye(3)
    R_cam = cam_param.R_c
    t_cam = cam_param.t_c.reshape(-1, 1)
    # Apply camera offset rotation to both rotation and translation
    R_cam = cam_offset @ R_cam
    rvec, _ = cv2.Rodrigues(R_cam)
    tvec = cam_offset @ t_cam
    pixel_coord, _ = cv2.projectPoints(coord_3d, rvec, tvec, cam_param.K, cam_param.D)
    pixel_coord_2d = pixel_coord.reshape(-1, 2)
    pixel_coord_check(pixel_coord_2d, cam_param.image_width, cam_param.image_height)
    return pixel_coord_2d


def d_weight(xyz: np.ndarray, weight_adv: bool, tol_dist: float = 0.02) -> float:
    '''
    Calculate the depth-dependent weight for the prediction term in heatmap generation.
    xyz: 3D point in the camera coordinate system
    weight_adv: whether to use the advanced exponential decay weight
    tol_dist: tolerance distance offset subtracted from the norm
    output: calculated weight scalar
    '''
    xyz_norm = np.linalg.norm(xyz)
    s = xyz_norm - tol_dist
    if weight_adv:
        # Advanced exponential decay — concentrates weight near the camera
        d = np.exp((-s + np.exp(-s)) / 2.0)
        return d
    else:
        # Simple inverse-distance weight with a minimum clamp to avoid division by zero
        if s < 4e-4:
            s = 4e-4
        d = 1.0 / (1000.0 * s)
        return d


def gen_heatmap(u: float, v: float, u_next: float, v_next: float,
                xyz: np.ndarray, sigma_x: float, sigma_y: float,
                img_w: int, img_h: int, weight_adv: bool, tol_dist: float,
                mgrid_cache: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
    '''
    Generate a kinematic heatmap combining current position and predicted next position.
    Uses 2D Gaussian kernels centered at (u,v) and (u_next,v_next), weighted by depth.
    u, v: pixel coordinates of the current point
    u_next, v_next: pixel coordinates of the predicted next point
    xyz: 3D point for depth-dependent weighting
    sigma_x, sigma_y: Gaussian kernel standard deviations
    img_w, img_h: image dimensions
    weight_adv: whether to use advanced depth weighting
    tol_dist: distance tolerance for weight calculation
    mgrid_cache: optional pre-computed (y_grid, x_grid) to avoid recomputing per frame
    output: 2D heatmap array of shape (img_h, img_w)
    '''
    # Use cached grids if provided for performance (avoids recomputation per frame)
    if mgrid_cache is not None:
        y, x = mgrid_cache
    else:
        y, x = np.mgrid[0:img_h, 0:img_w]

    mp_current = np.exp(-(((x - u) ** 2 / sigma_x ** 2) + ((y - v) ** 2 / sigma_y ** 2)))
    mp_predict = np.exp(-(((x - u_next) ** 2 / sigma_x ** 2) + ((y - v_next) ** 2 / sigma_y ** 2)))
    d = d_weight(xyz, weight_adv, tol_dist)
    return mp_current + d * mp_predict


if __name__ == '__main__':
    pass