from typing import Union, List, Tuple
from pathlib import Path
import shutil
import numpy as np
import yaml
import json
import cv2
import shutil
from dvrk_data_processing.utils.data_load_config import (CameraInfo, KinematicInfo, datacls_from_dict,
                                                         CameraInfoProcessed, CPInfo, MonoCameraInfoProcessed)
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


def load_json_cp(path: Union[Path, str], arm_name: str) -> CPInfo:
    """
    Load the json file including both measured_cp (world) and local_cp (base) data.
    The original return fields R, t, w, v, arm_name are kept unchanged.
    New fields added:
        • R_local : 3×3 rotation matrix of local_cp
        • t_local : 3×1 translation vector of local_cp
    """
    file_path = convert_pathlib_type(path)
    with open(file_path, "r") as f:
        data = json.load(f)

    kin_info = datacls_from_dict(KinematicInfo, data)
    arm_info = {}

    # -------- measured_cp (world coordinates) --------
    rot_world = R.from_quat(kin_info.arm.measured_data.cp.orientation)
    arm_info["R"] = rot_world.as_matrix()
    arm_info["t"] = np.array(kin_info.arm.measured_data.cp.position)
    arm_info["w"] = np.array(kin_info.arm.measured_data.cp.velocity)[0:3]
    arm_info["v"] = np.array(kin_info.arm.measured_data.cp.velocity)[3:6]

    # -------- local_cp (base coordinates) --------
    rot_local = R.from_quat(kin_info.arm.local_cp.orientation)
    arm_info["R_local"] = rot_local.as_matrix()
    arm_info["t_local"] = np.array(kin_info.arm.local_cp.position)

    # -------- meta --------
    arm_info["arm_name"] = arm_name

    cp_info = datacls_from_dict(CPInfo, arm_info)
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


#hand-eye
def load_handeye_json(path: Union[Path, str]) -> np.ndarray:

    file_path = convert_pathlib_type(path)
    with open(file_path, "r") as f:
        data = json.load(f)

    mat = np.array(data.get("measured_cp", data), dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"{file_path}  not a 4*4 matrix")
    return mat


def load_handeye_dict(calib_folder: Union[Path, str],
                      arm_names: List[str]) -> dict[str, np.ndarray]:
    """
     {'PSM1':4*4, 'PSM2':4*4, ...}
    file name: <arm>-registration-dVRK.json
    """
    calib_path = convert_pathlib_type(calib_folder)
    he_dict = {}
    for arm in arm_names:
        he_file = calib_path / f"{arm}-registration-dVRK.json"
        if not he_file.exists():
            raise FileNotFoundError(f"Lack Hand-Eye file: {he_file}")
        he_dict[arm] = load_handeye_json(he_file)
    return he_dict

# =========  ECM measured_cp   =========
def load_ecm_mat(path: Union[Path, str]) -> np.ndarray:
    """ ECM local measured_cp, 4*4 (World ← ECM-Tip)"""
    with open(convert_pathlib_type(path), "r") as f:
        obj = json.load(f)
    p = np.array(obj["arm"]["local_cp" ]["position"], dtype=float)
    q = np.array(obj["arm"]["local_cp" ]["orientation"], dtype=float)
    R_we = R.from_quat(q).as_matrix()
    T_we = np.eye(4)
    T_we[:3, :3] = R_we
    T_we[:3,  3] = p
    return T_we


if __name__ == '__main__':
    pass