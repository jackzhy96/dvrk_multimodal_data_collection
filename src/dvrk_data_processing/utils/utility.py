from typing import Union, List, Tuple
from pathlib import Path
import shutil
import numpy as np
import yaml
import json
import cv2
from dvrk_data_processing.utils.data_load_config import CameraInfo, KinematicInfo, datacls_from_dict, CameraInfoProcessed, CPInfo
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


def load_camera_param_yaml(path: Union[Path, str])->CameraInfoProcessed:
    '''
    load the camera parameters from the yaml file
    path: path to the yaml file
    output: loaded data class instance of CameraInfoProcessed
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        test_data = yaml.safe_load(f)
    cam_info = datacls_from_dict(CameraInfo, test_data)
    K = np.array(cam_info.camera_matrix.data).reshape(3,3)
    D = np.array(cam_info.distortion_coefficients.data).reshape(-1,1)
    R_c = np.array(cam_info.R_stereo.data).reshape(3,3)
    rvec, _ = cv2.Rodrigues(R_c)
    t_c = np.array(cam_info.T_stereo.data).reshape(-1,1)
    img_width = cam_info.image_width
    img_height = cam_info.image_height
    dict_cam_param = {'K': K, 'D': D, 'rvec': rvec, 'tvec': t_c,'R_c': R_c, 't_c': t_c,
                      'image_width': img_width, 'image_height': img_height}
    camera_params = datacls_from_dict(CameraInfoProcessed, dict_cam_param)
    return camera_params


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


def load_json_cp(path: Union[Path, str], arm_name:str)->CPInfo:
    '''
    Load the json file including the measured_cp and measured_cv
    path: path to the json file
    arm_names: the name of the arms to be loaded
    output: dictionary including the positons and velocities of the arms' end-effector
    '''
    file_path = convert_pathlib_type(path)
    arm_info = dict()
    with open(file_path, 'r') as f:
        data = json.load(f)
    kin_info = datacls_from_dict(KinematicInfo, data)
    arm_info = dict()
    rot = R.from_quat(kin_info.arm.measured_data.orientation)
    arm_info['R'] = rot.as_matrix()
    arm_info['t'] = np.array(kin_info.arm.measured_data.position)
    arm_info['w'] = np.array(kin_info.arm.measured_data.cartesian_velocity)[0:3]
    arm_info['v'] = np.array(kin_info.arm.measured_data.cartesian_velocity)[3:6]
    arm_info['arm_name'] = arm_name
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

if __name__ == '__main__':
    pass