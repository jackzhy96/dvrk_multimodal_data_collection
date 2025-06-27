from typing import Union, List, Tuple
from pathlib import Path
import shutil
import numpy as np
import yaml
import json


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


def load_camera_yaml(path: Union[Path, str])->np.ndarray:
    '''
    load the camera projection matrix from the yaml file
    path: path to the yaml file
    output: 3x4 camera projection matrix
    '''
    file_path = convert_pathlib_type(path)
    with open(file_path, 'r') as f:
        test_data = yaml.safe_load(f)
        # camera_mtx = test_data['camera_matrix']
        proj_mtx = test_data['projection_matrix']
    # K = np.array(camera_mtx['data']).reshape(3, 3)
    P = np.array(proj_mtx['data']).reshape(3, 4)
    # K[1, 2] = K[1, 2] - 60  ## comment it when getting new calibration matrix
    # return K
    # P[1, 2] = P[1, 2] - 60 ## comment it when getting new calibration matrix
    return P


def load_stereo_proj_mtx(path: Union[Path,str])->List[np.ndarray]:
    '''
    Load the camera projection matrix of the stereo camera from the calibration folder
    path: path to camera calibration folder
    output: camera projection matrix for both left and right cameras
    '''
    camera_calibration_path = convert_pathlib_type(path)
    num_files = sum(1 for p in camera_calibration_path.iterdir() if p.is_file())
    project_mtx = []
    if num_files == 2:
        file_names = ['left', 'right']
        for file_name in file_names:
            file_path = camera_calibration_path / f'{file_name}.yaml'
            P_cam = load_camera_yaml(file_path)
            P_cam = P_cam[0:3, 0:3]
            project_mtx.append(P_cam)
    elif num_files == 1:
        file_name = 'camera'
        file_path = camera_calibration_path / f'{file_name}.yaml'
        P_cam = load_camera_yaml(file_path)
        P_cam = P_cam[0:3, 0:3]
        project_mtx.append(P_cam)
    else:
        raise ValueError('Camera calibration folder have too many / no calibration files')
    return project_mtx


def load_json_cp(path: Union[Path, str], arm_names:List[str])->dict:
    '''
    Load the json file including the measured_cp and measured_cv
    path: path to the json file
    arm_names: list of arm names
    output: dictionary including the positons and velocities of the arms' end-effector
    '''
    file_path = convert_pathlib_type(path)
    cp_info = dict()
    with open(file_path, 'r') as f:
        data = json.load(f)
    if len(arm_names) == 0:
        raise ValueError('No arm names provided!')
    for arm_name in arm_names:
        arm_info = dict()
        arm_info['R'] = np.array(data[arm_name]['R']).reshape(3, 3)
        arm_info['t'] = np.array(data[arm_name]['t'])
        arm_info['w'] = np.array(data[f'{arm_name}_cv']['linear'])
        arm_info['v'] = np.array(data[f'{arm_name}_cv']['angular'])
        cp_info[arm_name] = arm_info
    return cp_info


def glob_sorted_frame(path: Union[Path, str])->List[Path]:
    '''
    glob the sorted file names of the frames
    path: path to the folder containing the frames
    output: list of sorted file names
    '''
    data_path = convert_pathlib_type(path)
    cp_file_list = sorted(data_path.glob('*'), key=lambda p: int(p.stem.replace("frame", "")))
    return cp_file_list

if __name__ == '__main__':
    pass