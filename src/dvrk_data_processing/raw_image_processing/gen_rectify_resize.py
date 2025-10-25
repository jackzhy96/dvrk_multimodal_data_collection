from dataclasses import dataclass, asdict
from typing import Union, Tuple, List
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import copy
from dvrk_data_processing.utils.hydra_config import PathConfig, ResizeRectifyConfig
from dvrk_data_processing.utils.utility import (create_folder, clear_folder, copy_folder, get_sorted_names,
                                                load_stereo_camera_param_yaml, load_mono_camera_param_yaml,
                                                load_raw_stereo_camera_param_yaml)
from dvrk_data_processing.utils.data_load_config import CameraInfoProcessed
from tqdm import tqdm
import yaml
import cv2
import json


@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: ResizeRectifyConfig
    workspace: str
    camera_names: List[str]
    camera_calibration_path: Union[Path, str]
    camera_offset: Union[None, List[float]]


def scale_camera_matrix(K: np.ndarray, scale_x: float, scale_y: float)->np.ndarray:
    '''
    scale the camera matrix.
    scale_x: scale factor for x-axis
    scale_y: scale factor for y-axis
    output: scaled camera matrix
    '''
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


def scale_flat_camera_matrix(K: List[float], scale_x: float, scale_y: float)->List[float]:
    '''
    scale the flat camera matrix.
    scale_x: scale factor for x-axis
    scale_y: scale factor for y-axis
    output: scaled flat camera matrix
    '''
    K_scaled = K.copy()
    K_scaled[0] *= scale_x  # fx
    K_scaled[2] *= scale_x  # cx
    K_scaled[4] *= scale_y  # fy
    K_scaled[5] *= scale_y  # cy
    return K_scaled


def get_stereo_camera_calibration_scaled(camera_calibration_path:Path, camera_names:List[str], original_size:List[int],
                                         new_size:List[int])->Tuple[CameraInfoProcessed, CameraInfoProcessed]:
    '''
    Get stereo camera calibration parameters.
    camera_calibration_path: path to the camera calibration file
    camera_names: list of camera names
    original_size: original size of the image
    new_size: new size of the image
    output: camera parameters for both cameras, using CameraInfo dataclass
    '''
    camera_param_list = []
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    for i_cam in range(len(camera_names)):
        camera_file_path = camera_calibration_path / f'{camera_names[i_cam]}.yaml'
        if len(camera_names) == 2:
            camera_params = load_stereo_camera_param_yaml(camera_file_path)
        elif len(camera_names) == 1:
            camera_params = load_mono_camera_param_yaml(camera_file_path)
        else:
            raise ValueError('Only support single or stereo camera setup.')
        camera_params.K = scale_camera_matrix(camera_params.K, scale_x, scale_y)
        camera_params.image_width = new_size[0]
        camera_params.image_height = new_size[1]
        camera_param_list.append(camera_params)
    return camera_param_list[0], camera_param_list[1]


def get_stereo_camera_calibration_original(camera_calibration_path:Path, camera_names:List[str])\
        ->Tuple[CameraInfoProcessed, CameraInfoProcessed]:
    '''
    Get stereo camera calibration parameters at original size.
    camera_calibration_path: path to the camera calibration file
    camera_names: list of camera names
    output: camera parameters for both cameras, using CameraInfo dataclass
    '''
    camera_param_list = []
    for i_cam in range(len(camera_names)):
        camera_file_path = camera_calibration_path / f'{camera_names[i_cam]}.yaml'
        if len(camera_names) == 2:
            camera_params = load_stereo_camera_param_yaml(camera_file_path)
        elif len(camera_names) == 1:
            camera_params = load_mono_camera_param_yaml(camera_file_path)
        else:
            raise ValueError('Only support single or stereo camera setup.')
        camera_param_list.append(camera_params)
    return camera_param_list[0], camera_param_list[1]


def copy_stereo_camera_calibration(camera_calibration_path:Path, camera_names:List[str], output_folder:Path,
                            original_size:List[int], new_size:List[int])->None:
    '''
    Copy the camera calibration parameters from the original folder to the new folder and implement scale factors to the intrinsic matrix.
    camera_calibration_path: path to the camera calibration file
    camera_names: list of camera names
    output_folder: path to the output folder
    original_size: original size of the image
    new_size: new size of the image
    '''
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    for camera_name in camera_names:
        camera_file_path = camera_calibration_path / f'{camera_name}.yaml'
        raw_cam_info = load_raw_stereo_camera_param_yaml(camera_file_path)
        new_cam_info = copy.deepcopy(raw_cam_info)
        new_cam_info.camera_matrix.data = scale_flat_camera_matrix(raw_cam_info.camera_matrix.data, scale_x, scale_y)
        new_cam_info.image_width = new_size[0]
        new_cam_info.image_height = new_size[1]
        new_cam_dict = asdict(new_cam_info)
        new_camera_calibration_path = output_folder / camera_calibration_path.name
        if not new_camera_calibration_path.exists():
            create_folder(new_camera_calibration_path)
        new_cam_file_path = new_camera_calibration_path / f'{camera_name}.yaml'
        with open(new_cam_file_path, 'w') as f:
            yaml.safe_dump(new_cam_dict, f)


def get_rectify_map_new_size(camera_param_left:CameraInfoProcessed, camera_param_right:CameraInfoProcessed,
                             original_size:List[int], new_size:List[int])-> tuple:
    '''
    Get rectify map, also get ROI for later crop so that we can remove the black border.
    camera_param_left: camera parameters for left camera
    camera_param_right: camera parameters for right camera
    new_size: new size of the image
    output: rectify map for both cameras, and valid pixels for both cameras
    '''

    ############## uncomment if on-campus ################
    # new_size = tuple(new_size)
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    Ks_L = scale_camera_matrix(camera_param_left.K, scale_x, scale_y)
    Ks_R = scale_camera_matrix(camera_param_right.K, scale_x, scale_y)
    R1, R2, P1, P2, Q, _, _= cv2.stereoRectify(Ks_L, camera_param_left.D, Ks_R, camera_param_right.D, new_size,
                                               camera_param_right.R_c, camera_param_right.t_c,
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, R2, P1, P2, Q


def get_rectify_map_original(camera_param_left:CameraInfoProcessed, camera_param_right:CameraInfoProcessed,
                             original_size:List[int])-> Tuple[List[list], list, tuple]:
    '''
    Get rectify map, also get ROI for later crop so that we can remove the black border.
    camera_param_left: camera parameters for left camera
    camera_param_right: camera parameters for right camera
    original_size: original size of the image
    output: rectify map for both cameras, and valid pixels for both cameras
    '''

    ############## uncomment if on-campus ################
    # original_size = tuple(original_size)

    R1, R2, P1, P2, Q, roi1, roi2= cv2.stereoRectify(camera_param_left.K, camera_param_left.D, camera_param_right.K,
                                                camera_param_right.D, original_size, camera_param_right.R_c,
                                                camera_param_right.t_c, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(camera_param_left.K, camera_param_left.D, R1, P1,
                                                       original_size, cv2.CV_32FC1)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(camera_param_right.K, camera_param_right.D, R2, P2,
                                                         original_size, cv2.CV_32FC1)
    return [[left_map1, left_map2], [right_map1, right_map2]], [roi1, roi2], (R1, R2, P1, P2, Q)

cs = ConfigStore.instance()
cs.store(name="resize_rectify", node=AppCfg)
# set config path
p_config = Path.cwd().parents[2] / 'config'

@hydra.main(
    version_base=None,
    config_path= str(p_config),
    config_name="config_rr_jack"
    # config_name="config_rr_jack_ubc"
    # config_name="config_rr_jack_campus"
)
def main(cfg: AppCfg):
    camera_calibration_path = Path(cfg.camera_calibration_path)
    camera_names = cfg.camera_names
    camera_offset = cfg.camera_offset
    if camera_offset is not None:
        camera_offset = np.array(cfg.camera_offset).reshape(3, 3)
    intermediate_dir = Path(cfg.path_config.intermediate_dir)
    raw_dir = Path(cfg.path_config.raw_dir)
    processed_dir = Path(cfg.path_config.processed_dir)

    enable_resize = cfg.preprocess.resize_config.enable_resize
    enable_rectify = cfg.preprocess.enable_rectify
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)
    original_size = cfg.preprocess.resize_config.original_size
    if enable_resize:
        new_size = cfg.preprocess.resize_config.new_size
    else:
        new_size = original_size

    if cfg.preprocess.folder_initialize:
        if processed_dir.exists():
            clear_folder(output_folder)
        else:
            print(f"Output folder does not exist - {processed_dir}")

    camera_param_left, camera_param_right = get_stereo_camera_calibration_original(camera_calibration_path, camera_names)

    img_map_list, roi_list, _ = get_rectify_map_original(camera_param_left, camera_param_right,
                                                                                  original_size)

    _, _, P1_t, P2_t, Q_target = get_rectify_map_new_size(camera_param_left, camera_param_right, original_size, new_size)

    # new_camera_calibration_path = output_folder.parent / camera_calibration_path.name
    new_camera_calibration_path = output_folder / camera_calibration_path.name

    if not new_camera_calibration_path.exists():
        create_folder(new_camera_calibration_path)

    rectify_dict = {
        "P1": np.asarray(P1_t, dtype=np.float64).tolist(),
        "P2": np.asarray(P2_t, dtype=np.float64).tolist(),
        "Q": np.asarray(Q_target, dtype=np.float64).tolist(),
        "img_width": int(new_size[0]),
        "img_height": int(new_size[1]),
        "convention": {
            "P_shape": "3x4",
            "Q_shape": "4x4",
            "notes": "P1/P2 are rectified projection matrices; Q from stereoRectify; units of Z follow baseline units."
        }
    }
    rectify_file_path = new_camera_calibration_path / 'rectify_params.json'

    with open(rectify_file_path, 'w') as f:
        json.dump(rectify_dict, f, indent=2)

    left_img_folder = input_folder / 'image' / 'left'

    file_names = get_sorted_names(left_img_folder)

    ## resize the images
    for i_cam in range(len(camera_names)):
        print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
        img_save_folder = output_folder / 'image' / camera_names[i_cam]
        if not img_save_folder.exists():
            create_folder(img_save_folder)
        raw_img_folder = input_folder / 'image' / camera_names[i_cam]
        map1, map2 = img_map_list[i_cam]
        x, y, w, h = roi_list[i_cam]
        for file_name in tqdm(file_names, desc=f"Resize and Rectify Frames of {camera_names[i_cam]}"):
            img_raw = cv2.imread(str(raw_img_folder / file_name))

            if enable_rectify:
                img_rectify = cv2.remap(img_raw, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                img_rectify = img_rectify[y:y + h, x:x + w]
                pad_bottom = original_size[1] - img_rectify.shape[0]
                if pad_bottom:
                    img_rectify = cv2.copyMakeBorder(
                        img_rectify, 0, pad_bottom, 0, 0,
                        borderType=cv2.BORDER_REPLICATE
                    )
            else:
                img_rectify = img_raw.copy()

            if enable_resize:
                img_out = cv2.resize(img_rectify, (new_size[0], new_size[1]), interpolation=cv2.INTER_AREA)
            else:
                img_out = img_rectify.copy()
            cv2.imwrite(str(img_save_folder / file_name), img_out)

    # copy the kinematic folder
    old_kinematic_folder = input_folder / 'kinematic'
    new_kinematic_folder = output_folder / 'kinematic'
    if not new_kinematic_folder.exists():
        create_folder(new_kinematic_folder)
    copy_folder(old_kinematic_folder, new_kinematic_folder)
    # copy the synchronized time folder
    old_timestamp_folder = input_folder / 'time_syn'
    new_timestamp_folder = output_folder / 'time_syn'
    if not new_timestamp_folder.exists():
        create_folder(new_timestamp_folder)
    copy_folder(old_timestamp_folder, new_timestamp_folder)
    # copy and scale the camera parameters
    # copy_stereo_camera_calibration(camera_calibration_path, camera_names, output_folder.parent, original_size,
    #                                new_size)
    copy_stereo_camera_calibration(camera_calibration_path, camera_names, output_folder, original_size, new_size)

if __name__ == '__main__':
    main()
    print('Rectify and then Resize Done!')