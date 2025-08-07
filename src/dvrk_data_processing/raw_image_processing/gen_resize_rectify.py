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

def get_stereo_camera_calibration(camera_calibration_path:Path, camera_names:List[str], original_size:List[int],
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
        new_cam_dict = asdict(new_cam_info)
        new_camera_calibration_path = output_folder / camera_calibration_path.name
        if not new_camera_calibration_path.exists():
            create_folder(new_camera_calibration_path)
        new_cam_file_path = new_camera_calibration_path / f'{camera_name}.yaml'
        with open(new_cam_file_path, 'w') as f:
            yaml.safe_dump(new_cam_dict, f)


def get_rectify_map(camera_param_left:CameraInfoProcessed, camera_param_right:CameraInfoProcessed,
                    new_size:List[int])-> Tuple[List[list], list]:
    '''
    Get rectify map, also get ROI for later crop so that we can remove the black border.
    camera_param_left: camera parameters for left camera
    camera_param_right: camera parameters for right camera
    new_size: new size of the image
    output: rectify map for both cameras, and valid pixels for both cameras
    '''
    R1, R2, P1, P2, Q, valid1, valid2= cv2.stereoRectify(camera_param_left.K, camera_param_left.D, camera_param_right.K,
                                                camera_param_right.D, new_size, camera_param_right.R_c,
                                                camera_param_right.t_c,alpha=0)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(camera_param_left.K, camera_param_left.D, R1, P1,
                                                       new_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(camera_param_right.K, camera_param_right.D, R2, P2,
                                                         new_size, cv2.CV_16SC2)
    return [[left_map1, left_map2], [right_map1, right_map2]], [valid1, valid2]


cs = ConfigStore.instance()
cs.store(name="resize_rectify", node=AppCfg)
# set config path
p_config = Path.cwd().parents[2] / 'config'

@hydra.main(
    version_base=None,
    config_path= str(p_config),
    config_name="config_rr_jack"
)
def main(cfg: AppCfg):
    camera_calibration_path = Path(cfg.camera_calibration_path)
    camera_names = cfg.camera_names
    camera_offset = cfg.camera_offset
    if camera_offset is not None:
        camera_offset = np.array(cfg.camera_offset).reshape(3, 3)
    intermediate_dir = Path(cfg.path_config.intermediate_dir)
    raw_dir = Path(cfg.path_config.raw_dir)

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

    camera_param_left, camera_param_right = get_stereo_camera_calibration(camera_calibration_path, camera_names,
                                                                          original_size, new_size)

    img_map_list, roi_list = get_rectify_map(camera_param_left, camera_param_right, new_size)

    left_img_folder = input_folder / 'image' / 'left'

    file_names = get_sorted_names(left_img_folder)

    ## resize the images
    for i_cam in range(len(camera_names)):
        print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
        img_save_folder = output_folder / 'image' / camera_names[i_cam]
        if not img_save_folder.exists():
            create_folder(img_save_folder)
        raw_img_folder = input_folder / 'image' / camera_names[i_cam]
        for file_name in tqdm(file_names, desc=f"Resize and Rectify Frames of {camera_names[i_cam]}"):
            img_raw = cv2.imread(str(raw_img_folder / file_name))
            if enable_resize:
                img_resize = cv2.resize(img_raw, (new_size[0], new_size[1]))
            else:
                img_resize = img_raw.copy()
            if enable_rectify:
                img_rectify = cv2.remap(img_resize, img_map_list[i_cam][0], img_map_list[i_cam][1], cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                x, y, w, h = roi_list[i_cam]
                img_rectify = img_rectify[y:y+h, x:x+w]
                pad_bottom = new_size[1] - img_rectify.shape[0]   # usually 0, sometimes it may have one row difference
                if pad_bottom:
                    img_rectify = cv2.copyMakeBorder(
                        img_rectify, 0, pad_bottom, 0, 0,
                        borderType=cv2.BORDER_REPLICATE
                    )
            else:
                img_rectify = img_resize.copy()
            cv2.imwrite(str(img_save_folder / file_name), img_rectify)
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
    copy_stereo_camera_calibration(camera_calibration_path, camera_names, output_folder.parent, original_size,
                                   new_size)

if __name__ == '__main__':
    main()
    print('Resize and Rectify Done!')
    # from hydra import compose, initialize
    #
    # with initialize(version_base=None, config_path='../../../config'):
    #     cfg = compose(config_name="config_rr")
    # camera_calibration_path = Path(cfg.camera_calibration_path)
    # camera_names = cfg.camera_names
    # camera_offset = cfg.camera_offset
    # if camera_offset is not None:
    #     camera_offset = np.array(cfg.camera_offset).reshape(3, 3)
    # intermediate_dir = Path(cfg.path_config.intermediate_dir)
    # raw_dir = Path(cfg.path_config.raw_dir)
    #
    # enable_resize = cfg.preprocess.resize_config.enable_resize
    # enable_rectify = cfg.preprocess.enable_rectify
    # input_folder = Path(cfg.preprocess.input_folder)
    # output_folder = Path(cfg.preprocess.output_folder)
    # original_size = cfg.preprocess.resize_config.original_size
    # if enable_resize:
    #     new_size = cfg.preprocess.resize_config.new_size
    # else:
    #     new_size = original_size
    #
    # if cfg.preprocess.folder_initialize:
    #     clear_folder(intermediate_dir)
    #
    # camera_param_left, camera_param_right = get_stereo_camera_calibration(camera_calibration_path, camera_names,
    #                                                                       original_size, new_size)
    #
    # img_map_list = get_rectify_map(camera_param_left, camera_param_right, new_size)
    #
    # left_img_folder = input_folder / 'image' / 'left'
    #
    # file_names = get_sorted_names(left_img_folder)
    #
    # ## resize the images
    # for i_cam in range(len(camera_names)):
    #     print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
    #     img_save_folder = output_folder / 'image' / camera_names[i_cam]
    #     if not img_save_folder.exists():
    #         create_folder(img_save_folder)
    #     raw_img_folder = input_folder / 'image' / camera_names[i_cam]
    #     for file_name in tqdm(file_names, desc=f"Resize and Rectify Frames of {camera_names[i_cam]}"):
    #         img_raw = cv2.imread(str(raw_img_folder / file_name))
    #         if enable_resize:
    #             img_resize = cv2.resize(img_raw, (new_size[0], new_size[1]))
    #         else:
    #             img_resize = img_raw.copy()
    #         if enable_rectify:
    #             img_rectify = cv2.remap(img_resize, img_map_list[i_cam][0], img_map_list[i_cam][1], cv2.INTER_LINEAR)
    #         else:
    #             img_rectify = img_resize.copy()
    #         cv2.imwrite(str(img_save_folder / file_name), img_rectify)
    # # copy the kinematic folder
    # old_kinematic_folder = input_folder / 'kinematic'
    # new_kinematic_folder = output_folder / 'kinematic'
    # if not new_kinematic_folder.exists():
    #     create_folder(new_kinematic_folder)
    # copy_folder(old_kinematic_folder, new_kinematic_folder)
    # # copy the synchronized time folder
    # old_timestamp_folder = input_folder / 'time_syn'
    # new_timestamp_folder = output_folder / 'time_syn'
    # if not new_timestamp_folder.exists():
    #     create_folder(new_timestamp_folder)
    # copy_folder(old_timestamp_folder, new_timestamp_folder)
    # # copy and scale the camera parameters
    # copy_stereo_camera_calibration(camera_calibration_path, camera_names, output_folder.parent, original_size,
    #                                new_size)