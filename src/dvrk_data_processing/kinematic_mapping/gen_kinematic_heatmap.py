from dataclasses import dataclass
from typing import Union, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from dvrk_data_processing.utils.hydra_config import PathConfig, KinematicMapConfig
import yaml
from dvrk_data_processing.utils.utility import load_stereo_proj_mtx, create_folder, clear_folder, load_json_cp, glob_sorted_frame
import copy
from tqdm import tqdm
import json
import cv2

@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: KinematicMapConfig
    workspace: str


def cam_project_3d_to_2d(xyz:np.ndarray, P_cam:np.ndarray)->Tuple[Union[None,float], Union[None,float]]:
    '''
    Project 3D point to 2D pixel coordinates.
    xyz: 3D point in the camera coordinate system
    P_cam: 3x3 camera projection matrix
    output: 2D pixel coordinates (u,v)
    '''
    xyz[2] = -xyz[2] # reverse z-axis, incorrect!!! only for test!!!
    px, py, pz = P_cam @ xyz
    if pz <= 0:
        print('z-coordinate of the camera projection point is non-positive!')
        return None, None
    u = px / pz
    v = py / pz
    return u, v


def d_weight(xyz:np.ndarray, adv_w:bool, tol:float=0.05)->float:
    '''
    Calculate the weight of the prediction term.
    xyz: 3D point in the camera coordinate system
    adv_w: whether to use the advanced d-weight
    tol: tolerance for the distance (the substraction offset)
    output: calculated weight
    '''
    xyz_norm = np.linalg.norm(xyz)
    s = xyz_norm - tol
    if adv_w:
        d = np.exp((-s + np.exp(-s))/2.0) / np.sqrt(2.0 * np.pi)
        return d
    else:
        if s < 1e-4:
            s = 1e-4
        d = 1.0 / (1000.0 * s)
        return d


def gen_heatmap(u:float,v:float, u_next:float, v_next:float, xyz:np.ndarray, sigma_x:float, sigma_y:float, img_w:float, img_h:float, adv_w:bool)->np.ndarray:
    '''
    Generate the kinematic heatmap.
    u: x-coordinate of the current point
    v: y-coordinate of the current point
    u_next: x-coordinate of the predicted point
    v_next: y-coordinate of the predicted point
    xyz: 3D point in the camera coordinate system
    sigma_x: standard deviation of the Gaussian kernel along the x-axis
    sigma_y: standard deviation of the Gaussian kernel along the y-axis
    img_w: width of the image
    img_h: height of the image
    adv_w: whether to use the advanced weight
    output: kinematic heatmap
    '''
    y, x = np.mgrid[0:img_h, 0:img_w]
    mp_current = np.exp(-(((x - u) ** 2 / sigma_x ** 2) + ((y - v) ** 2 / sigma_y ** 2)))
    mp_predict = np.exp(-(((x - u_next) ** 2 / sigma_x ** 2) + ((y - v_next) ** 2 / sigma_y ** 2)))
    d = d_weight(xyz, adv_w)
    return mp_current + d * mp_predict



cs = ConfigStore.instance()
cs.store(name="kinematic_map", node=AppCfg)
# set config path
p_config = Path.cwd().parents[2] / 'config'

@hydra.main(
    version_base=None,
    config_path= str(p_config),
    config_name="config"
)
def main(cfg: AppCfg):
    camera_calibration_path = Path(cfg.preprocess.camera_calibration_path)
    arm_list = list(cfg.preprocess.arm_name)
    img_w, img_h = cfg.preprocess.img_size
    fps = float(cfg.preprocess.fps)
    sigma_x = float(cfg.preprocess.sigma_x)
    sigma_y = float(cfg.preprocess.sigma_y)
    processed_dir = Path(cfg.path_config.processed_dir)
    raw_dir = Path(cfg.path_config.raw_dir)
    adv_w = cfg.preprocess.weight_adv
    # adv_w = True
    if cfg.preprocess.folder_initialize:
        clear_folder(processed_dir)

    dt = 1.0 / fps

    P_cameras = load_stereo_proj_mtx(camera_calibration_path)
    if np.array_equal(P_cameras[0], P_cameras[1]):
        P_cameras.pop()

    data_folder = raw_dir / cfg.preprocess.input_subfolder
    if adv_w:
        save_folder = processed_dir / f'{cfg.preprocess.output_subfolder}_wadv'
    else:
        save_folder = processed_dir / cfg.preprocess.output_subfolder

    for i_pcam in range(len(P_cameras)):
        P_cam = P_cameras[i_pcam]
        if len(P_cam) == 2:
            camera_names = ['left', 'right']
        else:
            camera_names = None
        data_path = data_folder / 'api_cp_files'
        cp_file_list = glob_sorted_frame(data_path)
        # cp_file_list = [cp_file_list[0]]
        count = 0
        for file_name in tqdm(cp_file_list, desc="Kinematic HeatMap Processing Frames"):
            data_kinematic = load_json_cp(file_name, arm_list)
            img_file_name = file_name.parts[-1].replace('frame', '').replace('json', 'png')
            heatmap_file_name = file_name.parts[-1].replace('frame', '').replace('json', 'npy')
            for i_arm in range(len(arm_list)):
                arm_name = arm_list[i_arm]
                if camera_names is not None:
                    arm_save_folder = save_folder / arm_name / camera_names[i_pcam]
                else:
                    arm_save_folder = save_folder / arm_name

                img_save_folder = arm_save_folder / 'image'
                heatmap_save_folder = arm_save_folder / 'heatmap'
                if not img_save_folder.exists():
                    create_folder(img_save_folder)
                if not heatmap_save_folder.exists():
                    create_folder(heatmap_save_folder)
                data_arm = data_kinematic[arm_name]
                # R = data_arm['R']
                t = data_arm['t']
                w = data_arm['w']
                v = data_arm['v']

                ### predict next pos using first-order approximation
                dx = v + np.cross(w, t)
                t_next = t + dx * dt

                u, v = cam_project_3d_to_2d(t, P_cam)
                u_next, v_next = cam_project_3d_to_2d(t_next, P_cam)

                uv_none = any(obj is None for obj in [u, v, u_next, v_next])

                if uv_none:
                    print(f'For frame {count}, {arm_name} has None in the pixel coordinates!')
                    continue

                kp_heat = gen_heatmap(u, v, u_next, v_next, t, sigma_x, sigma_y, img_w, img_h, adv_w)

                heatmap_file_path = heatmap_save_folder / heatmap_file_name
                np.save(heatmap_file_path, kp_heat)

                img_file_path = img_save_folder / img_file_name
                kp_norm = cv2.normalize(kp_heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(str(img_file_path), kp_norm)
            count += 1

if __name__ == '__main__':
    # from hydra import compose, initialize
    #
    # with initialize(version_base=None, config_path='../../../config'):
    #     cfg = compose(config_name="config")
    main()
    print('Done!')








