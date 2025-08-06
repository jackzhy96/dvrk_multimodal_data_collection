from dataclasses import dataclass
from typing import Union, Tuple, List
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
from dvrk_data_processing.utils.hydra_config import PathConfig, KinematicMapConfig
from dvrk_data_processing.utils.utility import (create_folder, clear_folder, load_json_cp, glob_sorted_frame,
                                                load_stereo_camera_param_yaml, skew, load_mono_camera_param_yaml,load_handeye_dict,load_ecm_mat)
from dvrk_data_processing.utils.data_load_config import CameraInfoProcessed
from tqdm import tqdm
import cv2

@dataclass
class AppCfg:
    path_config: PathConfig
    preprocess: KinematicMapConfig
    workspace: str
    camera_names: List[str]
    camera_calibration_path: Union[Path, str]
    camera_offset: Union[None, List[float]]
    handeye_calib_path: Union[Path, str]


def pixel_coord_check(pixel_coord:np.ndarray, img_w:int, img_h:int)->None:
    '''
    check if the pixel coordinates are within the image scope.
    pixel_coord: 2D pixel coordinates (u,v), dimension is Nx2
    img_w: width of the image
    img_h: height of the image
    output: print warning if any pixel coordinate is out of range (view scope)
    '''
    for i in range(len(pixel_coord)):
        u = pixel_coord[i][0]
        v = pixel_coord[i][1]
        if not (0 <= u < img_w and 0 <= v < img_h):
            print(f'pair {i} pixel coordinate ({u}, {v}) is out of range (view scope)!')

def cam_project_3d_to_2d(coord_3d:np.ndarray, cam_param: CameraInfoProcessed,
                         cam_offset: Union[None, np.ndarray])->np.ndarray:
    '''
    Project 3D point to 2D pixel coordinates,enable camera offset via Rotation matrix
    coord_3d: 3D point in the camera coordinate system, N points
    cam_param: camera parameters, using CameraInfo dataclass
    cam_offset: camera offset, 3x3 rotation matrix
    output: 2D pixel coordinates (u,v), dimension is Nx2
    '''
    if cam_offset is None:
        cam_offset = np.eye(3)
    R_cam = cam_param.R_c
    t_cam = cam_param.t_c.reshape(-1,1)
    R_cam = cam_offset @ R_cam
    rvec, _ = cv2.Rodrigues(R_cam)
    tvec = cam_offset @ t_cam
    pixel_coord, _ = cv2.projectPoints(coord_3d, rvec, tvec, cam_param.K, cam_param.D)
    pixel_coord_check(pixel_coord.reshape(-1, 2), cam_param.image_width, cam_param.image_height)
    return pixel_coord.reshape(-1,2)

def d_weight(xyz:np.ndarray, weight_adv:bool, tol_dist:float=0.05)->float:
    '''
    Calculate the weight of the prediction term.
    xyz: 3D point in the camera coordinate system
    weight_adv: whether to use the advanced d-weight
    tol_dist: tolerance for the distance (the substraction offset)
    output: calculated weight
    '''
    xyz_norm = np.linalg.norm(xyz)
    s = xyz_norm - tol_dist
    if weight_adv:
        # d = np.exp((-s + np.exp(-s))/2.0) / np.sqrt(2.0 * np.pi)
        d = np.exp((-s + np.exp(-s)) / 2.0)
        return d
    else:
        if s < 4e-4:
            s = 4e-4
        d = 1.0 / (1000.0 * s)
        return d


def gen_heatmap(u:float,v:float, u_next:float, v_next:float, xyz:np.ndarray, sigma_x:float, sigma_y:float,
                img_w:float, img_h:float, weight_adv:bool, tol_dist:float)->np.ndarray:
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
    weight_adv: whether to use the advanced weight
    tol_dist: tolerance for the distance (the substraction offset)
    output: kinematic heatmap
    '''
    y, x = np.mgrid[0:img_h, 0:img_w]
    mp_current = np.exp(-(((x - u) ** 2 / sigma_x ** 2) + ((y - v) ** 2 / sigma_y ** 2)))
    mp_predict = np.exp(-(((x - u_next) ** 2 / sigma_x ** 2) + ((y - v_next) ** 2 / sigma_y ** 2)))
    d = d_weight(xyz, weight_adv, tol_dist)
    return mp_current + d * mp_predict


cs = ConfigStore.instance()
cs.store(name="kinematic_map", node=AppCfg)
# set config path
p_config = Path.cwd().parents[2] / 'config'

@hydra.main(
    version_base=None,
    config_path= str(p_config),
    config_name="config_kp_jack"
)
def main(cfg: AppCfg):
    camera_calibration_path = Path(cfg.camera_calibration_path)
    camera_names = cfg.camera_names
    camera_offset = cfg.camera_offset

    if camera_offset is not None:
        camera_offset = np.array(cfg.camera_offset).reshape(3, 3)
    arm_list = list(cfg.preprocess.arm_name)
    
    handeye_path = Path(cfg.handeye_calib_path)
    he_dict_raw = load_handeye_dict(handeye_path, arm_list)   # {'PSM1':4×4 , 'PSM2':4×4}

    he_dict = he_dict_raw
    img_w, img_h = cfg.preprocess.img_size
    fps_img = float(cfg.preprocess.fps_img)
    fps_kin = float(cfg.preprocess.fps_kin)
    sigma_x = float(cfg.preprocess.weight_config.sigma_x)
    sigma_y = float(cfg.preprocess.weight_config.sigma_y)
    tol_dist = float(cfg.preprocess.weight_config.tol_dist)
    processed_dir = Path(cfg.path_config.processed_dir)
    raw_dir = Path(cfg.path_config.raw_dir)
    weight_adv = cfg.preprocess.weight_config.advanced_weight
    # weight_adv = False ## use this one if you only want to have the heatmap
    enable_overlay = cfg.preprocess.enable_overlay
    # enable_overlay = False ## use this one if you only want to have the heatmap
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)
    if cfg.preprocess.folder_initialize:
        clear_folder(processed_dir)

    dt = 1.0 / fps_kin

    data_folder = input_folder / 'kinematic'
    save_folder = output_folder
    for i_cam in range(len(camera_names)):
        camera_file_path = camera_calibration_path / f'{camera_names[i_cam]}.yaml'
        if len(camera_names) == 2:
            camera_params = load_stereo_camera_param_yaml(camera_file_path)
        elif len(camera_names) == 1:
            camera_params = load_mono_camera_param_yaml(camera_file_path)
        else:
            raise ValueError('Only support single or stereo camera setup.')
        image_folder = data_folder.parent / 'image' / camera_names[i_cam]
        print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
        if enable_overlay:
            img_file_list = glob_sorted_frame(image_folder)

        from copy import deepcopy
        camera_params_naked = deepcopy(camera_params)
        camera_params_naked.R_c = np.eye(3)
        camera_params_naked.t_c = np.zeros((3, 1))
        ecm_folder = data_folder.parent / 'kinematic' / 'ECM'
        ecm_files  = glob_sorted_frame(ecm_folder)
        for i_arm in range(len(arm_list)):
            arm_name = arm_list[i_arm]
            data_path = data_folder / arm_name
            file_list = glob_sorted_frame(data_path)

            if camera_names is not None:
                arm_save_folder = save_folder / arm_name / camera_names[i_cam]

                print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
            else:
                arm_save_folder = save_folder / arm_name


            for file_name in tqdm(file_list, desc=f"Kinematic HeatMap Processing Frames of {arm_name}"):
                # === Camera←World  ===
                T_W_E = load_ecm_mat(ecm_files[int(file_name.stem)])   # World ← ECM-Tip
                T_C_W = np.linalg.inv(T_W_E)                           #  Camera≈ECM-Tip → T_C_E=I
                R_C_W = T_C_W[:3, :3]
                p_C_W = T_C_W[:3,  3]

                data_arm = load_json_cp(file_name, arm_name)
                if enable_overlay:
                    img_file_path = img_file_list[int(file_name.stem)]
                    img_greyscale = cv2.imread(str(img_file_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
                img_file_name = file_name.parts[-1].replace('json', 'png')
                heatmap_file_name = file_name.parts[-1].replace('json', 'npy')
                img_save_folder = arm_save_folder / 'image'
                heatmap_save_folder = arm_save_folder / 'heatmap'
                if not img_save_folder.exists():
                    create_folder(img_save_folder)
                if not heatmap_save_folder.exists():
                    create_folder(heatmap_save_folder)

                # R = data_arm.R
                t = data_arm.t
                w = data_arm.w
                v = data_arm.v

                ### predict next pos using first-order approximation
                dx = v + np.cross(w, t)
                t_next = t + dx * dt
                t_local      = data_arm.t_local
                t_local_next = t_local + dx * dt

               # ---------- Hand-Eye ----------
                T_W_B = he_dict[arm_name]              # World ← PSM-RCM
                R_W_B = T_W_B[:3, :3];  p_W_B = T_W_B[:3, 3]

                tip_world      = R_W_B @ t_local      + p_W_B
                tip_world_next = R_W_B @ t_local_next + p_W_B

                tip_cam_he      = R_C_W @ tip_world      + p_C_W
                tip_cam_he_next = R_C_W @ tip_world_next + p_C_W

                pixel_coord_he      = cam_project_3d_to_2d(tip_cam_he.reshape(1,3),
                                                        camera_params_naked, camera_offset)
                pixel_coord_next_he = cam_project_3d_to_2d(tip_cam_he_next.reshape(1,3),
                                                        camera_params_naked, camera_offset)

                kp_heat = gen_heatmap(
                    pixel_coord_he[0,0], pixel_coord_he[0,1],
                    pixel_coord_next_he[0,0], pixel_coord_next_he[0,1],
                    tip_cam_he, sigma_x, sigma_y, img_w, img_h, weight_adv, tol_dist
                )


                if enable_overlay:
                    kp_heat = np.multiply(kp_heat, img_greyscale)

                heatmap_file_path = heatmap_save_folder / heatmap_file_name
                np.save(heatmap_file_path, kp_heat)

                img_file_path = img_save_folder / img_file_name
                kp_norm = cv2.normalize(kp_heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(str(img_file_path), kp_norm)


                # -----------------------------------------------------------
                #  DEBUG: print/record 3-D and pixel data  (every 50 frames)
                # -----------------------------------------------------------

                pixel_coord = cam_project_3d_to_2d(t,  camera_params, camera_offset)
                pixel_coord_next = cam_project_3d_to_2d(t_next,  camera_params, camera_offset)

                u = pixel_coord[0, 0]
                v = pixel_coord[0, 1]
                u_next = pixel_coord_next[0, 0]
                v_next = pixel_coord_next[0, 1]

                if int(file_name.stem) % 50 == 0:          
                    frame_id = file_name.stem
                    print(
                        f"[Frame {frame_id} | {arm_name}]  dVRK 3-D (cam): "
                        f"{t if 't' in locals() else 'N/A'}"
                    )
                    print(
                        f"[Frame {frame_id} | {arm_name}]  dVRK 2-D (u,v): "
                        f"{pixel_coord[0,0]:7.1f}, {pixel_coord[0,1]:7.1f}"
                    )
                    print(
                        f"[Frame {frame_id} | {arm_name}]  HE   3-D (cam): "
                        f"{tip_cam_he.round(4)}"
                    )
                    print(
                        f"[Frame {frame_id} | {arm_name}]  HE   2-D (u,v): "
                        f"{pixel_coord_he[0,0]:7.1f}, {pixel_coord_he[0,1]:7.1f}"
                    )
                    # quick sanity: delta in pixels
                    du = pixel_coord_he[0,0] - pixel_coord[0,0]
                    dv = pixel_coord_he[0,1] - pixel_coord[0,1]
                    print(
                        f"[Frame {frame_id} | {arm_name}]  Δ(pixel HE-dVRK): "
                        f"{du:+6.1f}, {dv:+6.1f}\n"
                    )

if __name__ == '__main__':
    # main()
    # print('Kinematic Mapping Done!')
    from hydra import compose, initialize

    with initialize(version_base=None, config_path='../../../config'):
        cfg = compose(config_name="config_kp_jack")
    camera_calibration_path = Path(cfg.camera_calibration_path)
    camera_names = cfg.camera_names
    camera_offset = cfg.camera_offset

    if camera_offset is not None:
        camera_offset = np.array(cfg.camera_offset).reshape(3, 3)
    arm_list = list(cfg.preprocess.arm_name)

    handeye_path = Path(cfg.handeye_calib_path)
    he_dict_raw = load_handeye_dict(handeye_path, arm_list)  # {'PSM1':4×4 , 'PSM2':4×4}

    he_dict = he_dict_raw
    img_w, img_h = cfg.preprocess.img_size
    fps_img = float(cfg.preprocess.fps_img)
    fps_kin = float(cfg.preprocess.fps_kin)
    sigma_x = float(cfg.preprocess.weight_config.sigma_x)
    sigma_y = float(cfg.preprocess.weight_config.sigma_y)
    tol_dist = float(cfg.preprocess.weight_config.tol_dist)
    processed_dir = Path(cfg.path_config.processed_dir)
    raw_dir = Path(cfg.path_config.raw_dir)
    weight_adv = cfg.preprocess.weight_config.advanced_weight
    # weight_adv = False ## use this one if you only want to have the heatmap
    enable_overlay = cfg.preprocess.enable_overlay
    # enable_overlay = False ## use this one if you only want to have the heatmap
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)
    if cfg.preprocess.folder_initialize:
        clear_folder(processed_dir)

    dt = 1.0 / fps_kin

    data_folder = input_folder / 'kinematic'
    save_folder = output_folder
    for i_cam in range(len(camera_names)):
        camera_file_path = camera_calibration_path / f'{camera_names[i_cam]}.yaml'
        if len(camera_names) == 2:
            camera_params = load_stereo_camera_param_yaml(camera_file_path)
        elif len(camera_names) == 1:
            camera_params = load_mono_camera_param_yaml(camera_file_path)
        else:
            raise ValueError('Only support single or stereo camera setup.')
        image_folder = data_folder.parent / 'image' / camera_names[i_cam]
        print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
        if enable_overlay:
            img_file_list = glob_sorted_frame(image_folder)

        from copy import deepcopy

        camera_params_naked = deepcopy(camera_params)
        camera_params_naked.R_c = np.eye(3)
        camera_params_naked.t_c = np.zeros((3, 1))
        ecm_folder = data_folder.parent / 'kinematic' / 'ECM'
        ecm_files = glob_sorted_frame(ecm_folder)

        # # # uncomment for quick test
        # # arm_list = [arm_list[0]]

        for i_arm in range(len(arm_list)):
            arm_name = arm_list[i_arm]
            data_path = data_folder / arm_name
            file_list = glob_sorted_frame(data_path)

            # # uncomment for quick test
            #         # file_list = [file_list[0]]

            if camera_names is not None:
                arm_save_folder = save_folder / arm_name / camera_names[i_cam]

                print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
            else:
                arm_save_folder = save_folder / arm_name

            for file_name in tqdm(file_list, desc=f"Kinematic HeatMap Processing Frames of {arm_name}"):
                # === Camera←World  ===
                T_W_E = load_ecm_mat(ecm_files[int(file_name.stem)])  # World ← ECM-Tip
                T_C_W = np.linalg.inv(T_W_E)  # Camera≈ECM-Tip → T_C_E=I
                R_C_W = T_C_W[:3, :3]
                p_C_W = T_C_W[:3, 3]

                data_arm = load_json_cp(file_name, arm_name)
                if enable_overlay:
                    img_file_path = img_file_list[int(file_name.stem)]
                    img_greyscale = cv2.imread(str(img_file_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
                img_file_name = file_name.parts[-1].replace('json', 'png')
                heatmap_file_name = file_name.parts[-1].replace('json', 'npy')
                img_save_folder = arm_save_folder / 'image'
                heatmap_save_folder = arm_save_folder / 'heatmap'
                if not img_save_folder.exists():
                    create_folder(img_save_folder)
                if not heatmap_save_folder.exists():
                    create_folder(heatmap_save_folder)

                # R = data_arm.R
                t = data_arm.t
                w = data_arm.w
                v = data_arm.v

                ### predict next pos using first-order approximation
                dx = v + np.cross(w, t)
                t_next = t + dx * dt
                t_local = data_arm.t_local
                t_local_next = t_local + dx * dt

                # ---------- Hand-Eye ----------
                T_W_B = he_dict[arm_name]  # World ← PSM-RCM
                R_W_B = T_W_B[:3, :3]
                p_W_B = T_W_B[:3, 3]

                tip_world = R_W_B @ t_local + p_W_B
                tip_world_next = R_W_B @ t_local_next + p_W_B

                tip_cam_he = R_C_W @ tip_world + p_C_W
                tip_cam_he_next = R_C_W @ tip_world_next + p_C_W

                pixel_coord_he = cam_project_3d_to_2d(tip_cam_he.reshape(1, 3),
                                                      camera_params_naked, camera_offset)
                pixel_coord_next_he = cam_project_3d_to_2d(tip_cam_he_next.reshape(1, 3),
                                                           camera_params_naked, camera_offset)

                kp_heat = gen_heatmap(
                    pixel_coord_he[0, 0], pixel_coord_he[0, 1],
                    pixel_coord_next_he[0, 0], pixel_coord_next_he[0, 1],
                    tip_cam_he, sigma_x, sigma_y, img_w, img_h, weight_adv, tol_dist
                )

                if enable_overlay:
                    kp_heat = np.multiply(kp_heat, img_greyscale)

                heatmap_file_path = heatmap_save_folder / heatmap_file_name
                np.save(heatmap_file_path, kp_heat)

                img_file_path = img_save_folder / img_file_name
                kp_norm = cv2.normalize(kp_heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(str(img_file_path), kp_norm)

                # -----------------------------------------------------------
                #  DEBUG: print/record 3-D and pixel data  (every 50 frames)
                # -----------------------------------------------------------

                pixel_coord = cam_project_3d_to_2d(t, camera_params, camera_offset)
                pixel_coord_next = cam_project_3d_to_2d(t_next, camera_params, camera_offset)

                u = pixel_coord[0, 0]
                v = pixel_coord[0, 1]
                u_next = pixel_coord_next[0, 0]
                v_next = pixel_coord_next[0, 1]

                if int(file_name.stem) % 50 == 0:
                    frame_id = file_name.stem
                    print(
                        f"[Frame {frame_id} | {arm_name}]  dVRK 3-D (cam): "
                        f"{t if 't' in locals() else 'N/A'}"
                    )
                    print(
                        f"[Frame {frame_id} | {arm_name}]  dVRK 2-D (u,v): "
                        f"{pixel_coord[0, 0]:7.1f}, {pixel_coord[0, 1]:7.1f}"
                    )
                    print(
                        f"[Frame {frame_id} | {arm_name}]  HE   3-D (cam): "
                        f"{tip_cam_he.round(4)}"
                    )
                    print(
                        f"[Frame {frame_id} | {arm_name}]  HE   2-D (u,v): "
                        f"{pixel_coord_he[0, 0]:7.1f}, {pixel_coord_he[0, 1]:7.1f}"
                    )
                    # quick sanity: delta in pixels
                    du = pixel_coord_he[0, 0] - pixel_coord[0, 0]
                    dv = pixel_coord_he[0, 1] - pixel_coord[0, 1]
                    print(
                        f"[Frame {frame_id} | {arm_name}]  Δ(pixel HE-dVRK): "
                        f"{du:+6.1f}, {dv:+6.1f}\n"
                    )








