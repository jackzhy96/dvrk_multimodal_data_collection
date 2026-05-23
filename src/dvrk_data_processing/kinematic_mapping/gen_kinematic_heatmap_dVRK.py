from dataclasses import dataclass
from typing import Any, Union, List
import logging
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
from dvrk_data_processing.utils.hydra_config import PathConfig, KinematicMapConfig
from dvrk_data_processing.utils.utility import (create_folder, clear_folder, load_json_cp, glob_sorted_frame,
                                                load_stereo_camera_param_yaml, load_mono_camera_param_yaml,
                                                cam_project_3d_to_2d, gen_heatmap,
                                                parse_tool_tip_offsets, apply_tool_tip_offset,
                                                resolve_per_arm_weight_configs)
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
    tool_tip_offset: Any = None  # per-PSM 4x4 offset matrices (dict of lists or None)


cs = ConfigStore.instance()
# Renamed from "kinematic_map" to "kinematic_reproject" alongside the
# output-folder rename. The dVRK variant does NOT emit
# calibrated_kinematic (its transform chain isn't hand-eye-calibrated)
# but it shares the same preprocess config schema and output folder
# structure as the hand-eye variant.
cs.store(name="kinematic_reproject", node=AppCfg)
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

    # Parse per-PSM tool-tip offsets (4x4 matrices, identity if None)
    tool_tip_offsets = parse_tool_tip_offsets(cfg.tool_tip_offset, arm_list)

    # handeye_path = Path(cfg.handeye_calib_path)

    img_w, img_h = cfg.preprocess.img_size
    # Resolve per-arm weight configs (falls back to global weight_config when per-PSM is null)
    weight_configs = resolve_per_arm_weight_configs(cfg.preprocess, arm_list)
    processed_dir = Path(cfg.path_config.processed_dir)
    enable_overlay = cfg.preprocess.enable_overlay
    # enable_overlay = False ## use this one if you only want to have the heatmap
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)

    if cfg.preprocess.folder_initialize:
        if processed_dir.exists():
            clear_folder(output_folder)
        else:
            logging.warning(f"Output folder does not exist - {processed_dir}")

    # Pre-compute the pixel coordinate grid once (reused for every frame)
    mgrid_cache = np.mgrid[0:img_h, 0:img_w]

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
        logging.info(f'Working on {camera_names[i_cam].upper()} Camera')
        if enable_overlay:
            img_file_list = glob_sorted_frame(image_folder)

        for i_arm in range(len(arm_list)):
            arm_name = arm_list[i_arm]
            data_path = data_folder / arm_name
            file_list = glob_sorted_frame(data_path)

            # Get the tool-tip offset for this arm
            T_offset = tool_tip_offsets[arm_name]
            # Check if offset is non-identity to decide whether to apply it
            has_offset = not np.allclose(T_offset, np.eye(4))
            if has_offset:
                logging.info(f"Applying tool-tip offset for {arm_name}:\n{T_offset}")

            # Extract per-arm weight config for heatmap generation
            wcfg = weight_configs[arm_name]
            sigma_x = wcfg['sigma_x']
            sigma_y = wcfg['sigma_y']
            weight_adv = wcfg['advanced_weight']
            tol_dist = wcfg['tol_dist']

            if camera_names is not None:
                arm_save_folder = save_folder / arm_name / camera_names[i_cam]
            else:
                arm_save_folder = save_folder / arm_name

            # Create output directories once (not per frame)
            img_save_folder = arm_save_folder / 'image'
            heatmap_save_folder = arm_save_folder / 'heatmap'
            if not img_save_folder.exists():
                create_folder(img_save_folder)
            if not heatmap_save_folder.exists():
                create_folder(heatmap_save_folder)

            for file_name in tqdm(file_list, desc=f"Kinematic HeatMap Processing Frames of {arm_name}"):
                data_arm = load_json_cp(file_name, arm_name)
                if enable_overlay:
                    img_file_path = img_file_list[int(file_name.stem)]
                    img_greyscale = cv2.imread(str(img_file_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
                img_file_name = file_name.parts[-1].replace('json', 'png')
                heatmap_file_name = file_name.parts[-1].replace('json', 'npy')

                R_arm = data_arm.R
                t = data_arm.t
                w = data_arm.w
                v = data_arm.v

                # Apply tool-tip offset if configured (extends rotation and translation)
                if has_offset:
                    _, t = apply_tool_tip_offset(R_arm, t, T_offset)

                dt = 1.0 / data_arm.measured_frequency

                ### predict next pos using first-order approximation
                dx = v + np.cross(w, t)
                t_next = t + dx * dt

                pixel_coord = cam_project_3d_to_2d(t, camera_params, camera_offset)
                pixel_coord_next = cam_project_3d_to_2d(t_next, camera_params, camera_offset)

                u = pixel_coord[0, 0]
                v = pixel_coord[0, 1]
                u_next = pixel_coord_next[0, 0]
                v_next = pixel_coord_next[0, 1]

                kp_heat = gen_heatmap(u, v, u_next, v_next, t, sigma_x, sigma_y,
                                     img_w, img_h, weight_adv, tol_dist,
                                     mgrid_cache=mgrid_cache)

                if enable_overlay:
                    kp_heat = np.multiply(kp_heat, img_greyscale)

                heatmap_file_path = heatmap_save_folder / heatmap_file_name
                np.save(heatmap_file_path, kp_heat)

                img_file_path = img_save_folder / img_file_name
                kp_norm = cv2.normalize(kp_heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(str(img_file_path), kp_norm)

                # if int(file_name.stem) % 50 == 0:
                #     frame_id = file_name.stem
                #     print(
                #         f"[Frame {frame_id} | {arm_name}]  dVRK 3-D (cam): "
                #         f"{t if 't' in locals() else 'N/A'}"
                #     )
                #     print(
                #         f"[Frame {frame_id} | {arm_name}]  dVRK 2-D (u,v): "
                #         f"{pixel_coord[0,0]:7.1f}, {pixel_coord[0,1]:7.1f}"
                #     )

if __name__ == '__main__':
    main()
    print('Kinematic Mapping Done!')
    # from hydra import compose, initialize
    #
    # with initialize(version_base=None, config_path='../../../config'):
    #     cfg = compose(config_name="config_kp")
    # camera_calibration_path = Path(cfg.camera_calibration_path)
    # camera_names = cfg.camera_names
    # camera_offset = cfg.camera_offset
    # if camera_offset is not None:
    #     camera_offset = np.array(cfg.camera_offset).reshape(3, 3)
    # arm_list = list(cfg.preprocess.arm_name)
    # img_w, img_h = cfg.preprocess.img_size
    # fps_img = float(cfg.preprocess.fps_img)
    # fps_kin = float(cfg.preprocess.fps_kin)
    # sigma_x = float(cfg.preprocess.weight_config.sigma_x)
    # sigma_y = float(cfg.preprocess.weight_config.sigma_y)
    # tol_dist = float(cfg.preprocess.weight_config.tol_dist)
    # processed_dir = Path(cfg.path_config.processed_dir)
    # raw_dir = Path(cfg.path_config.raw_dir)
    #
    # weight_adv = cfg.preprocess.weight_config.advanced_weight
    # # weight_adv = False ## use this one if you only want to have the heatmap
    # enable_overlay = cfg.preprocess.enable_overlay
    # # enable_overlay = False ## use this one if you only want to have the heatmap
    # if cfg.preprocess.folder_initialize:
    #     clear_folder(processed_dir)
    #
    # dt = 1.0 / fps_kin
    #
    # data_folder = raw_dir / cfg.preprocess.input_subfolder / 'kinematic'
    #
    # ## only for testing
    # if enable_overlay:
    #     save_folder = processed_dir / f'{cfg.preprocess.output_subfolder}_overlay'
    # elif weight_adv:
    #     save_folder = processed_dir / f'{cfg.preprocess.output_subfolder}_advw'
    # else:
    #     save_folder = processed_dir / cfg.preprocess.output_subfolder
    #
    # # # uncomment for quick test
    # # arm_list = [arm_list[0]]
    #
    # for i_cam in range(len(camera_names)):
    #     camera_file_path = camera_calibration_path / f'{camera_names[i_cam]}.yaml'
    #     if len(camera_names) == 2:
    #         camera_params = load_stereo_camera_param_yaml(camera_file_path)
    #     elif len(camera_names) == 1:
    #         camera_params = load_mono_camera_param_yaml(camera_file_path)
    #     else:
    #         raise ValueError('Only support single or stereo camera setup.')
    #     image_folder = data_folder.parent / 'image' / camera_names[i_cam]
    #     print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
    #     if enable_overlay:
    #         img_file_list = glob_sorted_frame(image_folder)
    #     for i_arm in range(len(arm_list)):
    #         arm_name = arm_list[i_arm]
    #         data_path = data_folder / arm_name
    #         file_list = glob_sorted_frame(data_path)
    #         # # uncomment for quick test
    #         # file_list = [file_list[0]]
    #
    #         if camera_names is not None:
    #             arm_save_folder = save_folder / arm_name / camera_names[i_cam]
    #             print(f'Working on {camera_names[i_cam].upper()} Camera: \n')
    #         else:
    #             arm_save_folder = save_folder / arm_name
    #
    #         for file_name in tqdm(file_list, desc=f"Kinematic HeatMap Processing Frames of {arm_name}"):
    #             data_arm = load_json_cp(file_name, arm_name)
    #             if enable_overlay:
    #                 img_file_path = img_file_list[int(file_name.stem)]
    #                 img_greyscale = cv2.imread(str(img_file_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
    #             img_file_name = file_name.parts[-1].replace('json', 'png')
    #             heatmap_file_name = file_name.parts[-1].replace('json', 'npy')
    #             img_save_folder = arm_save_folder / 'image'
    #             heatmap_save_folder = arm_save_folder / 'heatmap'
    #             if not img_save_folder.exists():
    #                 create_folder(img_save_folder)
    #             if not heatmap_save_folder.exists():
    #                 create_folder(heatmap_save_folder)
    #
    #             # R = data_arm.R
    #             t = data_arm.t
    #             w = data_arm.w
    #             v = data_arm.v
    #
    #             ### predict next pos using first-order approximation
    #             dx = v + np.cross(w, t)
    #             t_next = t + dx * dt
    #
    #             # if arm_name == 'PSM1':
    #             #     # offset = np.diag([-1.0, -1.0, 1.0])
    #             #     offset = np.diag([-1.0, -1.0, 1.0])
    #             #     camera_offset = None
    #             # else:
    #             #     offset = np.eye(3)
    #             offset = np.eye(3)
    #
    #             pixel_coord = cam_project_3d_to_2d(offset@t, camera_params, camera_offset)
    #             pixel_coord_next = cam_project_3d_to_2d(offset@t_next, camera_params, camera_offset)
    #
    #             u = pixel_coord[0, 0]
    #             v = pixel_coord[0, 1]
    #             u_next = pixel_coord_next[0, 0]
    #             v_next = pixel_coord_next[0, 1]
    #
    #             kp_heat = gen_heatmap(u, v, u_next, v_next, t, sigma_x, sigma_y, img_w, img_h, weight_adv, tol_dist)
    #
    #             if enable_overlay:
    #                 kp_heat = np.multiply(kp_heat, img_greyscale)
    #
    #             heatmap_file_path = heatmap_save_folder / heatmap_file_name
    #             np.save(heatmap_file_path, kp_heat)
    #
    #             img_file_path = img_save_folder / img_file_name
    #             kp_norm = cv2.normalize(kp_heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #             cv2.imwrite(str(img_file_path), kp_norm)
