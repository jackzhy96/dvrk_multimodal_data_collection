"""
Stage-2 kinematic projection pipeline (hand-eye variant) — the default.

Reads the rectified stereo images plus per-frame kinematic JSON, projects the
PSM tool tip into the **left-rectified camera frame** via the hand-eye
calibration chain (ECM-derived T_C_W ∘ T_W_B ∘ PSM-base), and writes:

  - heatmaps     : ``processed_dir/kinematic_reproject/<PSM>/<cam>/heatmap/<frame>.npy``
                   + an 8-bit-normalized PNG visualization sibling.
  - calibrated_kinematic (gated by ``preprocess.calibrated_kinematic.enable``):
                   per-frame 6-DoF pose JSON at
                   ``processed_dir/kinematic_reproject/<PSM>/calibrated_kinematic/<frame>.json``.
                   PSM1 / PSM2 only — never ECM. ``setpoint_cp_calibrated`` is
                   omitted when the raw ``setpoint_cp`` is missing (offline recorder
                   does not log Cartesian setpoints). Note the asymmetry: the
                   measured pose goes through the full hand-eye chain
                   (PSM-base → world → camera), but the setpoint only goes
                   through ``T_C_W`` (world → camera). This is intentional
                   — see ``specs/interm_data_spec.md`` § calibrated_kinematic.
  - drawframe (gated by ``preprocess.drawframe.enable``):
                   tool-tip 3D axes drawn on each rectified frame, saved to
                   ``processed_dir/kinematic_reproject_drawframe/<PSM>/<cam>/<frame>.png``.

Pure refactor of the previous heatmap-only entry point — the heatmap output
is bit-for-bit unchanged when both new features are disabled, just renamed
``kinematic_map`` → ``kinematic_reproject``.
"""
from dataclasses import dataclass
from typing import Any, Union, List
from copy import deepcopy
import logging
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
from dvrk_data_processing.utils.hydra_config import PathConfig, KinematicMapConfig
from dvrk_data_processing.utils.utility import (create_folder, clear_folder, load_json_cp, glob_sorted_frame,
                                                load_stereo_camera_param_yaml, load_mono_camera_param_yaml,
                                                load_handeye_dict, load_ecm_transformation_matrix,
                                                cam_project_3d_to_2d, gen_heatmap,
                                                parse_tool_tip_offsets, apply_tool_tip_offset,
                                                resolve_per_arm_weight_configs)
from dvrk_data_processing.kinematic_mapping.kinematic_handeye import (
    compute_calibrated_tip_pose, compute_calibrated_setpoint_pose,
    write_calibrated_kinematic_json, read_raw_setpoint_cp,
    parse_drawframe_config, render_drawframe_axes,
)
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
# ConfigStore name updated from "kinematic_map" → "kinematic_reproject" alongside
# the folder rename so the dataclass registration matches the YAML.
cs.store(name="kinematic_reproject", node=AppCfg)
# set config path
p_config = Path.cwd().parents[2] / 'config'

@hydra.main(
    version_base=None,
    config_path= str(p_config),
    config_name="config_kp_jack"
    # config_name="config_kp_jack_campus"
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

    handeye_path = Path(cfg.handeye_calib_path)
    dict_handeye_calib = load_handeye_dict(handeye_path, arm_list)   # {'PSM1':4×4 , 'PSM2':4×4}

    img_w, img_h = cfg.preprocess.img_size
    # Resolve per-arm weight configs (falls back to global weight_config when per-PSM is null)
    weight_configs = resolve_per_arm_weight_configs(cfg.preprocess, arm_list)
    processed_dir = Path(cfg.path_config.processed_dir)
    enable_overlay = cfg.preprocess.enable_overlay
    input_folder = Path(cfg.preprocess.input_folder)
    output_folder = Path(cfg.preprocess.output_folder)

    # calibrated_kinematic flag. Default off (we don't want to spam
    # JSONs into datasets that haven't opted in yet).
    calibrated_cfg = getattr(cfg.preprocess, 'calibrated_kinematic', None)
    emit_calibrated_kinematic = bool(getattr(calibrated_cfg, 'enable', False)) if calibrated_cfg else False

    # drawframe block. Coerced to a frozen Python dataclass so the
    # per-frame loop doesn't pay the OmegaConf resolution overhead per call.
    drawframe_raw = getattr(cfg.preprocess, 'drawframe', None)
    drawframe_cfg = parse_drawframe_config(drawframe_raw)
    drawframe_root = processed_dir / 'kinematic_reproject_drawframe'   # sibling of kinematic_reproject/
    behind_camera_count = 0   # surfaces in the structured log at end of run

    if cfg.preprocess.folder_initialize:
        if processed_dir.exists():
            clear_folder(output_folder)
        else:
            logging.warning(f"Output folder does not exist - {processed_dir}")

    # Pre-compute the pixel coordinate grid once (reused for every frame)
    mgrid_cache = np.mgrid[0:img_h, 0:img_w]

    # Drawframe shares the per-camera projection chain with the heatmap path
    # (cam_project_3d_to_2d with `camera_params_naked` + `camera_offset`), so
    # no separate K/baseline preload is needed — the per-camera loop below
    # already builds `camera_params_naked` for each camera.

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
        # When drawframe is enabled we always need access to the rectified
        # frame for this camera, regardless of enable_overlay.
        if drawframe_cfg.enable and camera_names[i_cam] in drawframe_cfg.cameras:
            drawframe_img_list = glob_sorted_frame(image_folder)
        else:
            drawframe_img_list = None

        # Create a "naked" camera params copy with identity extrinsics for hand-eye projection
        # (the hand-eye pipeline already accounts for camera-to-world transforms)
        camera_params_naked = deepcopy(camera_params)
        camera_params_naked.R_c = np.eye(3)
        camera_params_naked.t_c = np.zeros((3, 1))
        ecm_folder = data_folder.parent / 'kinematic' / 'ECM'
        ecm_files  = glob_sorted_frame(ecm_folder)
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

            # calibrated_kinematic — only PSM1/PSM2 (never ECM), only when
            # emit_calibrated_kinematic is on, and only on the **first camera
            # pass** (the output is per-PSM, not per-camera, so we'd otherwise
            # write the same files twice).
            calibrated_folder = None
            write_calibrated_now = (
                emit_calibrated_kinematic
                and ('PSM' in arm_name.upper() and arm_name.upper() != 'ECM')
                and i_cam == 0
            )
            if write_calibrated_now:
                # output_folder already includes the stage suffix (path interpolation
                # `${path_config.processed_dir}/${.stage}`), so this resolves to
                # processed_dir/kinematic_reproject/<PSM>/calibrated_kinematic/.
                calibrated_folder = output_folder / arm_name / 'calibrated_kinematic'
                if not calibrated_folder.exists():
                    create_folder(calibrated_folder)

            # drawframe output dirs — per-PSM × per-cam, only if drawframe.enable
            # is set AND this camera is in the configured camera list.
            draw_this_cam = (
                drawframe_cfg.enable
                and camera_names[i_cam] in drawframe_cfg.cameras
                and 'PSM' in arm_name.upper() and arm_name.upper() != 'ECM'
            )
            if draw_this_cam:
                drawframe_out_dir = drawframe_root / arm_name / camera_names[i_cam]
                if not drawframe_out_dir.exists():
                    create_folder(drawframe_out_dir)

            for file_name in tqdm(file_list, desc=f"Kinematic HeatMap Processing Frames of {arm_name}"):
                # === Camera←World  ===
                T_W_E = load_ecm_transformation_matrix(ecm_files[int(file_name.stem)])   # World ← ECM-Tip
                T_C_W = np.linalg.inv(T_W_E)                           #  Camera≈ECM-Tip → T_C_E=I
                R_C_W = T_C_W[:3, :3]
                p_C_W = T_C_W[:3,  3]

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

                dt = 1.0 / data_arm.measured_frequency

                ### predict next pos using first-order approximation
                dx = v + np.cross(w, t)
                t_next = t + dx * dt
                t_local      = data_arm.t_local
                R_local      = data_arm.R_local

                # Apply tool-tip offset to local coordinates if configured.
                # We hold onto the **offset-applied** R_local and t_local
                # for the calibrated_kinematic / drawframe consumers below,
                # so they don't repeat this work.
                if has_offset:
                    R_local, t_local = apply_tool_tip_offset(R_local, t_local, T_offset)

                t_local_next = t_local + dx * dt

               # ---------- Hand-Eye ----------
                T_W_B = dict_handeye_calib[arm_name]              # World ← PSM-RCM
                R_W_B = T_W_B[:3, :3]
                p_W_B = T_W_B[:3, 3]

                tip_world      = R_W_B @ t_local      + p_W_B
                tip_world_next = R_W_B @ t_local_next + p_W_B

                tip_cam_handeye      = R_C_W @ tip_world      + p_C_W
                tip_cam_handeye_next = R_C_W @ tip_world_next + p_C_W

                pixel_coord_handeye      = cam_project_3d_to_2d(tip_cam_handeye.reshape(3,1),
                                                        camera_params_naked, camera_offset)
                pixel_coord_handeye_next = cam_project_3d_to_2d(tip_cam_handeye_next.reshape(3,1),
                                                        camera_params_naked, camera_offset)

                kp_heat = gen_heatmap(
                    pixel_coord_handeye[0,0], pixel_coord_handeye[0,1],
                    pixel_coord_handeye_next[0,0], pixel_coord_handeye_next[0,1],
                    tip_cam_handeye, sigma_x, sigma_y, img_w, img_h, weight_adv, tol_dist,
                    mgrid_cache=mgrid_cache
                )


                if enable_overlay:
                    kp_heat = np.multiply(kp_heat, img_greyscale)

                heatmap_file_path = heatmap_save_folder / heatmap_file_name
                np.save(heatmap_file_path, kp_heat)

                img_file_path = img_save_folder / img_file_name
                kp_norm = cv2.normalize(kp_heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(str(img_file_path), kp_norm)

                # ----------------------------------------------------------- #
                # calibrated_kinematic per-frame JSON
                # ----------------------------------------------------------- #
                # The pose computation reuses the rotations we already
                # composed for the heatmap path — we just also keep the
                # rotation (the original code threw it away).
                if write_calibrated_now:
                    pose_measured = compute_calibrated_tip_pose(
                        R_local=R_local, t_local=t_local,
                        T_W_B=T_W_B, T_C_W=T_C_W,
                    )

                    # setpoint_cp_calibrated — only when the raw setpoint_cp
                    # block exists in the JSON. Online recorder logs it;
                    # offline recorder does not (it has setpoint_js only).
                    raw_setpoint = read_raw_setpoint_cp(file_name)
                    if raw_setpoint is not None:
                        R_W_set, t_W_set = raw_setpoint
                        pose_setpoint = compute_calibrated_setpoint_pose(
                            R_W_set=R_W_set, t_W_set=t_W_set, T_C_W=T_C_W,
                        )
                    else:
                        pose_setpoint = None

                    out_json = calibrated_folder / f"{int(file_name.stem)}.json"
                    write_calibrated_kinematic_json(
                        out_path=out_json,
                        frame_id=int(file_name.stem),
                        arm_name=arm_name,
                        measured_cp_calibrated=pose_measured,
                        setpoint_cp_calibrated=pose_setpoint,
                    )

                # ----------------------------------------------------------- #
                # drawframe — render tool-tip axes on rectified image.
                # ----------------------------------------------------------- #
                # The projection MUST mirror the heatmap chain so the axes
                # land at the same pixel as the heatmap's peak. That means
                # routing through `cam_project_3d_to_2d(p, camera_params_naked,
                # camera_offset)` (same call the heatmap uses above) so the
                # YAML `camera_offset` matrix (e.g. diag(-1,-1,1) on the canned
                # dVRK rig) is applied consistently. The earlier
                # `project_axes_for_camera` helper did not apply camera_offset
                # and produced visibly-misaligned axes.
                if draw_this_cam:
                    # Reuse the same pose computation rather than re-running
                    # the full chain.
                    pose_draw = compute_calibrated_tip_pose(
                        R_local=R_local, t_local=t_local,
                        T_W_B=T_W_B, T_C_W=T_C_W,
                    )

                    # 3D endpoints of the X/Y/Z axis triad in the **tip** frame.
                    L_axis = drawframe_cfg.axis_length_m
                    pts_tip = np.array([
                        [0.0, 0.0, 0.0],   # origin
                        [L_axis, 0.0, 0.0],   # +X
                        [0.0, L_axis, 0.0],   # +Y
                        [0.0, 0.0, L_axis],   # +Z
                    ])  # shape (4, 3)
                    # Transform endpoints into the (left-rectified) camera frame
                    # via the calibrated tool-tip pose.
                    pts_cam = (pose_draw.R_cam_tip @ pts_tip.T).T + pose_draw.t_cam_tip.reshape(1, 3)

                    # Behind-camera test in the camera_offset-adjusted frame
                    # (so the test agrees with the projection that follows).
                    # When camera_offset has no negative-z component the
                    # adjusted z equals the raw z, but doing the rotation
                    # explicitly keeps the check correct for arbitrary offsets.
                    if camera_offset is not None:
                        origin_adj = camera_offset @ pose_draw.t_cam_tip
                    else:
                        origin_adj = pose_draw.t_cam_tip
                    is_behind_cam = float(origin_adj[2]) <= 0.0

                    # Load the rectified image (color — we always re-read from
                    # disk for drawframe even when enable_overlay loaded the
                    # greyscale).
                    rect_img_path = drawframe_img_list[int(file_name.stem)]
                    img_color = cv2.imread(str(rect_img_path), cv2.IMREAD_COLOR)

                    if is_behind_cam:
                        behind_camera_count += 1
                        logging.debug(
                            f"[Frame {file_name.stem} | {arm_name} | {camera_names[i_cam]}] "
                            f"tool tip behind camera (adjusted z<=0); writing pass-through image."
                        )
                        out_img = img_color
                    else:
                        # Project each endpoint through the same utility the
                        # heatmap uses. This applies camera_offset, the per-
                        # camera scaled K, and the distortion model from the
                        # raw calibration — matching the heatmap exactly.
                        pts_uv = cam_project_3d_to_2d(
                            pts_cam.astype(np.float64),
                            camera_params_naked,
                            camera_offset,
                        )
                        out_img = render_drawframe_axes(img_color, pts_uv, drawframe_cfg)
                    drawframe_out_path = drawframe_out_dir / img_file_name
                    cv2.imwrite(str(drawframe_out_path), out_img)


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
                    logging.info(
                        f"[Frame {frame_id} | {arm_name}]  dVRK 3-D (cam): "
                        f"{t if 't' in locals() else 'N/A'}"
                    )
                    logging.info(
                        f"[Frame {frame_id} | {arm_name}]  dVRK 2-D (u,v): "
                        f"{pixel_coord[0,0]:7.1f}, {pixel_coord[0,1]:7.1f}"
                    )
                    logging.info(
                        f"[Frame {frame_id} | {arm_name}]  HE   3-D (cam): "
                        f"{tip_cam_handeye.round(4)}"
                    )
                    logging.info(
                        f"[Frame {frame_id} | {arm_name}]  HE   2-D (u,v): "
                        f"{pixel_coord_handeye[0,0]:7.1f}, {pixel_coord_handeye[0,1]:7.1f}"
                    )
                    # quick sanity: delta in pixels
                    du = pixel_coord_handeye[0,0] - pixel_coord[0,0]
                    dv = pixel_coord_handeye[0,1] - pixel_coord[0,1]
                    logging.info(
                        f"[Frame {frame_id} | {arm_name}]  Δ(pixel HE-dVRK): "
                        f"{du:+6.1f}, {dv:+6.1f}\n"
                    )

    if drawframe_cfg.enable and behind_camera_count > 0:
        logging.warning(
            f"drawframe: {behind_camera_count} frame(s) had the tool tip behind the camera "
            f"(z<=0). The pass-through copy was written instead. This usually indicates "
            f"a hand-eye drift or a tool that's not visible in the surgical view."
        )


if __name__ == '__main__':
    main()
    print('Kinematic Mapping Done!')
