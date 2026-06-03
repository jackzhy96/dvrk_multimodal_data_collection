"""
Shared utilities for the hand-eye-based kinematic mapping pipeline.

Holds two pieces of logic that the entry point (``gen_kinematic_heatmap_handeye.py``)
shells out to so the main file stays focused on Hydra glue and the per-frame
loop:

1. ``compute_calibrated_tip_pose`` — runs the hand-eye transform chain to
   produce the **6-DoF tool-tip pose** in the left-rectified camera frame. The
   chain is:

       T_W_E   : world ← ECM-tip (per-frame, from ECM kinematic JSON)
       T_C_W   = inv(T_W_E)        camera ≈ ECM-tip
       T_W_B   : world ← PSM-RCM   (constant, from hand-eye calibration)
       R_local : PSM tool tip rotation in PSM base
       t_local : PSM tool tip translation in PSM base
       (R_local, t_local) possibly extended by the tool-tip offset already

       R_W_tip  = R_W_B @ R_local
       t_W_tip  = R_W_B @ t_local + p_W_B
       R_cam_tip = R_C_W @ R_W_tip
       t_cam_tip = R_C_W @ t_W_tip + p_C_W

   This is the same translation-only chain the entry point used previously
   for the heatmap projection (``tip_cam_handeye``), but extended to compose
   the rotation too — the existing code discarded the rotation.

2. ``write_calibrated_kinematic_json`` — atomic, per-frame JSON writer that
   conforms to the calibrated_kinematic schema:

       {
         "frame": <int>,
         "arm_name": "PSM1" | "PSM2",
         "measured_cp_calibrated": {"position": [x,y,z],
                                    "orientation": [qx,qy,qz,qw]},
         "setpoint_cp_calibrated": <same, omitted if raw setpoint_cp missing>
       }

3. ``compute_calibrated_setpoint_pose`` — applies only ``T_C_W`` to a
   world-frame setpoint (no ``T_W_B``; see asymmetry note in the spec).

4. ``render_drawframe_axes`` and ``project_axes_for_camera`` —
   drawframe helpers: project the tool tip's 3D axes through the scaled
   left/right intrinsics and draw colored lines on the rectified image.

Quaternion convention everywhere is **xyzw** (matches ``scipy.spatial.transform.
Rotation.as_quat()`` and the dVRK raw kinematic schema).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
# Pose computation
# --------------------------------------------------------------------------- #

@dataclass
class CalibratedPose:
    """Tool-tip pose in the left-rectified camera frame."""
    R_cam_tip: np.ndarray            # 3x3 rotation matrix
    t_cam_tip: np.ndarray            # length-3 translation vector (meters)

    def to_position_quat(self) -> Tuple[list, list]:
        """Serialize as (position[xyz], orientation[xyzw])."""
        # scipy's Rotation.as_quat() returns xyzw by default; we keep that
        # ordering to match the raw dVRK kinematic JSON convention.
        quat_xyzw = R.from_matrix(self.R_cam_tip).as_quat()
        return self.t_cam_tip.astype(float).tolist(), quat_xyzw.astype(float).tolist()


def compute_calibrated_tip_pose(R_local: np.ndarray,
                                t_local: np.ndarray,
                                T_W_B: np.ndarray,
                                T_C_W: np.ndarray) -> CalibratedPose:
    """
    Compose the per-frame tool-tip pose in the left-rectified camera frame.

    R_local, t_local already include the tool-tip offset (the caller applies
    ``apply_tool_tip_offset`` first), so this function only does the
    PSM-base→world→camera chain.

    R_local : 3x3 rotation matrix of the PSM tool tip in PSM base
    t_local : length-3 translation vector of the PSM tool tip in PSM base
    T_W_B   : 4x4 hand-eye matrix, world ← PSM base
    T_C_W   : 4x4 ECM-derived matrix, camera ≈ world  (inv of T_W_E)
    output  : CalibratedPose with (R_cam_tip, t_cam_tip)
    """
    R_W_B = T_W_B[:3, :3]
    p_W_B = T_W_B[:3, 3]
    R_C_W = T_C_W[:3, :3]
    p_C_W = T_C_W[:3, 3]

    # World-frame tool tip (rotation and translation composed independently;
    # this matches the existing translation-only code at lines 167–168 of the
    # original entry point, just extended with the rotation leg).
    R_W_tip = R_W_B @ R_local
    t_W_tip = R_W_B @ t_local + p_W_B

    # Camera-frame tool tip.
    R_cam_tip = R_C_W @ R_W_tip
    t_cam_tip = R_C_W @ t_W_tip + p_C_W

    return CalibratedPose(R_cam_tip=R_cam_tip, t_cam_tip=t_cam_tip)


def compute_calibrated_setpoint_pose(R_W_set: np.ndarray,
                                     t_W_set: np.ndarray,
                                     T_C_W: np.ndarray) -> CalibratedPose:
    """
    Apply only the ECM-derived T_C_W to a world-frame setpoint pose.

    Per the spec, the raw kinematic JSON does NOT expose ``local_setpoint_cp``
    (PSM base frame), so the full ``T_W_B``-based hand-eye chain used for the
    measured pose cannot be re-derived for the setpoint. We deliberately skip
    ``T_W_B`` here — re-applying it would double-count the registration error.

    R_W_set, t_W_set : world-frame setpoint rotation and translation
    T_C_W            : 4x4 ECM-derived matrix, camera ≈ world (inv of T_W_E)
    output           : CalibratedPose for the setpoint in camera frame
    """
    R_C_W = T_C_W[:3, :3]
    p_C_W = T_C_W[:3, 3]
    R_C_set = R_C_W @ R_W_set
    t_C_set = R_C_W @ t_W_set + p_C_W
    return CalibratedPose(R_cam_tip=R_C_set, t_cam_tip=t_C_set)


# --------------------------------------------------------------------------- #
# Calibrated_kinematic writer
# --------------------------------------------------------------------------- #

def write_calibrated_kinematic_json(out_path: Union[Path, str],
                                    frame_id: int,
                                    arm_name: str,
                                    measured_cp_calibrated: CalibratedPose,
                                    setpoint_cp_calibrated: Optional[CalibratedPose] = None) -> None:
    """
    Atomic per-frame JSON writer for calibrated_kinematic.

    The calibrated_kinematic schema:

        {
          "frame": 123,
          "arm_name": "PSM1",
          "measured_cp_calibrated": {
            "position":    [x, y, z],
            "orientation": [qx, qy, qz, qw]
          },
          // setpoint_cp_calibrated is OMITTED entirely when offline (raw
          // setpoint_cp absent), not written as null — this lets consumers
          // branch on key presence and keeps the file smaller.
          "setpoint_cp_calibrated": { ... }
        }

    Uses tempfile + os.replace for crash-safety: thousands of these tiny
    (<500 byte) files per clip mean a partial write would corrupt the dataset
    half-way through a long run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos_m, quat_m = measured_cp_calibrated.to_position_quat()
    payload = {
        "frame": int(frame_id),
        "arm_name": arm_name,
        "measured_cp_calibrated": {
            "position": pos_m,
            "orientation": quat_m,
        },
    }

    if setpoint_cp_calibrated is not None:
        pos_s, quat_s = setpoint_cp_calibrated.to_position_quat()
        payload["setpoint_cp_calibrated"] = {
            "position": pos_s,
            "orientation": quat_s,
        }
    # When the raw setpoint_cp is missing (offline recorder), skip the key
    # entirely — the spec calls this out explicitly.

    # Atomic write: write to a sibling temp file and rename. Both files live
    # on the same filesystem so os.replace() is atomic on POSIX.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)


# --------------------------------------------------------------------------- #
# Drawframe — render 3D tool-tip axes on rectified images
# --------------------------------------------------------------------------- #

@dataclass
class DrawframeConfig:
    """Plain-old-data drawframe knobs, decoupled from Hydra/OmegaConf types."""
    enable: bool
    axis_length_m: float
    line_thickness_px: int
    origin_marker_radius_px: int
    colors_bgr: dict          # {'x': (B,G,R), 'y': (...), 'z': (...)} integers
    cameras: Sequence[str]    # subset of ('left', 'right')


def project_axes_for_camera(pose: CalibratedPose,
                            axis_length_m: float,
                            K_cam: np.ndarray,
                            baseline_m: float,
                            cam_name: str) -> Optional[np.ndarray]:
    """
    **Synthetic-only** projection helper kept for the analytical unit test.

    Production drawframe in ``gen_kinematic_heatmap_handeye.py`` routes the
    axis endpoints through ``cam_project_3d_to_2d(p, camera_params_naked,
    camera_offset)`` so the YAML ``camera_offset`` matrix and the per-camera
    distortion / scaled K are applied consistently with the heatmap. This
    function does NOT apply ``camera_offset`` — it assumes the canonical
    OpenCV camera convention with no extra rotation. Use it only when you
    already have a clean K and want a one-shot analytical projection (the
    behavior tested in ``tests/processing/test_unit_synthetic.py``).

    For the **left** camera: the calibrated pose is already in the left-rectified
    frame, so we just apply ``K_left`` after the pose transform.

    For the **right** camera: after stereo rectification with
    ``CALIB_ZERO_DISPARITY`` (alpha=0), the right camera is a pure
    translation along ``-x`` by ``baseline_m`` relative to the left. So we
    translate the camera-frame points by ``[-baseline_m, 0, 0]`` and apply
    ``K_right``. The spec calls out this assumption explicitly — verify with
    the integration test.

    pose          : tool-tip pose in left-rectified camera frame
    axis_length_m : metric length of each axis arm
    K_cam         : 3x3 intrinsics for this camera (scaled)
    baseline_m    : stereo baseline in meters; only used for the right camera
    cam_name      : 'left' or 'right'

    output : (4, 2) array of pixel coordinates in order [origin, x, y, z],
             or None if the origin is behind the camera (z<=0). Pixel
             coordinates outside the image are NOT clipped here — cv2.line
             clips at draw time.
    """
    L = axis_length_m
    # Axis endpoints in the **tip** frame.
    pts_tip = np.array([
        [0.0, 0.0, 0.0],   # origin
        [L,   0.0, 0.0],   # +X
        [0.0, L,   0.0],   # +Y
        [0.0, 0.0, L  ],   # +Z
    ])  # shape (4, 3)

    # Tip → camera: P_cam = R_cam_tip @ P_tip + t_cam_tip
    pts_cam = (pose.R_cam_tip @ pts_tip.T).T + pose.t_cam_tip.reshape(1, 3)

    if cam_name == 'right':
        # Right-rectified camera frame: translate by -baseline_m along x.
        # (After stereoRectify with CALIB_ZERO_DISPARITY+alpha=0, R1=R2=I and
        # the right camera is a pure x-translation.)
        pts_cam = pts_cam + np.array([[-baseline_m, 0.0, 0.0]])

    # Behind-camera test: if the origin's z is <=0 there's no meaningful
    # projection (it would land at infinity or wrap behind the image). The
    # caller should still write the unmodified frame so indices stay aligned.
    if pts_cam[0, 2] <= 0:
        return None

    # Perspective projection (no distortion — the rectified image is undistorted).
    # uv_homog = K * P_cam; then divide by z. We do this per-point.
    z = pts_cam[:, 2:3]
    # Guard against tiny z values for axis endpoints (origin already filtered).
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9 * np.sign(z + 1e-12), z)
    pts_norm = pts_cam / z_safe              # (4, 3); last column ≈ 1
    pts_pixel = (K_cam @ pts_norm.T).T[:, :2]  # (4, 2)
    return pts_pixel


def render_drawframe_axes(image_bgr: np.ndarray,
                          axes_uv: np.ndarray,
                          cfg: DrawframeConfig) -> np.ndarray:
    """
    Draw the X/Y/Z axes onto a copy of the rectified image.

    image_bgr : 3-channel uint8 BGR image (rectified)
    axes_uv   : (4, 2) array of pixel coords in order [origin, x, y, z]
    cfg       : DrawframeConfig
    output    : a copy of the image with axes drawn on it
    """
    out = image_bgr.copy()
    if axes_uv is None:
        return out                           # caller handles "behind camera"

    # Clamp pixel coordinates to a sane integer range before passing to cv2.
    # ``cam_project_3d_to_2d`` can return ~1e12-magnitude values when a tip
    # endpoint sits at or near the focal plane (z → 0 makes u,v explode);
    # cv2.line() then rejects the input with
    #     "Can't parse 'pt2'. Sequence item with index 0 has a wrong type"
    # because the int conversion overflows int32. Clamping to a margin
    # around the image keeps cv2 happy and still draws whatever portion of
    # the axis actually crosses the visible window (cv2 clips line drawing
    # at the image bounds).
    h, w = image_bgr.shape[:2]
    margin = max(w, h) * 16     # generous bound — far off-screen but finite

    def _clip_pt(uv: np.ndarray) -> tuple:
        u, v = float(uv[0]), float(uv[1])
        # Replace NaN/inf with off-image-but-finite sentinels so cv2.line
        # can still process the call (we don't want a single bad frame to
        # kill the whole sweep).
        if not (np.isfinite(u) and np.isfinite(v)):
            u, v = -margin, -margin
        u = int(np.clip(round(u), -margin, w + margin))
        v = int(np.clip(round(v), -margin, h + margin))
        return (u, v)

    origin = _clip_pt(axes_uv[0])
    x_end = _clip_pt(axes_uv[1])
    y_end = _clip_pt(axes_uv[2])
    z_end = _clip_pt(axes_uv[3])

    cx = tuple(int(c) for c in cfg.colors_bgr['x'])
    cy = tuple(int(c) for c in cfg.colors_bgr['y'])
    cz = tuple(int(c) for c in cfg.colors_bgr['z'])
    thick = int(cfg.line_thickness_px)

    # Lines (anti-aliased to look reasonable when zoomed in).
    cv2.line(out, origin, x_end, cx, thickness=thick, lineType=cv2.LINE_AA)
    cv2.line(out, origin, y_end, cy, thickness=thick, lineType=cv2.LINE_AA)
    cv2.line(out, origin, z_end, cz, thickness=thick, lineType=cv2.LINE_AA)

    # White origin dot on top so it's visible on bright tissue and dark voids alike.
    radius = int(cfg.origin_marker_radius_px)
    cv2.circle(out, origin, radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    return out


def parse_drawframe_config(raw) -> DrawframeConfig:
    """
    Convert a Hydra (OmegaConf) drawframe sub-block into a frozen Python
    dataclass with the defaults baked in. Returning a dataclass decouples the
    per-frame draw loop from OmegaConf's lazy-resolution machinery.

    raw : OmegaConf node or plain dict with the same shape as the YAML below
          (see ``config/preprocess/kinematic_reproject.yaml``)
    output : DrawframeConfig
    """
    # Hydra may pass either a DictConfig or a plain dict — accept both. Falling
    # back to .get() keeps the function lenient when the user only sets a
    # subset of the knobs.
    def _get(node, key, default):
        if node is None:
            return default
        if hasattr(node, 'get'):
            v = node.get(key, default)
        else:
            v = getattr(node, key, default)
        return v if v is not None else default

    colors_node = _get(raw, 'colors_bgr', None)
    colors = {
        'x': tuple(_get(colors_node, 'x', (0, 0, 255))),     # red in BGR
        'y': tuple(_get(colors_node, 'y', (0, 255, 0))),     # green
        'z': tuple(_get(colors_node, 'z', (255, 0, 0))),     # blue
    }

    cams = _get(raw, 'cameras', ['left', 'right'])
    # Coerce OmegaConf ListConfig → plain list.
    cams = [str(c) for c in cams]

    return DrawframeConfig(
        enable=bool(_get(raw, 'enable', False)),
        axis_length_m=float(_get(raw, 'axis_length_m', 0.010)),
        line_thickness_px=int(_get(raw, 'line_thickness_px', 2)),
        origin_marker_radius_px=int(_get(raw, 'origin_marker_radius_px', 3)),
        colors_bgr=colors,
        cameras=cams,
    )


# --------------------------------------------------------------------------- #
# Setpoint extraction
# --------------------------------------------------------------------------- #

def read_raw_setpoint_cp(json_path: Union[Path, str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Peek at the raw kinematic JSON and return the world-frame setpoint pose,
    or None if absent (offline recorder, ECM, etc.).

    Returns (R_W_set: 3x3, t_W_set: length-3) when present.
    Returns None when the JSON has no ``setpoint_data.setpoint_cp`` key.

    Done here (rather than extending PSMCPInfo) because it is the only
    consumer of this field; pushing it into the dataclass loader would
    require touching the existing schema and risks breaking other scripts.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    # Both online and offline formats list-wrap the dict.
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    arm = raw.get('arm', None)
    if arm is None:
        return None
    setpoint_data = arm.get('setpoint_data', None)
    if setpoint_data is None:
        return None
    setpoint_cp = setpoint_data.get('setpoint_cp', None)
    if setpoint_cp is None:
        return None

    quat = setpoint_cp.get('orientation')
    pos = setpoint_cp.get('position')
    if quat is None or pos is None:
        # Malformed setpoint_cp block; log and treat as missing rather than crash.
        logging.warning(f"setpoint_cp present but malformed in {json_path}; skipping")
        return None

    # scipy expects xyzw (the dVRK convention). Build the rotation matrix.
    R_W_set = R.from_quat(quat).as_matrix()
    t_W_set = np.array(pos, dtype=float)
    return R_W_set, t_W_set


if __name__ == "__main__":
    pass
