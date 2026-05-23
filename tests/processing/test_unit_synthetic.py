"""
Synthetic unit tests for the preprocessing building blocks.

These are deliberately tiny (numerical sanity checks on hand-built inputs)
so they can run on any CPU-only machine without the sample data. They cover:

  - rotation composition for ``compute_calibrated_tip_pose`` and
    the asymmetric setpoint variant.
  - ``disparity_to_depth_m`` round-trip and NaN handling.
  - analytical projection of a known axis triad through a known K.

Run with::

    cd /home/jackzhy/claude_code_projects/dvrk_multimodal_data_collection
    python tests/processing/test_unit_synthetic.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

# Make sure the local src package is importable when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dvrk_data_processing.kinematic_mapping.kinematic_handeye import (
    compute_calibrated_tip_pose, compute_calibrated_setpoint_pose,
    project_axes_for_camera, CalibratedPose,
)
from dvrk_data_processing.depth_estimation.depth_utils import (
    disparity_to_depth_m,
)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _assert_close(a, b, label: str, atol=1e-9) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.allclose(a, b, atol=atol):
        raise AssertionError(f"{label} mismatch:\n  expected {b}\n  got      {a}")


# --------------------------------------------------------------------------- #
# Calibrated kinematics
# --------------------------------------------------------------------------- #

def test_calibrated_pose_identity():
    """Identity inputs → identity pose."""
    pose = compute_calibrated_tip_pose(
        R_local=np.eye(3), t_local=np.array([0.1, 0.2, 0.3]),
        T_W_B=np.eye(4), T_C_W=np.eye(4),
    )
    _assert_close(pose.R_cam_tip, np.eye(3), "identity R")
    _assert_close(pose.t_cam_tip, [0.1, 0.2, 0.3], "identity t")


def test_calibrated_pose_compose_z():
    """
    Rotate around Z by 30° in PSM base, translate world by [1,0,0], camera
    sits at -2x of world (T_C_W translates the world into camera by [+2,0,0]).
    Hand-computed expectation below.
    """
    theta = math.radians(30)
    R_local = _rot_z(theta)
    t_local = np.array([0.1, 0.0, 0.0])

    T_W_B = np.eye(4); T_W_B[:3, 3] = [1.0, 0.0, 0.0]
    T_C_W = np.eye(4); T_C_W[:3, 3] = [2.0, 0.0, 0.0]

    pose = compute_calibrated_tip_pose(R_local, t_local, T_W_B, T_C_W)

    # Expected: R_cam_tip = I @ I @ Rz(30) = Rz(30)
    _assert_close(pose.R_cam_tip, _rot_z(theta), "compose R")
    # Expected t: T_W_B applies [+1,0,0] → world tip = [1.1, 0, 0]; then
    # T_C_W applies [+2,0,0] → cam tip = [3.1, 0, 0].
    _assert_close(pose.t_cam_tip, [3.1, 0.0, 0.0], "compose t")


def test_setpoint_skips_TWB():
    """Setpoint applies only T_C_W — not T_W_B — by spec."""
    T_W_B = np.eye(4); T_W_B[:3, 3] = [10.0, 10.0, 10.0]   # large, would shift if applied
    T_C_W = np.eye(4); T_C_W[:3, 3] = [0.5, 0.0, 0.0]
    R_W_set = np.eye(3)
    t_W_set = np.array([0.0, 0.0, 0.1])

    pose = compute_calibrated_setpoint_pose(R_W_set, t_W_set, T_C_W)
    _assert_close(pose.t_cam_tip, [0.5, 0.0, 0.1], "setpoint t skips T_W_B")
    _assert_close(pose.R_cam_tip, np.eye(3), "setpoint R")


# --------------------------------------------------------------------------- #
# Depth conversion
# --------------------------------------------------------------------------- #

def test_depth_round_trip():
    """Pick a known fx/baseline, known disparity → check depth analytically."""
    fx = 1175.93
    baseline = 0.004183       # ≈ 4.18 mm (sample data)
    disparity = np.array([[50.0, 100.0],
                          [25.0, 12.5]])
    depth = disparity_to_depth_m(disparity, fx, baseline, eps=1e-3)
    expected = (baseline * fx) / disparity
    _assert_close(depth, expected.astype(np.float32), "depth = b*fx/disp", atol=1e-6)


def test_depth_zero_disparity_is_nan():
    """Zero disparity → NaN (never inf, never a saturated positive)."""
    fx = 1175.93
    baseline = 0.004183
    disparity = np.array([[0.0, 1e-9, 50.0]])
    depth = disparity_to_depth_m(disparity, fx, baseline, eps=1e-3)
    assert math.isnan(float(depth[0, 0])), "zero disp must produce NaN"
    assert math.isnan(float(depth[0, 1])), "sub-eps disp must produce NaN"
    assert not math.isnan(float(depth[0, 2])), "valid disp must produce finite depth"
    # And the valid one is in a sensible range for the chosen parameters.
    assert 0.05 < float(depth[0, 2]) < 0.5, f"unexpected depth {depth[0,2]}"


def test_depth_nan_passthrough():
    """NaN disparity → NaN depth (don't masquerade as eps-failure)."""
    fx = 1175.93
    baseline = 0.004183
    disparity = np.array([[float('nan'), 100.0]])
    depth = disparity_to_depth_m(disparity, fx, baseline, eps=1e-3)
    assert math.isnan(float(depth[0, 0])), "NaN disp must propagate to NaN depth"
    assert not math.isnan(float(depth[0, 1]))


# --------------------------------------------------------------------------- #
# Drawframe projection
# --------------------------------------------------------------------------- #

def test_drawframe_projection_identity():
    """
    With identity pose at t=[0,0,0.1] and a known K, projected axes land at
    analytically-predicted pixels.

      K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      origin → (cx, cy)            since x=y=0, z=0.1
      x-axis → (cx + fx*L/0.1, cy)
      y-axis → (cx, cy + fy*L/0.1)
      z-axis → (cx, cy)            since x=y=0, z=0.1+L; the *pixel* is still (cx, cy)
                                   for a frontal-z move (only depth changes)
    """
    fx, fy, cx, cy = 1000.0, 1000.0, 256.0, 144.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    pose = CalibratedPose(R_cam_tip=np.eye(3),
                          t_cam_tip=np.array([0.0, 0.0, 0.1]))
    L = 0.01
    uv = project_axes_for_camera(pose=pose, axis_length_m=L,
                                 K_cam=K, baseline_m=0.004,
                                 cam_name='left')
    assert uv is not None
    # origin
    _assert_close(uv[0], [cx, cy], "origin", atol=1e-6)
    # x-axis endpoint at z=0.1: u = cx + fx*L/0.1; v = cy
    _assert_close(uv[1], [cx + fx * L / 0.1, cy], "x-axis", atol=1e-6)
    # y-axis endpoint at z=0.1: u = cx; v = cy + fy*L/0.1
    _assert_close(uv[2], [cx, cy + fy * L / 0.1], "y-axis", atol=1e-6)
    # z-axis endpoint at z=0.1+L; u = cx; v = cy
    _assert_close(uv[3], [cx, cy], "z-axis", atol=1e-6)


def test_drawframe_behind_camera_returns_none():
    """z<=0 must return None so the caller writes the unmodified pass-through."""
    fx, cx, cy = 1000.0, 256.0, 144.0
    K = np.array([[fx, 0, cx],
                  [0, fx, cy],
                  [0, 0, 1]])
    pose = CalibratedPose(R_cam_tip=np.eye(3),
                          t_cam_tip=np.array([0.0, 0.0, -0.05]))
    uv = project_axes_for_camera(pose=pose, axis_length_m=0.01,
                                 K_cam=K, baseline_m=0.004,
                                 cam_name='left')
    assert uv is None, "behind-camera should yield None"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [
        test_calibrated_pose_identity,
        test_calibrated_pose_compose_z,
        test_setpoint_skips_TWB,
        test_depth_round_trip,
        test_depth_zero_disparity_is_nan,
        test_depth_nan_passthrough,
        test_drawframe_projection_identity,
        test_drawframe_behind_camera_returns_none,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
