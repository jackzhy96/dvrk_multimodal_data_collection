"""
Pure-numeric helpers for the depth-in-meters augmentation.

Extracted from ``gen_depth_estimate.py`` so they can be unit-tested without
pulling in heavy dependencies (tqdm, torch, FoundationStereo). The Hydra
entry point re-exports these.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


def load_stereo_depth_params(camera_calibration_path: Path,
                              stereo_calib_filename: str) -> Tuple[float, float]:
    """
    Load (fx_left, baseline_m) from the stage-1-scaled camera calibration.

    fx_left is the top-left entry of ``left.yaml`` ``camera_matrix`` (scaled
    to the stage-1 ``new_size``). ``baseline_m`` is read directly from
    ``stereo_calib_params.json``. The function is intentionally tolerant of
    layout drift (different filename, missing JSON) — see fall-back below.

    camera_calibration_path : intermediate_dir/camera_calibration/
    stereo_calib_filename   : name of the stereo JSON (default "stereo_calib_params.json")
    output : (fx_left, baseline_m) — both floats in meters / pixels
    """
    left_yaml = camera_calibration_path / 'left.yaml'
    if not left_yaml.exists():
        raise FileNotFoundError(
            f"Cannot compute depth — left.yaml missing under {camera_calibration_path}. "
            f"Make sure stage 1 (gen_rectify_resize.py) has run."
        )
    with open(left_yaml, 'r') as f:
        data = yaml.safe_load(f)
    K_flat = data['camera_matrix']['data']
    # Layout per CameraInfo dataclass: row-major 3x3, fx at index 0.
    fx_left = float(K_flat[0])

    stereo_json = camera_calibration_path / stereo_calib_filename
    if stereo_json.exists():
        with open(stereo_json, 'r') as f:
            stereo = json.load(f)
        # baseline_m is the field name produced by the dVRK stereo calibration
        # tool. Tolerate small key variations as a future-proofing measure.
        baseline_m = float(stereo.get('baseline_m', stereo.get('baseline', 0.0)))
        if baseline_m <= 0:
            # Some legacy files report baseline_m as the negative x-component
            # of T; what we want is its magnitude. Use abs as a guard.
            baseline_m = abs(baseline_m)
            if baseline_m == 0:
                raise ValueError(
                    f"baseline_m in {stereo_json} is zero — cannot compute depth."
                )
    else:
        # Fall-back: the magnitude of T_stereo from right.yaml is the same
        # baseline (up to sign). Useful when stage 1 didn't copy the JSON.
        right_yaml = camera_calibration_path / 'right.yaml'
        if not right_yaml.exists():
            raise FileNotFoundError(
                f"Neither {stereo_json} nor {right_yaml} exists; cannot get baseline."
            )
        with open(right_yaml, 'r') as f:
            rdata = yaml.safe_load(f)
        T_stereo = np.array(rdata['T_stereo']['data'])
        baseline_m = float(np.linalg.norm(T_stereo))
        logging.warning(
            f"{stereo_calib_filename} missing under {camera_calibration_path}; "
            f"falling back to ||T_stereo|| = {baseline_m:.6f} m."
        )
    return fx_left, baseline_m


def disparity_to_depth_m(disparity_px: np.ndarray, fx_left: float,
                         baseline_m: float, eps: float) -> np.ndarray:
    """
    Convert a disparity field (in pixels) to depth in meters.

    The standard rectified stereo formula:

        depth = (baseline * fx) / disparity

    Wherever ``disparity_px <= eps`` or ``disparity_px`` is NaN, the output
    is explicitly set to ``np.nan`` — never clamped to a saturated positive
    number. The NaN is the "model failed here" sentinel; downstream
    consumers (the packer's geometry encoder) NaN-mask before quantizing.

    disparity_px : (H, W) array of pixel disparities (any sign; we mask
                   non-positives as invalid)
    fx_left      : focal length in pixels (scaled, post-stage-1)
    baseline_m   : stereo baseline in meters
    eps          : disparity threshold below which depth is NaN
    output       : (H, W) float32 depth in meters, NaN-masked
    """
    # Work in float64 internally for the division, cast to float32 at the
    # end. This avoids accumulated rounding error near eps.
    disp = disparity_px.astype(np.float64, copy=False)
    invalid = np.logical_or(np.isnan(disp), disp <= eps)
    # np.errstate to suppress runtime warnings on the masked cells; we
    # overwrite them with NaN right after.
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (baseline_m * fx_left) / disp
    depth[invalid] = np.nan
    return depth.astype(np.float32)


# Name → cv2 colormap constant. Names are case-insensitive when looked up via
# `_resolve_cv2_colormap` below. Turbo is the modern professional default —
# perceptually monotonic, vivid, matches what Open3D, Intel RealSense, Azure
# Kinect, and FoundationStereo's own `vis_disparity` use. Inferno is kept for
# backwards-compat with older configs.
_CV2_COLORMAP_BY_NAME = {
    "turbo":    cv2.COLORMAP_TURBO,
    "jet":      cv2.COLORMAP_JET,
    "inferno":  cv2.COLORMAP_INFERNO,
    "magma":    cv2.COLORMAP_MAGMA,
    "plasma":   cv2.COLORMAP_PLASMA,
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "parula":   cv2.COLORMAP_PARULA,
    "hot":      cv2.COLORMAP_HOT,
    "bone":     cv2.COLORMAP_BONE,
    "rainbow":  cv2.COLORMAP_RAINBOW,
}


def _resolve_cv2_colormap(name: str) -> int:
    """Look up a cv2 colormap constant by case-insensitive name."""
    if not isinstance(name, str):
        raise TypeError(f"depth_viz_cmap must be a string, got {type(name).__name__}")
    key = name.strip().lower()
    if key not in _CV2_COLORMAP_BY_NAME:
        raise ValueError(
            f"Unknown depth_viz_cmap {name!r}. Supported names: "
            f"{sorted(_CV2_COLORMAP_BY_NAME)}"
        )
    return _CV2_COLORMAP_BY_NAME[key]


def colorize_depth_m(depth_m: np.ndarray, viz_range_m: Tuple[float, float],
                     cmap: str = "turbo") -> np.ndarray:
    """
    Render a depth-in-meters field as a colorized 8-bit BGR image.

    NaNs become black; values are clipped to ``viz_range_m`` before
    normalization to keep the colormap stable across a clip.

    depth_m     : (H, W) float32 depth in meters
    viz_range_m : (depth_min, depth_max) — colormap clipping window
    cmap        : name of a supported cv2 colormap. Defaults to "turbo"
                  (Google's perceptually-uniform Jet replacement, standard
                  for modern depth/disparity visualization).
    output      : (H, W, 3) uint8 BGR image
    """
    dmin, dmax = float(viz_range_m[0]), float(viz_range_m[1])
    if not dmax > dmin:
        raise ValueError(
            f"depth_viz_range_m must be (min, max) with max > min; got ({dmin}, {dmax})"
        )

    nan_mask = np.isnan(depth_m)
    safe = np.where(nan_mask, dmin, depth_m).astype(np.float32)
    safe = np.clip(safe, dmin, dmax)
    norm01 = (safe - dmin) / (dmax - dmin)
    norm_u8 = (norm01 * 255.0).astype(np.uint8)

    # cv2.applyColorMap returns BGR — directly compatible with cv2.imwrite.
    colorized = cv2.applyColorMap(norm_u8, _resolve_cv2_colormap(cmap))
    # Paint NaNs as solid black for an unambiguous "no depth here" signal.
    colorized[nan_mask] = (0, 0, 0)
    return colorized
