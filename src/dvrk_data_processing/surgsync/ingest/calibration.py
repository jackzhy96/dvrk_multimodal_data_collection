"""Load camera + hand-eye calibration into typed records.

Sources of truth on disk:
- `raw_dir/camera_calibration/{left,right}.yaml`         — **original**
  CRTK YAMLs at the camera's native resolution. These are the canonical
  calibration files; everything else is derived from them.
- `raw_dir/camera_calibration/stereo_calib_params.json`  — original
  stereo extrinsics + baseline.
- `raw_dir/hand_eye_calibration/PSM{1,2}-registration-{dVRK,open-cv}.json`
  — both conventions kept verbatim.
- `intermediate_dir/camera_calibration/rectify_params.json`  — **only
  produced by preprocessing's rectify_resize stage**. Records the OpenCV
  `stereoRectify` P1/P2/Q output at the resized resolution. Optional.

The packer copies all of these verbatim into the episode's
`calibration/` folder; consumers re-derive the rectified-resolution
intrinsics on demand from the raw YAMLs + rectify_params.json.

This is a deliberate departure from an earlier design that shipped the
already-scaled intermediate YAMLs as the canonical calibration — those
were lossy (rounded to the resized resolution) and surprised consumers
who expected raw calibration to round-trip the original capture.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """Verbatim original calibration files + a few parsed conveniences.

    Field naming reflects the *raw* (original-resolution) intrinsics:
    `image_width` / `image_height` are the camera's native pixel
    dimensions, and `fx_left` is the left camera's focal length in
    pixels at that resolution. `rectify_params_json_text` is set only
    when preprocessing's rectify_resize stage produced one — consumers can use it
    to compute the rectified-resolution intrinsics if they need them.
    """
    left_yaml_text: str            # raw CRTK YAML, verbatim
    right_yaml_text: str
    stereo_calib_params_json_text: Optional[str]  # raw, verbatim
    rectify_params_json_text: Optional[str]       # preprocessing output, verbatim or None
    # Parsed convenience values (always from the raw YAML):
    image_width: int
    image_height: int
    fx_left: float
    baseline_m: Optional[float] = None


@dataclass
class HandEye:
    """One arm's hand-eye registration. Both conventions kept verbatim."""
    arm_name: str
    dvrk_json_text: Optional[str]    # raw file contents (None if missing)
    opencv_json_text: Optional[str]


@dataclass
class CalibrationBundle:
    camera: CameraCalibration
    hand_eye: dict[str, HandEye]   # keyed by "PSM1" / "PSM2"


def _read_text(p: Path) -> str:
    return p.read_text()


def _safe_read_text(p: Path) -> Optional[str]:
    return p.read_text() if p.exists() else None


def _parse_camera_yaml(yaml_text: str) -> dict:
    import yaml
    return yaml.safe_load(yaml_text)


def _camera_yaml_image_size(yaml_text: str) -> tuple[int, int]:
    d = _parse_camera_yaml(yaml_text)
    w = d.get("image_width")
    h = d.get("image_height")
    if w is None or h is None:
        raise ValueError("camera YAML missing image_width / image_height")
    return int(w), int(h)


def _camera_yaml_fx(yaml_text: str) -> float:
    """First element of `camera_matrix.data` (3×3 row-major) = fx."""
    d = _parse_camera_yaml(yaml_text)
    cm = d.get("camera_matrix")
    if not cm or "data" not in cm:
        raise ValueError("camera YAML missing camera_matrix.data")
    return float(cm["data"][0])


def load_camera_calibration(
    raw_camera_dir: Path,
    intermediate_camera_dir: Optional[Path] = None,
) -> CameraCalibration:
    """Read original `{left,right}.yaml` + optional `stereo_calib_params.json`
    from `raw_camera_dir`. If `intermediate_camera_dir` is supplied and
    contains a `rectify_params.json` (only present after preprocessing stage 1),
    that gets carried through too.
    """
    raw_camera_dir = Path(raw_camera_dir)
    left_p  = raw_camera_dir / "left.yaml"
    right_p = raw_camera_dir / "right.yaml"
    if not left_p.exists():
        raise FileNotFoundError(f"missing {left_p}")
    if not right_p.exists():
        raise FileNotFoundError(f"missing {right_p}")

    left_text  = _read_text(left_p)
    right_text = _read_text(right_p)
    stereo_p   = raw_camera_dir / "stereo_calib_params.json"
    stereo_text = _safe_read_text(stereo_p)

    rect_text: Optional[str] = None
    if intermediate_camera_dir is not None:
        rect_p = Path(intermediate_camera_dir) / "rectify_params.json"
        rect_text = _safe_read_text(rect_p)

    w, h = _camera_yaml_image_size(left_text)
    fx = _camera_yaml_fx(left_text)

    # baseline_m: prefer raw stereo_calib_params if present; else derive
    # from rectify_params P2's translation (`baseline_m = -P2[0][3] /
    # fx_rect`). Falls back to None if neither source can provide it.
    baseline_m: Optional[float] = None
    if stereo_text is not None:
        try:
            d = json.loads(stereo_text)
            if "baseline_m" in d:
                baseline_m = float(d["baseline_m"])
        except (ValueError, KeyError):
            baseline_m = None
    if baseline_m is None and rect_text is not None:
        try:
            r = json.loads(rect_text)
            P2 = np.array(r["P2"], dtype=np.float64)   # 3×4
            fx_rect = float(P2[0, 0])
            baseline_m = float(-P2[0, 3] / fx_rect)
        except Exception:
            baseline_m = None

    return CameraCalibration(
        left_yaml_text=left_text,
        right_yaml_text=right_text,
        stereo_calib_params_json_text=stereo_text,
        rectify_params_json_text=rect_text,
        image_width=w,
        image_height=h,
        fx_left=fx,
        baseline_m=baseline_m,
    )


def load_hand_eye(hand_eye_dir: Path) -> dict[str, HandEye]:
    """Load both conventions for PSM1 and PSM2 verbatim. dVRK is the
    canonical projection chain; open-cv is preserved for forward compat
    with consumers that want the OpenCV camera frame."""
    out: dict[str, HandEye] = {}
    for arm in ("PSM1", "PSM2"):
        dvrk_p = hand_eye_dir / f"{arm}-registration-dVRK.json"
        opencv_p = hand_eye_dir / f"{arm}-registration-open-cv.json"
        out[arm] = HandEye(
            arm_name=arm,
            dvrk_json_text=_safe_read_text(dvrk_p),
            opencv_json_text=_safe_read_text(opencv_p),
        )
    return out


def load_calibration_bundle(
    raw_dir: Path,
    intermediate_dir: Optional[Path] = None,
) -> CalibrationBundle:
    """End-to-end calibration loader — what `pipeline/per_clip.py` calls.

    `raw_dir` is the clip's raw root (`raw_dir/camera_calibration/` and
    `raw_dir/hand_eye_calibration/` are the canonical sources).
    `intermediate_dir` is preprocessing's `preprocess/rectify_resize/`
    output; when present the rectify_params.json gets carried into the
    bundle. When absent (no preprocessing), the bundle ships raw-only
    — still valid.
    """
    cam = load_camera_calibration(
        raw_camera_dir=raw_dir / "camera_calibration",
        intermediate_camera_dir=(
            intermediate_dir / "camera_calibration" if intermediate_dir is not None else None
        ),
    )
    he = load_hand_eye(raw_dir / "hand_eye_calibration")
    return CalibrationBundle(camera=cam, hand_eye=he)
