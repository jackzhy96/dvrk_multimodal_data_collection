"""Write the per-episode `<staging>/calibration/` folder.

Layout (per the preprocessing → packing invertibility contract):

    calibration/
      left.yaml                    ← original raw CRTK YAML, verbatim
      right.yaml                   ← original raw CRTK YAML, verbatim
      stereo_calib_params.json     ← original raw JSON, verbatim (if present)
      rectify_params.json          ← preprocessing stage-1 output (if preprocessing ran)
      hand_eye/
        PSM{1,2}-registration-dVRK.json     ← original raw, verbatim
        PSM{1,2}-registration-open-cv.json  ← original raw, verbatim
      camera.json                  ← small convenience index for consumers

The raw YAMLs are at the native camera resolution. `rectify_params.json`
records the OpenCV `stereoRectify` output at the rectified resolution;
consumers who want the rectified-resolution intrinsics can derive them
from raw + rectify_params on demand, so we don't ship the scaled YAMLs
the preprocessing stage 1 emits.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from dvrk_data_processing.surgsync.ingest.calibration import CalibrationBundle


log = logging.getLogger(__name__)


def write_calibration(bundle: CalibrationBundle, dst_dir: Path) -> None:
    """Dump the calibration files under `<dst>/calibration/`."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Camera YAMLs — original raw resolution, verbatim.
    (dst_dir / "left.yaml").write_text(bundle.camera.left_yaml_text)
    (dst_dir / "right.yaml").write_text(bundle.camera.right_yaml_text)

    # Raw stereo extrinsics (if present on disk).
    if bundle.camera.stereo_calib_params_json_text is not None:
        (dst_dir / "stereo_calib_params.json").write_text(
            bundle.camera.stereo_calib_params_json_text
        )

    # rectify_params is preprocessing-only — only present after rectify_resize ran.
    if bundle.camera.rectify_params_json_text is not None:
        (dst_dir / "rectify_params.json").write_text(
            bundle.camera.rectify_params_json_text
        )

    # Hand-eye — both conventions where present.
    he_dir = dst_dir / "hand_eye"
    he_dir.mkdir(exist_ok=True)
    for arm, he in bundle.hand_eye.items():
        if he.dvrk_json_text is not None:
            (he_dir / f"{arm}-registration-dVRK.json").write_text(he.dvrk_json_text)
        if he.opencv_json_text is not None:
            (he_dir / f"{arm}-registration-open-cv.json").write_text(he.opencv_json_text)

    # Convenience camera.json — quick-glance pointer the unpack reader can
    # use without parsing YAML. Reflects RAW resolution; rectified
    # intrinsics live in rectify_params.json when present.
    camera_index = {
        "image_width":  bundle.camera.image_width,
        "image_height": bundle.camera.image_height,
        "fx_left":      bundle.camera.fx_left,
        "baseline_m":   bundle.camera.baseline_m,
        "files": {
            "left":   "left.yaml",
            "right":  "right.yaml",
        },
    }
    if bundle.camera.stereo_calib_params_json_text is not None:
        camera_index["files"]["stereo_calib_params"] = "stereo_calib_params.json"
    if bundle.camera.rectify_params_json_text is not None:
        camera_index["files"]["rectify_params"] = "rectify_params.json"
    (dst_dir / "camera.json").write_text(json.dumps(camera_index, indent=2))
