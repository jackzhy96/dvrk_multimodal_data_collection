"""Calibration encoder — verbatim round-trip from raw_dir + optional
preprocessing rectify_params."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.encode.calibration import write_calibration
from dvrk_data_processing.surgsync.ingest.calibration import load_calibration_bundle


REPO = Path(__file__).resolve().parents[3]
ONLINE_RAW = REPO / "data" / "online_data" / "2"
ONLINE_INTER = REPO / "data" / "online_data" / "2" / "preprocess" / "rectify_resize"


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_write_calibration_round_trip_raw_plus_intermediate(tmp_path: Path):
    """With both raw + intermediate present, all four artifact kinds
    land in the pack: raw left.yaml, raw right.yaml, raw
    stereo_calib_params.json, and preprocessing's rectify_params.json. The YAML
    bytes must match the **raw** files — not the preprocessing-scaled versions."""
    bundle = load_calibration_bundle(
        ONLINE_RAW, ONLINE_INTER if ONLINE_INTER.exists() else None,
    )
    write_calibration(bundle, tmp_path)

    # Verbatim copies must round-trip byte-for-byte.
    assert (tmp_path / "left.yaml").read_text() == bundle.camera.left_yaml_text
    assert (tmp_path / "right.yaml").read_text() == bundle.camera.right_yaml_text

    # The shipped YAML must match the **raw** file, not the scaled
    # intermediate one — that's the whole point of the source change.
    raw_left_text = (ONLINE_RAW / "camera_calibration" / "left.yaml").read_text()
    assert (tmp_path / "left.yaml").read_text() == raw_left_text

    # stereo_calib_params.json — shipped iff present in raw.
    raw_stereo_path = ONLINE_RAW / "camera_calibration" / "stereo_calib_params.json"
    if raw_stereo_path.is_file():
        assert (tmp_path / "stereo_calib_params.json").read_text() == raw_stereo_path.read_text()

    # rectify_params.json — shipped iff preprocessing stage 1 ran. This sample
    # has preprocessing outputs, so it should be there.
    if ONLINE_INTER.exists():
        rect_path = ONLINE_INTER / "camera_calibration" / "rectify_params.json"
        if rect_path.is_file():
            assert (tmp_path / "rectify_params.json").read_text() == rect_path.read_text()

    # Hand-eye preserved.
    assert (tmp_path / "hand_eye" / "PSM1-registration-dVRK.json").is_file()
    assert (tmp_path / "hand_eye" / "PSM2-registration-dVRK.json").is_file()
    assert (tmp_path / "hand_eye" / "PSM1-registration-open-cv.json").is_file()

    # Convenience index — `image_width/height` now reflect the RAW
    # camera resolution (1920x1080 on this sample), not the scaled
    # 512x288 the previous spec carried.
    cam = json.loads((tmp_path / "camera.json").read_text())
    assert cam["image_width"]  == bundle.camera.image_width
    assert cam["image_height"] == bundle.camera.image_height
    # On the sample data the native resolution is much larger than the
    # preprocessing-scaled thumbnail.
    assert cam["image_width"]  >= 512
    assert cam["image_height"] >= 288


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_calibration_works_without_m1(tmp_path: Path):
    """When preprocessing hasn't run on a clip, the bundle should still load and
    write — minus rectify_params.json. This is the common case for
    fresh clips before depth/flow have been computed."""
    bundle = load_calibration_bundle(ONLINE_RAW, intermediate_dir=None)
    assert bundle.camera.rectify_params_json_text is None
    write_calibration(bundle, tmp_path)

    assert (tmp_path / "left.yaml").is_file()
    assert (tmp_path / "right.yaml").is_file()
    assert not (tmp_path / "rectify_params.json").exists()
    assert (tmp_path / "hand_eye" / "PSM1-registration-dVRK.json").is_file()
    cam = json.loads((tmp_path / "camera.json").read_text())
    assert "rectify_params" not in cam["files"]
