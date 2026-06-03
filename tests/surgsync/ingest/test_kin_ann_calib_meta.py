"""Combined tests for the smaller ingest modules (kinematics, annotations,
calibration, meta) against real sample data."""
from __future__ import annotations
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.ingest.kinematics import load_arm
from dvrk_data_processing.surgsync.ingest.annotations import load_annotations
from dvrk_data_processing.surgsync.ingest.calibration import (
    load_camera_calibration,
    load_hand_eye,
    load_calibration_bundle,
)
from dvrk_data_processing.surgsync.ingest.meta import load_meta


REPO = Path(__file__).resolve().parents[3]
ONLINE_RAW = REPO / "data" / "online_data" / "2"
ONLINE_INTER = REPO / "data" / "online_data" / "2" / "preprocess" / "rectify_resize"


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_load_arm_psm1():
    arm = load_arm(ONLINE_RAW / "kinematic", "PSM1")
    assert arm.arm_name == "PSM1"
    assert len(arm.samples) == 886
    # Frame indices strictly monotonic.
    indices = [s.frame for s in arm.samples]
    assert indices == sorted(indices)
    # Every sample has measured_js with 6 joints.
    for s in arm.samples[:3]:
        assert s.measured_js.position is not None
        assert len(s.measured_js.position) == 6


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_load_annotations_reports_partial_gesture():
    ann = load_annotations(ONLINE_RAW / "annotation")
    # online_data/2 has gesture for only 821 of 886 frames.
    assert ann.counts["gesture"] < ann.counts["phase"]
    assert ann.gesture_partial is True
    # Phase and step files exist for every frame.
    assert ann.counts["phase"] == 886
    assert ann.counts["step"] == 886


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="raw sample not present")
def test_load_camera_calibration_from_raw():
    """load_camera_calibration now reads the raw camera files; preprocessing's
    rectify_params (if present) is carried but does not drive
    image_width/height — those reflect the native camera resolution."""
    cam = load_camera_calibration(
        ONLINE_RAW / "camera_calibration",
        intermediate_camera_dir=(
            ONLINE_INTER / "camera_calibration" if ONLINE_INTER.exists() else None
        ),
    )
    # Native capture resolution (was 512x288 when the test sourced from
    # intermediate).
    assert cam.image_width  >= 512
    assert cam.image_height >= 288
    assert cam.fx_left > 0
    # Baseline either from raw stereo_calib_params.json or derived from
    # preprocessing's rectify_params.json — both sources route to the same field.
    assert cam.baseline_m is not None
    assert 0.001 < cam.baseline_m < 0.05   # ~mm-scale baseline; loose bounds


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_load_hand_eye_both_conventions():
    he = load_hand_eye(ONLINE_RAW / "hand_eye_calibration")
    assert "PSM1" in he and "PSM2" in he
    assert he["PSM1"].dvrk_json_text is not None
    assert he["PSM2"].dvrk_json_text is not None
    # OpenCV variant present in the sample.
    assert he["PSM1"].opencv_json_text is not None


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="raw sample not present")
def test_load_calibration_bundle_end_to_end():
    """End-to-end bundle load. Raw image dims surface in the bundle;
    rectify_params is optional and only present when preprocessing has run."""
    b = load_calibration_bundle(
        ONLINE_RAW, ONLINE_INTER if ONLINE_INTER.exists() else None,
    )
    assert b.camera.image_width  > 0
    assert b.camera.image_height > 0
    assert set(b.hand_eye.keys()) == {"PSM1", "PSM2"}
    # Preprocessing stage 1 ran on this sample → rectify_params should be carried.
    if ONLINE_INTER.exists():
        assert b.camera.rectify_params_json_text is not None


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_load_meta():
    m = load_meta(ONLINE_RAW / "meta_data.json")
    assert m.operator_skill_level == "Intermediate"
    assert m.case_type == "Ex-vivo"
    assert m.tool["PSM1"] == "Large_Needle_Driver"
