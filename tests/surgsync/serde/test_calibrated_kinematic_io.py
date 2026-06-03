"""Calibrated kinematic JSON round-trip tests."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.serde.calibrated_kinematic_io import (
    CalibratedKinematicSample,
    load_calibrated_frame,
    calibrated_sample_to_dict,
)


REPO = Path(__file__).resolve().parents[3]
ONLINE = REPO / "data" / "online_data" / "2" / "preprocess" / "kinematic_reproject" / "PSM1" / "calibrated_kinematic"
OFFLINE = REPO / "data" / "offline_data" / "3" / "preprocess" / "kinematic_reproject" / "PSM1" / "calibrated_kinematic"


@pytest.mark.skipif(not ONLINE.exists(), reason="preprocessing calibrated_kinematic not present (online)")
def test_online_carries_both_measured_and_setpoint():
    s = load_calibrated_frame(ONLINE / "0.json", "PSM1", 0)
    assert s.measured_cp_calibrated is not None
    pos, orient = s.measured_cp_calibrated
    assert len(pos) == 3
    assert len(orient) == 4
    # Online recorder → setpoint_cp_calibrated populated.
    assert s.setpoint_cp_calibrated is not None


@pytest.mark.skipif(not OFFLINE.exists(), reason="preprocessing calibrated_kinematic not present (offline)")
def test_offline_omits_setpoint_cp_calibrated():
    s = load_calibrated_frame(OFFLINE / "0.json", "PSM1", 0)
    assert s.measured_cp_calibrated is not None
    # Offline recorder has no setpoint_cp → key omitted in JSON → None here.
    assert s.setpoint_cp_calibrated is None


def test_inverse_omits_setpoint_when_none(tmp_path: Path):
    """Inverse should re-emit JSON with the `setpoint_cp_calibrated`
    key entirely absent when the field is None — matching the
    offline-recorder semantics."""
    s = CalibratedKinematicSample(
        frame=0, arm_name="PSM1",
        measured_cp_calibrated=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        setpoint_cp_calibrated=None,
    )
    d = calibrated_sample_to_dict(s)
    assert "measured_cp_calibrated" in d
    assert "setpoint_cp_calibrated" not in d
    # Round-trip via JSON.
    s2_path = tmp_path / "out.json"
    s2_path.write_text(json.dumps(d))
    s2 = load_calibrated_frame(s2_path, "PSM1", 0)
    assert s2.setpoint_cp_calibrated is None
    assert s2.measured_cp_calibrated is not None
