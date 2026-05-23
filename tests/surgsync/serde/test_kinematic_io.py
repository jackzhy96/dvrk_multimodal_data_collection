"""Forward + inverse round-trip tests for `serde/kinematic_io.py` using
the real sample data on disk."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.serde.kinematic_io import (
    KinematicSample,
    JointSnapshot,
    CartesianSnapshot,
    load_arm_frame_json,
    kinematic_sample_to_raw_dict,
)


REPO = Path(__file__).resolve().parents[3]
ONLINE_PSM1 = REPO / "data" / "online_data" / "2" / "kinematic" / "PSM1"
OFFLINE_PSM1 = REPO / "data" / "offline_data" / "3" / "kinematic" / "PSM1"
ONLINE_ECM = REPO / "data" / "online_data" / "2" / "kinematic" / "ECM"


@pytest.mark.skipif(not ONLINE_PSM1.exists(), reason="sample data not present")
def test_online_psm1_load_carries_setpoint_cp():
    s = load_arm_frame_json(ONLINE_PSM1 / "0.json", "PSM1", 0)
    assert s.frame == 0 and s.arm_name == "PSM1"
    assert s.measured_js.position is not None and len(s.measured_js.position) == 6
    assert s.measured_cp.position is not None and len(s.measured_cp.position) == 3
    assert s.measured_cp.orientation is not None and len(s.measured_cp.orientation) == 4
    # Online recorder has setpoint_cp populated.
    assert s.setpoint_cp is not None
    assert s.setpoint_cp.position is not None and len(s.setpoint_cp.position) == 3
    # PSMs have jaw + source_frequency_hz.
    assert s.measured_jaw_position is not None
    assert s.source_frequency_hz is not None


@pytest.mark.skipif(not OFFLINE_PSM1.exists(), reason="sample data not present")
def test_offline_psm1_load_omits_setpoint_cp():
    s = load_arm_frame_json(OFFLINE_PSM1 / "0.json", "PSM1", 0)
    # Offline recorder has no Cartesian setpoint.
    assert s.setpoint_cp is None
    # measured_cp still populated.
    assert s.measured_cp.position is not None


@pytest.mark.skipif(not ONLINE_ECM.exists(), reason="sample data not present")
def test_ecm_load_has_no_jaw_no_source_frequency():
    s = load_arm_frame_json(ONLINE_ECM / "0.json", "ECM", 0)
    assert s.measured_jaw_position is None
    assert s.setpoint_jaw_position is None
    # ECM has 4 joints not 6.
    assert s.measured_js.position is not None and len(s.measured_js.position) == 4
    # ECM doesn't carry measured_frequency.
    assert s.source_frequency_hz is None


def test_inverse_preserves_core_fields():
    """Round-trip a synthetic PSM-shape sample through the inverse and
    re-load — measured_cp / setpoint_cp / jaw / freq must survive
    intact."""
    original = KinematicSample(
        frame=42,
        arm_name="PSM1",
        measured_js=JointSnapshot(
            position=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            velocity=[0.1] * 6,
            effort=[10.0] * 6,
        ),
        setpoint_js=JointSnapshot(
            position=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        ),
        measured_cp=CartesianSnapshot(
            position=[0.1, 0.2, 0.3],
            orientation=[0.0, 0.0, 0.0, 1.0],
        ),
        setpoint_cp=CartesianSnapshot(
            position=[0.11, 0.21, 0.31],
            orientation=[0.1, 0.0, 0.0, 0.995],
        ),
        measured_jaw_position=0.42,
        setpoint_jaw_position=0.43,
        source_frequency_hz=843.04,
    )

    payload = kinematic_sample_to_raw_dict(original)
    # Round-trip via JSON to mimic disk.
    s = json.dumps(payload)
    revived_payload = json.loads(s)
    # The forward parser expects a list-wrapped {"arm": ...}; verify shape.
    assert isinstance(revived_payload, list) and len(revived_payload) == 1
    assert "arm" in revived_payload[0]

    # Now feed it through the forward parser by writing to a tmp file.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp_name = f.name
    try:
        revived = load_arm_frame_json(Path(tmp_name), "PSM1", 42)
    finally:
        Path(tmp_name).unlink()

    assert revived.measured_js.position == original.measured_js.position
    assert revived.setpoint_js.position == original.setpoint_js.position
    assert revived.measured_cp.position == original.measured_cp.position
    assert revived.measured_cp.orientation == original.measured_cp.orientation
    assert revived.setpoint_cp is not None
    assert revived.setpoint_cp.position == original.setpoint_cp.position
    assert revived.measured_jaw_position == pytest.approx(0.42)
    assert revived.setpoint_jaw_position == pytest.approx(0.43)
    assert revived.source_frequency_hz == pytest.approx(843.04)


def test_inverse_omits_setpoint_cp_when_missing():
    """An offline-shaped sample (no setpoint_cp) round-trips with the
    key absent from the JSON entirely — not present-but-null."""
    s = KinematicSample(
        frame=0, arm_name="PSM1",
        measured_cp=CartesianSnapshot(position=[0.0, 0.0, 0.0], orientation=[0, 0, 0, 1]),
    )
    payload = kinematic_sample_to_raw_dict(s)
    sp = payload[0]["arm"]["setpoint_data"]
    assert "setpoint_cp" not in sp  # absent, not null


def test_inverse_places_jaw_and_frequency_at_top_level():
    """Real on-disk layout puts jaw + measured_frequency as siblings of
    arm, not children. The inverse must mirror this exactly so a
    decompose round-trip matches the original layout."""
    s = KinematicSample(
        frame=0, arm_name="PSM1",
        measured_jaw_position=0.42,
        setpoint_jaw_position=0.43,
        source_frequency_hz=843.04,
    )
    payload = kinematic_sample_to_raw_dict(s)
    top = payload[0]
    assert "jaw" in top
    assert "measured_frequency" in top
    # Sanity: should NOT also be nested inside arm.
    assert "jaw" not in top["arm"]
    assert "measured_frequency" not in top["arm"]
