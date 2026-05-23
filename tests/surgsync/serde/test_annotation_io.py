"""Round-trip tests for annotation serde against real sample data."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.serde.annotation_io import (
    AnnotationSample,
    load_annotation_frame,
    annotation_sample_to_files,
)


REPO = Path(__file__).resolve().parents[3]
ONLINE = REPO / "data" / "online_data" / "2" / "annotation"


@pytest.mark.skipif(not ONLINE.exists(), reason="sample data not present")
def test_load_frame_zero_online():
    s = load_annotation_frame(
        contact_path=ONLINE / "contact_detection" / "0.json",
        gesture_path=ONLINE / "gesture" / "0.json",
        phase_path=ONLINE / "phase" / "0.json",
        step_path=ONLINE / "step" / "0.json",
        frame_idx=0,
    )
    assert s.frame == 0
    # online_data/2 frame 0 has no gesture file (partial coverage)
    assert s.gesture_PSM1 is None and s.gesture_PSM2 is None
    # phase/step always strings, never coerced to int
    assert isinstance(s.phase, str)
    assert isinstance(s.step, str)
    assert s.contact_PSM1 in (0, 1)


def test_partial_gesture_load_tolerates_missing_file(tmp_path: Path):
    """If gesture/<frame>.json doesn't exist, load returns gesture_*=None
    without raising."""
    contact_dir = tmp_path / "contact_detection"
    phase_dir = tmp_path / "phase"
    step_dir = tmp_path / "step"
    contact_dir.mkdir(); phase_dir.mkdir(); step_dir.mkdir()

    with open(contact_dir / "5.json", "w") as f:
        json.dump({"PSM1": 1, "PSM2": 0}, f)
    with open(phase_dir / "5.json", "w") as f:
        json.dump({"phase": "3"}, f)
    with open(step_dir / "5.json", "w") as f:
        json.dump({"step": "12"}, f)

    s = load_annotation_frame(
        contact_path=contact_dir / "5.json",
        gesture_path=tmp_path / "gesture" / "5.json",   # doesn't exist
        phase_path=phase_dir / "5.json",
        step_path=step_dir / "5.json",
        frame_idx=5,
    )
    assert s.contact_PSM1 == 1 and s.contact_PSM2 == 0
    assert s.gesture_PSM1 is None and s.gesture_PSM2 is None
    assert s.phase == "3" and s.step == "12"


def test_inverse_round_trip(tmp_path: Path):
    """Write a sample via the inverse and re-load — fields must survive."""
    original = AnnotationSample(
        frame=10,
        contact_PSM1=1, contact_PSM2=0,
        gesture_PSM1="17", gesture_PSM2="4",
        phase="2", step="41",
    )
    contact_path = tmp_path / "contact_detection" / "10.json"
    gesture_path = tmp_path / "gesture" / "10.json"
    phase_path   = tmp_path / "phase" / "10.json"
    step_path    = tmp_path / "step" / "10.json"

    annotation_sample_to_files(
        original,
        contact_path=contact_path,
        gesture_path=gesture_path,
        phase_path=phase_path,
        step_path=step_path,
    )

    revived = load_annotation_frame(
        contact_path=contact_path,
        gesture_path=gesture_path,
        phase_path=phase_path,
        step_path=step_path,
        frame_idx=10,
    )
    assert revived.contact_PSM1 == 1
    assert revived.contact_PSM2 == 0
    assert revived.gesture_PSM1 == "17"
    assert revived.gesture_PSM2 == "4"
    assert revived.phase == "2"
    assert revived.step == "41"
