from __future__ import annotations
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.validate import validate_raw_clip


REPO = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not (REPO / "data" / "online_data" / "2").exists(),
                    reason="sample data not present")
def test_online_clip_validates_clean():
    issues = validate_raw_clip(REPO / "data" / "online_data" / "2")
    errors = [i for i in issues if i.severity == "ERROR"]
    assert errors == [], errors


def test_missing_clip_dir(tmp_path: Path):
    issues = validate_raw_clip(tmp_path / "nonexistent")
    assert any(i.code == "raw_clip_missing" for i in issues)


def test_missing_required_subdir(tmp_path: Path):
    # Build a minimal-but-broken raw clip (no image/).
    (tmp_path / "annotation" / "contact_detection").mkdir(parents=True)
    (tmp_path / "annotation" / "phase").mkdir(parents=True)
    (tmp_path / "annotation" / "step").mkdir(parents=True)
    (tmp_path / "camera_calibration").mkdir()
    (tmp_path / "hand_eye_calibration").mkdir()
    (tmp_path / "kinematic" / "ECM").mkdir(parents=True)
    (tmp_path / "kinematic" / "PSM1").mkdir(parents=True)
    (tmp_path / "kinematic" / "PSM2").mkdir(parents=True)
    (tmp_path / "time_syn").mkdir()
    (tmp_path / "meta_data.json").write_text("{}")
    # image/ missing entirely
    issues = validate_raw_clip(tmp_path)
    codes = {i.code for i in issues if i.severity == "ERROR"}
    assert "raw_missing_subdir" in codes
