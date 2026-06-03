from __future__ import annotations
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.ingest.clip import (
    discover_clip,
    discover_clips,
    sorted_frames,
    sorted_frame_indices,
)


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"


@pytest.mark.skipif(not (DATA / "online_data" / "2").exists(), reason="sample data not present")
def test_discover_online_clip():
    clip = discover_clip(DATA, "online_data", "2")
    assert clip.recorder_variant == "online"
    assert clip.side_dir_name == "side"
    assert clip.intermediate_present
    assert clip.processed_present["kinematic_reproject"]
    assert clip.processed_present["depth_estimation"]
    assert clip.processed_present["optical_flow"]
    assert clip.source_clip_str == "data/online_data/2/"


@pytest.mark.skipif(not (DATA / "offline_data" / "3").exists(), reason="sample data not present")
def test_discover_offline_clip():
    clip = discover_clip(DATA, "offline_data", "3")
    assert clip.recorder_variant == "offline"
    assert clip.side_dir_name == "side1"


def test_sorted_frames_numeric_order(tmp_path: Path):
    """Frame stems sort numerically, not lexicographically."""
    # 1, 2, 10, 100 — lex sort puts 10 before 2 which is wrong.
    for n in [1, 2, 10, 100, 20]:
        (tmp_path / f"{n}.png").write_bytes(b"")
    out = sorted_frames(tmp_path, suffix=".png")
    assert [int(p.stem) for p in out] == [1, 2, 10, 20, 100]


def test_sorted_frame_indices_skips_non_integer_stems(tmp_path: Path):
    (tmp_path / "0.png").write_bytes(b"")
    (tmp_path / "notice.png").write_bytes(b"")   # ignored
    (tmp_path / "1.png").write_bytes(b"")
    assert sorted_frame_indices(tmp_path, suffix=".png") == [0, 1]


def test_discover_clip_raises_on_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_clip(tmp_path, "nonexistent", "99")


def test_discover_clip_rejects_unknown_dataset_variant(tmp_path: Path):
    (tmp_path / "weird_data" / "0").mkdir(parents=True)
    with pytest.raises(ValueError, match="recorder variant"):
        discover_clip(tmp_path, "weird_data", "0")


@pytest.mark.skipif(not DATA.exists(), reason="data root not present")
def test_discover_clips_sweeps_both_datasets():
    clips = discover_clips(DATA)
    found = {(c.dataset_name, c.clip_index) for c in clips}
    if (DATA / "online_data" / "2").exists():
        assert ("online_data", "2") in found
    if (DATA / "offline_data" / "3").exists():
        assert ("offline_data", "3") in found
