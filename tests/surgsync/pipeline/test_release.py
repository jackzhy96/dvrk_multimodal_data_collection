"""Tests for the release-docs generator."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.pipeline.release import (
    bump_version, run_release, write_release_docs, _parse_semver,
)


# ---------------------------------------------------------------------------
# Version parsing + bump
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,expected", [
    ("1.2.3", (1, 2, 3)),
    ("1.2",   (1, 2, 0)),    # M.m form (the packer default)
    ("7",     (7, 0, 0)),    # bare-major also tolerated
    (" 0.0.0 ", (0, 0, 0)),  # leading/trailing whitespace
])
def test_parse_semver_tolerant(inp, expected):
    assert _parse_semver(inp) == expected


@pytest.mark.parametrize("inp", ["abc", "1.x", "1.2.3.4", "", "1.2.3-rc1"])
def test_parse_semver_rejects_garbage(inp):
    with pytest.raises(ValueError):
        _parse_semver(inp)


@pytest.mark.parametrize("current,kind,expected", [
    ("1.0.0", "patch", "1.0.1"),
    ("1.0.0", "minor", "1.1.0"),
    ("1.0.0", "major", "2.0.0"),
    # Tolerant input — bumping a M.m string lands on the proper M.m.p.
    ("1.0",   "patch", "1.0.1"),
    ("1.2",   "minor", "1.3.0"),
    ("3",     "major", "4.0.0"),
])
def test_bump_version(current, kind, expected):
    assert bump_version(current, kind) == expected


def test_bump_version_unknown_kind():
    with pytest.raises(ValueError):
        bump_version("1.0.0", "foo")


# ---------------------------------------------------------------------------
# Minimal synthetic dataset for the doc generator
# ---------------------------------------------------------------------------

def _make_fake_dataset(root: Path) -> None:
    """Build a tiny `meta/dataset.json` + one finalized episode dir
    under the standard `<dataset>/episodes/<task>/<clip_idx>/` layout.
    No parquets — the release generator only reads JSON.
    """
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "dataset.json").write_text(json.dumps({
        "name": "FakeSync",
        "schema_version": "1.0.0",
        "data_version":   "1.0",
        "release_option": "B",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "modalities": {
            "video":      ["stereo_left", "stereo_right"],
            "preprocess": ["depth"],
            "state":      ["ECM", "PSM1", "PSM2"],
            "action":     ["PSM1", "PSM2"],
            "annotation": ["phase", "step"],
        },
        "conventions": {
            "master_clock": "stereo_left_capture_ros_header_stamp",
            "alignment_policy": "nearest_neighbor_within_tolerance",
            "quaternion_order": "xyzw",
            "length_unit": "m",
            "angle_unit":  "rad",
            "image_size":  [512, 288],
        },
        "tasks": ["fake_task_A", "fake_task_B"],
    }, indent=2))

    for task in ("fake_task_A", "fake_task_B"):
        ep_dir = root / "fake_ds" / "episodes" / task / "1"
        ep_dir.mkdir(parents=True)
        (ep_dir / ".surgsync_complete.json").write_text("{}")
        (ep_dir / "episode_meta.json").write_text(json.dumps({
            "schema_version": "1.0.0",
            "episode_id": f"fake_ds_1_0_{task}",
            "task": task,
            "length_frames": 100,
            "master_t0_ns": 0,
            "recorder_variant": "online",
            "source_clip": "data/fake_ds/1/",
        }))


def test_run_release_writes_readme_and_changelog(tmp_path: Path):
    root = tmp_path / "ds"
    _make_fake_dataset(root)

    summary = run_release(root, bump=None, notes="Hello world.")
    readme = (root / "README.md").read_text()
    changelog = (root / "CHANGELOG.md").read_text()

    # README carries the right summary numbers.
    assert summary["n_episodes"] == 2
    assert summary["n_tasks"] == 2
    assert summary["total_frames"] == 200
    assert "FakeSync" in readme
    assert "v1.0" in readme
    # Both tasks listed.
    assert "fake_task_A" in readme
    assert "fake_task_B" in readme
    # Quick-start code block present.
    assert "surgsync.open_dataset" in readme

    # CHANGELOG seeded with header + our entry.
    assert changelog.startswith("# Changelog")
    assert "Hello world." in changelog
    assert "## v1.0 — " in changelog


def test_run_release_bumps_data_version(tmp_path: Path):
    root = tmp_path / "ds"
    _make_fake_dataset(root)

    summary = run_release(root, bump="minor", notes="Bump test.")
    assert summary["bumped_from"] == "1.0"
    assert summary["bumped_to"] == "1.1.0"

    # dataset.json was updated in place.
    meta = json.loads((root / "meta" / "dataset.json").read_text())
    assert meta["data_version"] == "1.1.0"
    # CHANGELOG has the new version at the head.
    cl = (root / "CHANGELOG.md").read_text()
    assert cl.split("## ")[1].startswith("v1.1.0")


def test_run_release_appends_changelog_on_repeat(tmp_path: Path):
    root = tmp_path / "ds"
    _make_fake_dataset(root)
    run_release(root, bump=None, notes="First release.")
    run_release(root, bump="patch", notes="Bug fix.")

    cl = (root / "CHANGELOG.md").read_text()
    # Both entries present; newer at the top.
    entries = cl.split("## v")
    # First chunk is the header; the rest are entries.
    versions = [e.split(" — ")[0] for e in entries[1:]]
    assert versions == ["1.0.1", "1.0"]


def test_run_release_missing_meta_raises(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        run_release(root)


def test_run_release_omits_changelog_notes_leaves_todo(tmp_path: Path):
    """A run with `notes=None` should leave a TODO placeholder visible
    to operators so they fill it in on the next diff."""
    root = tmp_path / "ds"
    _make_fake_dataset(root)
    run_release(root, notes=None)
    cl = (root / "CHANGELOG.md").read_text()
    assert "TODO" in cl
