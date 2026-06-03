from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.validate import validate_episode


SAMPLE = Path("<release_root>/online_data/episodes/single_interrupted_stitch/2")


def _sample_is_current_schema() -> bool:
    """Return True if the on-disk packed sample carries the current
    EpisodeMeta schema AND was finalized with the new sentinel-based
    flow. Tests below skip when the pack predates a breaking change
    (master_t0_ns field, `.surgsync_complete.json` sentinel) — re-running
    `surgsync build` regenerates both."""
    meta = SAMPLE / "episode_meta.json"
    sentinel = SAMPLE / ".surgsync_complete.json"
    if not meta.is_file() or not sentinel.is_file():
        return False
    try:
        return "master_t0_ns" in json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return False


@pytest.mark.skipif(
    not _sample_is_current_schema(),
    reason="packed sample missing or predates current EpisodeMeta schema "
           "(re-run `surgsync build` to regenerate)",
)
def test_packed_episode_validates_clean():
    issues = validate_episode(SAMPLE)
    errors = [i for i in issues if i.severity == "ERROR"]
    assert errors == [], errors


def test_missing_episode_dir(tmp_path: Path):
    """A missing or incomplete episode dir surfaces as `ep_incomplete`
    (no sentinel). The sentinel check fires first, so we don't bother
    inspecting episode_meta.json on a clearly-broken episode."""
    issues = validate_episode(tmp_path / "nope")
    assert any(i.code == "ep_incomplete" for i in issues)
