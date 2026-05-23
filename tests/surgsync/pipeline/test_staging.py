from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.pipeline.staging import (
    COMPLETE_MARKER,
    FAILED_MARKER,
    RUNNING_MARKER,
    episode_final_dir,
    episode_staging,
    finalize_episode,
    is_episode_complete,
    load_completion_manifest,
)


# ---- fixture helpers -------------------------------------------------------

def _open(tmp_path: Path, dataset="online_data", task="suturing", clip="2",
          episode_id="online_data_2_42", clean_existing=False):
    """Shorthand for the typical episode_staging call across tests."""
    return episode_staging(
        tmp_path, dataset, task, clip,
        episode_id=episode_id, clean_existing=clean_existing,
    )


def _finalize(tmp_path: Path, ep: Path, dataset="online_data", task="suturing",
              clip="2", episode_id="online_data_2_42",
              length_frames=947, duration_s=94.6, extra=None):
    return finalize_episode(
        ep, tmp_path, dataset, task, clip,
        episode_id=episode_id,
        length_frames=length_frames, duration_s=duration_s,
        extra=extra,
    )


# ---- tests -----------------------------------------------------------------

def test_final_dir_layout(tmp_path: Path):
    p = episode_final_dir(tmp_path, "online_data", "suturing", "2")
    assert p == tmp_path / "online_data" / "episodes" / "suturing" / "2"


def test_running_marker_appears_on_entry(tmp_path: Path):
    """As soon as we enter the staging context, the running marker
    is stamped — so a concurrent observer can tell the dir is open."""
    with _open(tmp_path) as ep:
        running = ep / RUNNING_MARKER
        assert running.is_file()
        payload = json.loads(running.read_text())
        assert payload["kind"] == "running"
        assert payload["episode_id"] == "online_data_2_42"
        assert "started_at_utc" in payload
        assert "pid" in payload
        # No complete marker yet.
        assert not is_episode_complete(ep)


def test_complete_marker_replaces_running_on_finalize(tmp_path: Path):
    """finalize_episode writes the manifest, removes the running marker.
    Order observable to scanners: running disappears, complete appears."""
    with _open(tmp_path) as ep:
        (ep / "marker.txt").write_text("hello")
    assert (ep / RUNNING_MARKER).is_file()  # still there after context exit (caller hasn't finalized)
    assert not is_episode_complete(ep)

    _finalize(tmp_path, ep, length_frames=947, duration_s=94.6,
              extra={"source_clip": "data/online_data/2/"})
    assert is_episode_complete(ep)
    assert not (ep / RUNNING_MARKER).exists()
    assert not (ep / FAILED_MARKER).exists()

    manifest = load_completion_manifest(ep)
    assert manifest is not None
    assert manifest["kind"] == "complete"
    assert manifest["episode_id"] == "online_data_2_42"
    assert manifest["length_frames"] == 947
    assert manifest["duration_s"] == 94.6
    assert manifest["source_clip"] == "data/online_data/2/"
    # start + complete timestamps both populated.
    assert manifest["started_at_utc"]
    assert manifest["completed_at_utc"]


def test_exception_writes_failed_marker(tmp_path: Path):
    """A crash mid-pack leaves the failed marker behind, with error
    type + message + traceback, and removes the running marker so a
    scanner can't mistake "failed" for "in-flight"."""
    with pytest.raises(RuntimeError, match="synthetic"):
        with _open(tmp_path) as ep:
            (ep / "halfway.txt").write_text("partial")
            raise RuntimeError("synthetic crash")

    assert not is_episode_complete(ep)
    assert not (ep / RUNNING_MARKER).exists()
    failed = ep / FAILED_MARKER
    assert failed.is_file()
    payload = json.loads(failed.read_text())
    assert payload["kind"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert "synthetic crash" in payload["error_message"]
    assert "synthetic crash" in payload["traceback"]
    # Partial artifact still observable for forensic inspection.
    assert (ep / "halfway.txt").exists()


def test_re_pack_clears_prior_failed_marker(tmp_path: Path):
    """A successful re-pack after a previous failure removes the
    failed marker — scanners see a clean completion."""
    # First attempt: fail.
    with pytest.raises(RuntimeError):
        with _open(tmp_path) as ep:
            raise RuntimeError("boom")
    assert (ep / FAILED_MARKER).is_file()

    # Re-pack with clean_existing.
    with _open(tmp_path, clean_existing=True) as ep:
        (ep / "marker.txt").write_text("redo")
    _finalize(tmp_path, ep)
    assert is_episode_complete(ep)
    assert not (ep / FAILED_MARKER).exists()


def test_refuse_completed_without_force(tmp_path: Path):
    """Once complete, we won't reopen the dir for writing — caller
    must explicitly wipe via convert_clip's force=True branch."""
    with _open(tmp_path) as ep:
        pass
    _finalize(tmp_path, ep)
    with pytest.raises(FileExistsError, match="COMPLETE"):
        with _open(tmp_path):
            pass


def test_refuse_incomplete_leftover_without_clean(tmp_path: Path):
    """An incomplete leftover (running or failed marker present, or
    just stray files) needs explicit `clean_existing=True`."""
    with _open(tmp_path):
        pass  # leaves dir with running marker only
    with pytest.raises(FileExistsError, match="Incomplete"):
        with _open(tmp_path):
            pass


def test_finalize_requires_existing_dir(tmp_path: Path):
    """finalize_episode against a non-existent dir is a programming
    error — the caller should have built into the dir first."""
    fake = episode_final_dir(tmp_path, "online_data", "suturing", "999")
    with pytest.raises(FileNotFoundError):
        finalize_episode(
            fake, tmp_path, "online_data", "suturing", "999",
            episode_id="x", length_frames=0, duration_s=0.0,
        )


def test_finalize_atomic_no_stray_temp_files(tmp_path: Path):
    """Sentinel writes use temp+rename — no stray `.tmp` files left."""
    with _open(tmp_path) as ep:
        pass
    _finalize(tmp_path, ep)
    stray = [p for p in ep.iterdir() if p.name.endswith(".tmp")]
    assert stray == [], stray


def test_load_completion_manifest_returns_none_when_missing(tmp_path: Path):
    """Helper returns None on a non-existent / non-complete dir."""
    assert load_completion_manifest(tmp_path / "nope") is None
    with _open(tmp_path) as ep:
        pass
    assert load_completion_manifest(ep) is None  # only running marker, no complete


def test_is_episode_complete_false_when_missing(tmp_path: Path):
    assert not is_episode_complete(tmp_path / "does_not_exist")
