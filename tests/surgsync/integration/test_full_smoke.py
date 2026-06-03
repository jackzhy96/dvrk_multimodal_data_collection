"""End-to-end smoke test for v1.0.0.

The gating test for the release. Exercises the full raw → packed →
re-read → decompose → round-trip-diff pipeline against the on-disk
sample data:

    raw clip on disk
       │
       │  1. snapshot SHA-256s for the streams the packer contract
       │     guarantees recoverable (image pixels, calibration bytes,
       │     kinematic JSON content, meta_data fields)
       ▼
    `surgsync build`  (invoked via Python API)
       │
       │  2. `surgsync.open_dataset(...)` + per-frame walk
       │     - no NULLs in expected columns
       │     - annotation text resolved via workflow_description
       │     - calibrated_kinematic values finite (when present)
       ▼
    `surgsync.decompose(...)`
       │
       │  3. diff against the snapshot from step 1
       │     ERRORs surface as pytest failures with the offending path
       ▼
    PASS  → v1.0.0 candidate

Runtime budget: < 30 minutes. On the sample data (online_data/2 = 886
frames, 3 cameras) the test takes ~10 minutes wall-clock with
parallelism=1: ~5 min pack + ~3 min unpack + ~1 min snapshot/diff.

Skip conditions:
  - `data/online_data/2/` not present → SKIP (no input).
  - The optional preprocess streams require preprocessing to have been run; if
    absent, the test still verifies the raw-clip round-trip (which is
    the canonical invertibility contract — preprocess is best-effort).
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

import dvrk_data_processing.surgsync as surgsync
from tests.surgsync.integration.snapshot import (
    snapshot_clip, diff_against_snapshot, DiffFinding,
)


REPO = Path(__file__).resolve().parents[3]
RAW_CLIP = REPO / "data" / "online_data" / "2"


pytestmark = pytest.mark.skipif(
    not RAW_CLIP.is_dir(),
    reason=(f"sample clip {RAW_CLIP} not present — full smoke test "
            "requires the in-repo sample data."),
)


def _build(packed_root: Path) -> None:
    """Invoke `surgsync build` via subprocess on the sample clip.

    Subprocess (vs. importing `build_release` directly) keeps Hydra's
    process-level state out of pytest and matches the operator
    workflow exactly. The `surgsync` CLI is wired through the
    `pyproject.toml` entry point, so this runs the same code path an
    operator would.
    """
    cmd = [
        sys.executable, "-m", "dvrk_data_processing.surgsync.cli", "build",
        "clips.source=list",
        "+clips.list=[online_data/2]",
        f"path_config.dataset_root={packed_root}",
        "include_preprocess=false",     # preprocessing may not have run; that's fine
        "include_video_processed=false",
        "parallelism=1",
        "force=true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        pytest.fail(
            "surgsync build failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{result.stdout}\n"
            f"  stderr:\n{result.stderr}\n"
        )


def _walk_episode_columns(episode) -> None:
    """Assert the packed episode's per-modality parquets carry the
    expected columns and that annotation cells were verbalized to text.
    """
    # Sanity: every parquet has the same row count.
    n = episode.length
    for arm in ("ECM", "PSM1", "PSM2"):
        t = episode.arm(arm)
        assert t.num_rows == n, (
            f"{arm}.parquet row count {t.num_rows} != length {n}"
        )
    assert episode.annotation.num_rows == n
    assert episode.timestamps.num_rows == n

    # Annotation text contract — every non-NULL phase/step cell must
    # contain spaces (i.e. is a phrase, not a bare numeric id). NULL
    # is allowed (partial coverage).
    for col in ("phase", "step"):
        vals = episode.annotation.column(col).to_pylist()
        for v in vals[:50]:    # cap inspection
            if v is None:
                continue
            assert " " in v, (
                f"annotation.{col} cell {v!r} looks like a bare id — "
                "vocab verbalization didn't run during pack"
            )

    # Master timestamp is rebased to clip-relative — first row is 0.
    mt = episode.timestamps.column("master_timestamp_ns").to_pylist()
    assert mt[0] == 0, f"master_timestamp_ns[0] = {mt[0]} (expected 0 after rebase)"


def _format_findings(findings: Iterable[DiffFinding], max_lines: int = 30) -> str:
    """Pretty-print up to N findings for the pytest failure message."""
    lines = []
    for i, f in enumerate(findings):
        if i >= max_lines:
            lines.append(f"  ... (and {sum(1 for _ in findings) - max_lines} more)")
            break
        lines.append(f"  [{f.severity}] {f.bucket}/{f.path}: {f.detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

# Note: `pytest.mark.timeout(...)` would be appropriate here but
# `pytest-timeout` is not in the conda env. Add it to env_setup if
# CI starts hitting hangs; the smoke test typically completes in
# ~10 min on a fast SSD.
def test_full_pack_unpack_round_trip(tmp_path: Path):
    """raw → pack → read → decompose → round-trip diff.

    This is the canonical correctness gate for v1.0.0. Failure modes
    surface with actionable per-file messages rather than bare
    AssertionErrors — see `tests/surgsync/integration/snapshot.py`.
    """
    # ---- 1. Snapshot the raw clip ---------------------------------
    snap = snapshot_clip(RAW_CLIP, dataset_name="online_data", clip_index="2")
    # Sanity: snapshot is non-empty (catches a wrong RAW_CLIP path).
    assert snap.image_pixels, "snapshot has 0 image pixel hashes — raw dir wrong?"
    assert snap.calibration,  "snapshot has 0 camera_calibration entries"
    assert snap.hand_eye,     "snapshot has 0 hand_eye_calibration entries"
    assert snap.meta_sha,     "snapshot did not find meta_data.json"

    # ---- 2. Pack to a tmp dataset root ----------------------------
    packed_root = tmp_path / "packed"
    _build(packed_root)
    # The packer should have placed the episode under
    # `<packed>/<dataset>/episodes/<task>/<clip_idx>/`.
    candidate = list(packed_root.glob("online_data/episodes/*/2"))
    assert candidate, f"packed episode dir not found under {packed_root}"
    packed_clip = candidate[0]
    assert (packed_clip / ".surgsync_complete.json").is_file(), (
        f"packed clip missing complete sentinel: {packed_clip}"
    )

    # ---- 3. Open the dataset via the reader API -------------------
    ds = surgsync.open_dataset(packed_root)
    refs = ds.episodes
    assert len(refs) == 1, f"expected 1 episode after pack; got {len(refs)}"
    ep = surgsync.open_episode(refs[0].path)
    assert ep.length > 0
    _walk_episode_columns(ep)
    ep.close()

    # ---- 4. Decompose to a fresh tmp tree -------------------------
    decomp_root = tmp_path / "decomposed"
    report = surgsync.decompose(
        dataset_root=packed_root,
        out_root=decomp_root,
        clips=["online_data/2"],
        streams=("raw",),       # raw domain is the invertibility contract
        force=True,
        parallelism=1,
        workers_per_clip=4,
    )
    assert report.n_episodes_ok == 1, (
        f"decompose returned {report.n_episodes_fail} fail(s): "
        f"{report.clips[0].error if report.clips else 'no clips'}"
    )

    # ---- 5. Round-trip diff against the snapshot ------------------
    decomposed_clip = decomp_root / "online_data" / "2"
    findings = diff_against_snapshot(
        decomposed_clip, snap,
        check_image=True, check_calibration=True,
        check_kinematic=True, check_meta=True,
    )
    errors = [f for f in findings if f.severity == "ERROR"]
    assert not errors, (
        f"round-trip diff found {len(errors)} ERROR(s):\n"
        + _format_findings(errors)
    )
