"""End-to-end test: pack a tiny dataset, then run all four index builders."""
from __future__ import annotations
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dvrk_data_processing.surgsync.index import (
    build_episodes_index, build_frames_index, build_stats, build_manifest,
)


SAMPLE_ROOT = Path("/media/jackzhy/Extreme SSD/surgsync_release")


def _sample_is_current_schema() -> bool:
    """Skip when the on-disk pack predates a breaking change. Two
    things must be true: (a) EpisodeMeta carries the current
    `master_t0_ns` field, and (b) each episode dir carries the
    sentinel file from the new finalize path. Otherwise the
    sentinel-checking index scan would silently drop every episode
    and the assertions below would fire."""
    if not SAMPLE_ROOT.is_dir():
        return False
    for ep in SAMPLE_ROOT.rglob("episode_meta.json"):
        try:
            has_t0 = "master_t0_ns" in json.loads(ep.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        has_sentinel = (ep.parent / ".surgsync_complete.json").is_file()
        return has_t0 and has_sentinel
    return False


@pytest.mark.skipif(
    not _sample_is_current_schema(),
    reason="packed sample missing or predates current EpisodeMeta schema "
           "(re-run `surgsync build` to regenerate)",
)
def test_indexes_against_real_pack(tmp_path: Path):
    """Run all four index builders against the on-disk packed dataset.
    Validates that everything wires up without modifying the dataset.
    """
    # Use a sandbox to avoid touching the real meta/ folder.
    # (Symlink the per-episode dirs into a tmp dataset_root.)
    import os, shutil
    sandbox = tmp_path / "ds"
    sandbox.mkdir()
    for dataset_dir in SAMPLE_ROOT.iterdir():
        if dataset_dir.name.startswith(".") or dataset_dir.name == "meta":
            continue
        os.symlink(dataset_dir, sandbox / dataset_dir.name)

    ep = build_episodes_index(sandbox)
    fr = build_frames_index(sandbox)
    st = build_stats(sandbox)
    mn = build_manifest(sandbox, data_version="test")

    assert ep["n_episodes"] >= 1
    assert fr["n_frames"] >= 1
    assert st["n_columns"] >= 1
    assert mn["n_files"] >= 1

    # Spot-check the parquets exist and the manifest round-trips.
    assert (sandbox / "meta" / "episodes.parquet").is_dir()
    assert (sandbox / "meta" / "episodes.jsonl").is_file()
    assert (sandbox / "meta" / "index.parquet").is_dir()
    assert (sandbox / "meta" / "stats.parquet").is_file()
    assert (sandbox / "meta" / "manifest.json").is_file()

    eps_files = list((sandbox / "meta" / "episodes.parquet").rglob("*.parquet"))
    assert eps_files, "no episodes parquet part files written"
    # ParquetFile.read() bypasses Hive auto-inference — the file already
    # carries the `task` column explicitly (per spec § 2.2 "redundant
    # with directory; included for portability"), so pyarrow's
    # auto-detection of `task=<name>/` in the path would clash with the
    # stored column.
    eps_tab = pq.ParquetFile(eps_files[0]).read()
    assert "episode_id" in eps_tab.column_names
