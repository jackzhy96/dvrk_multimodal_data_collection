from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.align.aligner import align_clip
from dvrk_data_processing.surgsync.align.policy import TolerancePolicy
from dvrk_data_processing.surgsync.encode.episode_meta import (
    compute_episode_id, write_episode_meta,
)
from dvrk_data_processing.surgsync.ingest.annotations import load_annotations
from dvrk_data_processing.surgsync.ingest.kinematics import load_arm
from dvrk_data_processing.surgsync.ingest.meta import load_meta
from dvrk_data_processing.surgsync.ingest.timestamps import load_timestamps


REPO = Path(__file__).resolve().parents[3]
ONLINE_RAW = REPO / "data" / "online_data" / "2"


def test_episode_id_deterministic():
    a = compute_episode_id("online_data", "2", 1000)
    b = compute_episode_id("online_data", "2", 1000)
    assert a == b
    # Time-based composite — human-readable, embeds the master_t0_ns.
    assert a == "online_data_2_1000"


def test_episode_id_changes_with_inputs():
    a = compute_episode_id("online_data",  "2", 1000)
    b = compute_episode_id("online_data",  "2", 2000)
    c = compute_episode_id("online_data",  "3", 1000)
    d = compute_episode_id("offline_data", "2", 1000)
    assert len({a, b, c, d}) == 4


def _policy():
    return TolerancePolicy.for_variant(
        "online",
        tol_ms_image_right=2.0, tol_ms_image_side=33.0, tol_ms_kinematic=100.0,
    )


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_write_episode_meta_round_trip(tmp_path: Path):
    ts = load_timestamps(ONLINE_RAW / "time_syn")
    kin_ECM  = load_arm(ONLINE_RAW / "kinematic", "ECM")
    kin_PSM1 = load_arm(ONLINE_RAW / "kinematic", "PSM1")
    kin_PSM2 = load_arm(ONLINE_RAW / "kinematic", "PSM2")
    ann = load_annotations(ONLINE_RAW / "annotation")
    cm = load_meta(ONLINE_RAW / "meta_data.json")
    aligned = align_clip(
        ts, kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann, policy=_policy(),
    )

    dst = tmp_path / "episode_meta.json"
    em = write_episode_meta(
        clip_meta=cm,
        aligned=aligned,
        task="suturing",
        source_clip="data/online_data/2/",
        recorder_variant="online",
        dataset_name="online_data",
        clip_index="2",
        image_size=[512, 288],
        has_preprocess=True,
        has_preview=False,
        has_video_raw=True,
        dst_path=dst,
    )
    assert dst.exists()
    with open(dst) as f:
        payload = json.load(f)
    assert payload["task"] == "suturing"
    assert payload["recorder_variant"] == "online"
    assert payload["length_frames"] == aligned.n_frames
    assert payload["schema_version"] == em.schema_version
    assert payload["has_preprocess"] is True
    assert payload["has_video_raw"] is True
    assert payload["tool"]["PSM1"] == "Large_Needle_Driver"
    # `master_t0_ns` is the absolute ns timestamp of frame 0 — large
    # (since-epoch) and matches the value embedded in episode_id.
    # `aligned.master_ns` is rebased so its first row is 0.
    assert payload["master_t0_ns"] == aligned.master_t0_ns
    assert int(payload["master_t0_ns"]) > 10**18  # since-epoch ns is ~1.7e18 today
    assert int(aligned.master_ns[0]) == 0
    # episode_id encodes the absolute t0 so two clips that both start
    # at relative 0 still get distinct ids.
    assert payload["episode_id"].endswith(f"_{payload['master_t0_ns']}")
    # Cross-modal latency lands in episode_meta.json under sync_stats,
    # with median + mean + std + max all computed from the same pool.
    sync_stats = payload["sync_stats"]
    for k in ("cross_modal_median_delta_ms", "cross_modal_mean_delta_ms",
              "cross_modal_std_delta_ms", "cross_modal_max_delta_ms"):
        assert k in sync_stats
        assert sync_stats[k] is not None  # real online sample → populated
    assert sync_stats["cross_modal_max_delta_ms"] >= sync_stats["cross_modal_median_delta_ms"]
    assert sync_stats["cross_modal_std_delta_ms"] >= 0.0
    assert 0.0 <= sync_stats["cross_modal_mean_delta_ms"] <= sync_stats["cross_modal_max_delta_ms"]
    # Episode length replaces the old `expected_frame_period_ns`.
    assert "expected_frame_period_ns" not in sync_stats  # removed
    assert sync_stats["episode_length_s"] > 0.0
    assert abs(sync_stats["episode_length_s"] - payload["duration_s"]) < 1e-9
    # Per-modality detail is no longer in episode_meta.json — it moved
    # to the sibling `time_sync_stat.json` so episode_meta stays light.
    assert "per_modality_median_delta_ms" not in sync_stats
    assert "per_modality_max_delta_ms" not in sync_stats
