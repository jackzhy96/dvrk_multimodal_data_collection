"""`time_sync_stat.json` encoder — per-topic latency detail.

Sibling file to episode_meta.json; this test exercises both the
in-memory `AlignedClip.per_topic_latency` payload produced by the
aligner and the on-disk JSON written by `write_time_sync_stat`.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.align.aligner import align_clip
from dvrk_data_processing.surgsync.align.policy import TolerancePolicy
from dvrk_data_processing.surgsync.encode.time_sync_stat import write_time_sync_stat
from dvrk_data_processing.surgsync.ingest.annotations import load_annotations
from dvrk_data_processing.surgsync.ingest.kinematics import load_arm
from dvrk_data_processing.surgsync.ingest.timestamps import load_timestamps


REPO = Path(__file__).resolve().parents[3]
ONLINE_RAW = REPO / "data" / "online_data" / "2"


def _policy() -> TolerancePolicy:
    return TolerancePolicy.for_variant(
        "online",
        tol_ms_image_right=2.0, tol_ms_image_side=33.0, tol_ms_kinematic=100.0,
    )


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_time_sync_stat_round_trip(tmp_path: Path):
    ts = load_timestamps(ONLINE_RAW / "time_syn")
    kin_ECM  = load_arm(ONLINE_RAW / "kinematic", "ECM")
    kin_PSM1 = load_arm(ONLINE_RAW / "kinematic", "PSM1")
    kin_PSM2 = load_arm(ONLINE_RAW / "kinematic", "PSM2")
    ann = load_annotations(ONLINE_RAW / "annotation")
    aligned = align_clip(
        ts, kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann, policy=_policy(),
    )

    dst = tmp_path / "time_sync_stat.json"
    doc = write_time_sync_stat(
        aligned=aligned,
        episode_id="test_episode_0",
        dst_path=dst,
    )

    # File exists, atomically written (no .tmp leftover).
    assert dst.exists()
    assert not list(tmp_path.glob("*.tmp"))

    with open(dst) as f:
        payload = json.load(f)

    # Schema version + episode_id round-trip.
    assert payload["schema_version"] == doc.schema_version
    assert payload["episode_id"] == "test_episode_0"

    # `per_topic` covers every modality the aligner tracks
    # (out_of_tol_counts is the canonical set).
    expected_topics = set(aligned.sync_stats.out_of_tol_counts.keys())
    assert set(payload["per_topic"].keys()) == expected_topics

    # Field shape per topic.
    expected_fields = {
        "median_delta_ms", "mean_delta_ms", "std_delta_ms",
        "max_delta_ms", "max_delta_frame_idx", "n_present",
    }
    for topic, entry in payload["per_topic"].items():
        assert set(entry.keys()) == expected_fields, topic

    # Online sample has every modality populated — every topic
    # should have non-null stats and a valid max_delta_frame_idx.
    for topic, entry in payload["per_topic"].items():
        assert entry["n_present"] > 0, topic
        for k in ("median_delta_ms", "mean_delta_ms",
                  "std_delta_ms", "max_delta_ms"):
            assert entry[k] is not None and entry[k] >= 0.0, (topic, k)
        # Invariants: max ≥ median ≥ 0; std ≥ 0; mean inside [0, max].
        assert entry["max_delta_ms"] >= entry["median_delta_ms"]
        assert entry["std_delta_ms"] >= 0.0
        assert 0.0 <= entry["mean_delta_ms"] <= entry["max_delta_ms"]
        # The max_delta_frame_idx is a valid master-frame index.
        assert isinstance(entry["max_delta_frame_idx"], int)
        assert 0 <= entry["max_delta_frame_idx"] < aligned.n_frames


@pytest.mark.skipif(not (REPO / "data" / "offline_data" / "3").exists(),
                    reason="offline sample not present")
def test_offline_setpoint_cp_topics_are_null(tmp_path: Path):
    """Offline recorder has no `setpoint_cp_stamp` for either PSM,
    so the `PSM{1,2}.setpoint_cp` topics in `time_sync_stat.json`
    must surface as null stats with `n_present == 0` — not zero
    deltas. This pins the null-not-zero contract for the absent-
    modality case.
    """
    OFFLINE = REPO / "data" / "offline_data" / "3"
    ts = load_timestamps(OFFLINE / "time_syn")
    kin_ECM  = load_arm(OFFLINE / "kinematic", "ECM")
    kin_PSM1 = load_arm(OFFLINE / "kinematic", "PSM1")
    kin_PSM2 = load_arm(OFFLINE / "kinematic", "PSM2")
    ann = load_annotations(OFFLINE / "annotation")
    aligned = align_clip(
        ts, kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann,
        policy=TolerancePolicy.for_variant(
            "offline",
            tol_ms_image_right=2.0, tol_ms_image_side=33.0, tol_ms_kinematic=100.0,
        ),
    )

    dst = tmp_path / "time_sync_stat.json"
    write_time_sync_stat(
        aligned=aligned,
        episode_id="test_offline_setpoint_null",
        dst_path=dst,
    )
    payload = json.loads(dst.read_text())

    for topic in ("PSM1.setpoint_cp", "PSM2.setpoint_cp"):
        entry = payload["per_topic"][topic]
        assert entry["n_present"] == 0, topic
        for k in ("median_delta_ms", "mean_delta_ms",
                  "std_delta_ms", "max_delta_ms",
                  "max_delta_frame_idx"):
            assert entry[k] is None, (topic, k)
