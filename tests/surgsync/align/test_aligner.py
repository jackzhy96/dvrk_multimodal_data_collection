"""End-to-end align_clip test against real sample data."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from dvrk_data_processing.surgsync.align.aligner import align_clip
from dvrk_data_processing.surgsync.align.policy import TolerancePolicy
from dvrk_data_processing.surgsync.ingest.timestamps import load_timestamps
from dvrk_data_processing.surgsync.ingest.kinematics import load_arm
from dvrk_data_processing.surgsync.ingest.annotations import load_annotations


REPO = Path(__file__).resolve().parents[3]
ONLINE_RAW = REPO / "data" / "online_data" / "2"


def _policy_online() -> TolerancePolicy:
    """The sample online_data/2 was recorded via standard ROS bag
    capture, not a tight real-time loop, so kinematic topic stamps lag
    the camera stamps by 10–30 ms typically. We use 100 ms here so the
    sample data round-trips cleanly. Production-grade real-time
    recordings can use the spec's 2 ms target."""
    return TolerancePolicy.for_variant(
        "online",
        tol_ms_image_right=2.0,
        tol_ms_image_side=33.0,
        tol_ms_kinematic=100.0,
    )


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_align_online_clip():
    ts = load_timestamps(ONLINE_RAW / "time_syn")
    kin_ECM  = load_arm(ONLINE_RAW / "kinematic", "ECM")
    kin_PSM1 = load_arm(ONLINE_RAW / "kinematic", "PSM1")
    kin_PSM2 = load_arm(ONLINE_RAW / "kinematic", "PSM2")
    ann = load_annotations(ONLINE_RAW / "annotation")

    aligned = align_clip(
        ts,
        kin_ECM=kin_ECM,
        kin_PSM1=kin_PSM1,
        kin_PSM2=kin_PSM2,
        annotations=ann,
        policy=_policy_online(),
    )

    assert aligned.n_frames == len(ts.master_ns)
    assert aligned.n_frames == 886
    # Master timestamps strictly monotonic.
    assert (np.diff(aligned.master_ns) > 0).all()
    # The master timeline is rebased so the first row is 0 ns. The
    # original absolute t0 is preserved on `master_t0_ns`; adding it
    # back to every row reproduces the absolute timeline.
    assert int(aligned.master_ns[0]) == 0
    assert aligned.master_t0_ns > 10**18  # since-epoch ns ≈ 1.7e18 today
    # The smallest non-NULL stamp in the raw timeline equals master_t0_ns.
    from dvrk_data_processing.surgsync.ingest.timestamps import NULL_TS
    raw_nonnull = ts.master_ns[ts.master_ns != NULL_TS]
    assert int(aligned.master_t0_ns) == int(raw_nonnull.min())
    # Round-trip: rebased + offset == original sorted absolute timeline.
    absolute = aligned.master_ns.astype(np.int64) + np.int64(aligned.master_t0_ns)
    assert int(absolute[0]) == int(aligned.master_t0_ns)
    assert int(absolute[-1]) > int(aligned.master_t0_ns)
    # PSM1 kinematic samples populated for every frame (online).
    assert all(s is not None for s in aligned.PSM1_aligned)
    # Annotation alignment surfaces gesture-partial frames.
    n_with_annotation = sum(1 for a in aligned.annotations if a is not None)
    assert n_with_annotation == 886
    # Sync stats sanity checks. The sample recording is variable-rate ROS
    # so the contiguity ratio reflects actual recorder jitter, not a
    # uniform 30 Hz expectation.
    assert aligned.sync_stats.episode_length_s > 0.0
    assert 0.0 <= aligned.sync_stats.contiguity_ratio <= 1.0
    # The median kinematic delta on this recording is ~12 ms (real ROS
    # bag latency between camera + PSM topics). Below the 100 ms test
    # policy comfortably.
    assert aligned.sync_stats.median_kin_delta_ms < 50.0
    # Cross-modal aggregate (every synced modality pooled). On real
    # online sample data every modality has present stamps, so all
    # four summary stats are finite floats — None would indicate
    # "zero modalities had any data," which doesn't happen here.
    cm_med = aligned.sync_stats.cross_modal_median_delta_ms
    cm_mean = aligned.sync_stats.cross_modal_mean_delta_ms
    cm_std  = aligned.sync_stats.cross_modal_std_delta_ms
    cm_max  = aligned.sync_stats.cross_modal_max_delta_ms
    for x in (cm_med, cm_mean, cm_std, cm_max):
        assert x is not None and np.isfinite(x)
    # Pool ordering: max ≥ median ≥ 0; std is non-negative; mean
    # bracketed by median and max for a typical right-skewed delta
    # distribution (loose check — we don't pin tight numerics).
    assert cm_max >= cm_med >= 0.0
    assert cm_std >= 0.0
    assert 0.0 <= cm_mean <= cm_max
    # Per-topic latency detail populates every key in
    # out_of_tol_counts so consumers can join the two structures
    # cleanly. The dict lives on AlignedClip and is written to
    # `time_sync_stat.json` by the encoder.
    assert (set(aligned.per_topic_latency.keys())
            == set(aligned.sync_stats.out_of_tol_counts.keys()))
    for topic, entry in aligned.per_topic_latency.items():
        # Every record has the same key set.
        assert set(entry.keys()) == {
            "median_delta_ms", "mean_delta_ms", "std_delta_ms",
            "max_delta_ms", "max_delta_frame_idx", "n_present",
        }
        if entry["n_present"] == 0:
            for k in ("median_delta_ms", "mean_delta_ms", "std_delta_ms",
                      "max_delta_ms", "max_delta_frame_idx"):
                assert entry[k] is None, f"{topic}.{k} should be null for n_present=0"
        else:
            for k in ("median_delta_ms", "mean_delta_ms", "std_delta_ms",
                      "max_delta_ms"):
                assert entry[k] is not None and entry[k] >= 0.0, (topic, k)
            assert entry["max_delta_ms"] >= entry["median_delta_ms"]
            assert entry["std_delta_ms"] >= 0.0
            # max_delta_frame_idx must be a real master-frame index.
            assert 0 <= entry["max_delta_frame_idx"] < aligned.n_frames
    # out_of_tol_counts is purely informational now (no data is dropped
    # based on it). For the sample under the 100 ms policy, the median
    # delta is ~12 ms so most counts should be 0.
    for key in ("PSM1.measured_cp", "PSM1.measured_js", "PSM1.setpoint_js"):
        assert aligned.sync_stats.out_of_tol_counts[key] >= 0
    assert aligned.sync_stats.out_of_tol_counts["PSM1.measured_cp"] == 0
    assert aligned.sync_stats.out_of_tol_counts["PSM1.measured_js"] == 0

    # Critically: delta columns preserve actual values for every row
    # whose source stamp is present. Verify by checking the PSM1
    # measured_cp delta — every entry should be non-NULL on this
    # well-formed sample. `topic_deltas` is dict-keyed by canonical
    # topic name from `align.topics`.
    from dvrk_data_processing.surgsync.align.aligner import NULL_DELTA
    d = aligned.topic_deltas["PSM1.measured_cp"]
    assert (d != NULL_DELTA).all(), "delta_to_master should not NULL based on tolerance"
    # The catalog drives every key — every topic in TIMESTAMP_TOPICS
    # has an array of length n_frames.
    from dvrk_data_processing.surgsync.align.topics import TOPIC_NAMES
    assert set(aligned.topic_deltas.keys()) == set(TOPIC_NAMES)
    for name, arr in aligned.topic_deltas.items():
        assert arr.shape == (aligned.n_frames,), name
