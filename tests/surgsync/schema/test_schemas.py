"""Schema round-trip tests — every Arrow schema instantiates, every
pydantic model round-trips through JSON.

These are intentionally cheap. The goal is to catch typos in column names
and field-type drift, not to validate semantic content (that's the
encoder's job).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from dvrk_data_processing.surgsync.schema import (
    SCHEMA_VERSION,
    build_timestamp_schema,
    build_ecm_schema,
    build_psm_schema,
    build_annotation_schema,
    build_episodes_schema,
    build_index_schema,
    build_stats_schema,
    EpisodeMeta,
    SyncStats,
    DatasetMeta,
    TaskVocab,
    Manifest,
    ManifestFile,
)


# ---------------------------------------------------------------------------
# Arrow schema builders
# ---------------------------------------------------------------------------

def test_schema_version_is_semver_like():
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_timestamp_schema_basics():
    """timestamp.parquet carries the master clock + one
    `delta_to_master.<topic>_ns` column per topic in the catalog."""
    from dvrk_data_processing.surgsync.align.topics import (
        TIMESTAMP_TOPICS, DELTA_COLUMN_NAMES,
    )
    s = build_timestamp_schema()
    names = s.names
    assert "frame_index" in names
    assert "source_frame_index" in names
    assert "master_timestamp_ns" in names
    assert "is_contiguous_to_prev" in names
    # Every topic shows up as a `delta_to_master.<name>_ns` column.
    for col in DELTA_COLUMN_NAMES:
        assert col in names, col
    # Spot-check a couple of canonical topic names.
    assert "delta_to_master.image_right_ns" in names
    assert "delta_to_master.image_side_ns" in names
    assert "delta_to_master.PSM1.measured_cp_ns" in names
    assert "delta_to_master.PSM2.jaw_measured_ns" in names
    assert "delta_to_master.ECM.measured_js_ns" in names


def test_ecm_schema_carries_every_json_field():
    """ECM.parquet captures every block from `kinematic/ECM/<i>.json`.
    Delta-to-master columns no longer live here — they're in
    timestamp.parquet only."""
    s = build_ecm_schema()
    names = set(s.names)
    assert {"frame_index", "master_timestamp_ns"} <= names
    # No per-arm delta columns anymore (they moved to timestamp.parquet).
    assert not any(n.startswith("delta_to_master.") for n in names)
    # local_measured_cp + measured_cp (with twist) + measured_cv
    assert "local_measured_cp.position" in names
    assert "local_measured_cp.orientation" in names
    assert "measured_cp.position" in names
    assert "measured_cp.orientation" in names
    assert "measured_cp.velocity" in names
    assert "measured_cv.linear" in names
    assert "measured_cv.angular" in names
    # measured_js / setpoint_js
    for jp in ("measured_js", "setpoint_js"):
        for f in ("position", "velocity", "effort"):
            assert f"{jp}.{f}" in names
    # No PSM-only fields.
    assert "jaw.measured_position" not in names
    assert "source_frequency_hz" not in names


def test_psm_schema_is_superset_of_ecm():
    """PSM schema = ECM superset (jaw, setpoint_cp, calibrated, freq)
    minus the delta columns, which moved to timestamp.parquet."""
    psm = set(build_psm_schema().names)
    # PSM has every ECM column.
    ecm_kinematic_fields = {
        "local_measured_cp.position", "local_measured_cp.orientation",
        "measured_cp.position", "measured_cp.orientation", "measured_cp.velocity",
        "measured_cv.linear", "measured_cv.angular",
        "measured_js.position", "measured_js.velocity", "measured_js.effort",
        "setpoint_js.position", "setpoint_js.velocity", "setpoint_js.effort",
    }
    assert ecm_kinematic_fields <= psm
    # Plus jaw + measured_frequency + setpoint_cp (online-only)
    assert {"jaw.measured_position", "jaw.setpoint_position",
            "source_frequency_hz",
            "setpoint_cp.position", "setpoint_cp.orientation",
            "measured_cp_calibrated.position",
            "setpoint_cp_calibrated.position"} <= psm
    # No delta_to_master.* columns under the arm schema — they're all
    # in timestamp.parquet now.
    assert not any(n.startswith("delta_to_master.") for n in psm)


def test_annotation_schema_carries_stereo_left_timestamp():
    s = build_annotation_schema()
    names = set(s.names)
    assert "master_timestamp_ns" in names   # aligned to stereo-left
    for col in ("contact.PSM1", "contact.PSM2",
                "gesture.PSM1", "gesture.PSM2",
                "phase", "step"):
        assert col in names


def test_schema_types_spot_check():
    """Spot-check Arrow types."""
    s_ts = build_timestamp_schema()
    assert s_ts.field("master_timestamp_ns").type == pa.int64()
    assert s_ts.field("frame_index").type == pa.int32()

    s_psm = build_psm_schema()
    assert s_psm.field("measured_cp.velocity").type.value_type == pa.float32()
    assert s_psm.field("jaw.measured_position").type == pa.float32()
    assert s_psm.field("source_frequency_hz").type == pa.float32()


def test_episodes_index_schema_has_partition_column():
    s = build_episodes_schema()
    names = s.names
    assert "task" in names
    assert "episode_id" in names


def test_index_schema_subset_of_timestamp_plus_annotation():
    """The cross-episode frame index pulls columns from the per-episode
    timestamp.parquet (master_ts + is_contiguous) and annotation.parquet
    (contact/gesture/phase/step). Verify types stay consistent across
    the union."""
    ts = build_timestamp_schema()
    ann = build_annotation_schema()
    idx = build_index_schema()

    for name in idx.names:
        if name in ("episode_id", "task"):
            continue  # index-only
        if name in ts.names:
            assert idx.field(name).type == ts.field(name).type, name
        elif name in ann.names:
            assert idx.field(name).type == ann.field(name).type, name
        else:
            raise AssertionError(f"index column {name!r} not in timestamp or annotation schema")


def test_stats_schema_basics():
    s = build_stats_schema()
    assert "column_name" in s.names
    assert "min" in s.names and "max" in s.names
    assert "mean" in s.names and "std" in s.names


# ---------------------------------------------------------------------------
# pydantic round-trips
# ---------------------------------------------------------------------------

def _make_sync_stats():
    return SyncStats(
        episode_length_s=31.6,
        median_kin_delta_ms=0.8,
        max_kin_delta_ms=2.4,
        cross_modal_median_delta_ms=1.1,
        cross_modal_mean_delta_ms=1.4,
        cross_modal_std_delta_ms=0.6,
        cross_modal_max_delta_ms=12.3,
        frames_dropped=7,
        contiguity_ratio=0.993,
        out_of_tol_counts={"PSM1.measured_cp": 3},
    )


def _make_episode_meta():
    return EpisodeMeta(
        schema_version=SCHEMA_VERSION,
        episode_id="8b3d4e2a" * 4,
        task="suturing",
        length_frames=947,
        duration_s=31.6,
        # Realistic ~now since-epoch ns for the fixture; the field is
        # required so callers know how to recover absolute time.
        master_t0_ns=1_700_000_000_000_000_000,
        recorder_variant="offline",
        sync_policy="nearest_interp",
        source_clip="data/offline_data/3/",
        operator_skill_level="Intermediate",
        case_type="Ex-vivo",
        image_size=[512, 288],
        sync_stats=_make_sync_stats(),
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def test_episode_meta_round_trip():
    em = _make_episode_meta()
    serialized = em.model_dump_json()
    revived = EpisodeMeta.model_validate_json(serialized)
    assert revived == em


def test_episode_meta_rejects_unknown_fields():
    with pytest.raises(Exception):
        EpisodeMeta.model_validate({
            **_make_episode_meta().model_dump(),
            "no_such_field": True,
        })


def test_dataset_meta_round_trip():
    dm = DatasetMeta(
        schema_version=SCHEMA_VERSION,
        data_version="2026-05-21",
        release_option="A",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=["suturing", "peg_transfer"],
    )
    j = dm.model_dump_json()
    DatasetMeta.model_validate_json(j)


def test_task_vocab_strict_string_keys():
    """TaskVocab is the row schema for the auto-generated
    meta/tasks.jsonl — each row projects one entry of
    workflow_description.json:_task_routing."""
    tv = TaskVocab(
        task="single_interrupted_stitch",
        phase_id="1",
        phase_description="perform a simple interrupted stitch on ex-vivo tissue",
        step_vocab={"00": "bilateral pause", "11": "Needle Handling and Positioning"},
        gesture_vocab={"1": "Reach for needle", "2": "Grasp needle"},
    )
    # JSON round-trip preserves string keys, not silently int-coerced.
    payload = json.loads(tv.model_dump_json())
    assert payload["task"] == "single_interrupted_stitch"
    assert payload["phase_id"] == "1"
    assert all(isinstance(k, str) for k in payload["step_vocab"])
    assert all(isinstance(k, str) for k in payload["gesture_vocab"])


def test_task_vocab_auto_generation_from_workflow_json():
    """The build's auto-generator (`serde.workflow_text.task_vocab_rows`)
    must produce rows that round-trip through TaskVocab without
    error — and no gesture text may carry the legacy `Gx —` prefix.
    """
    import re
    from dvrk_data_processing.surgsync.serde.workflow_text import task_vocab_rows
    rows = task_vocab_rows()
    assert len(rows) > 0, "workflow_description.json has no _task_routing entries"
    g_prefix = re.compile(r"^G\d+\s*[—\-]")
    for r in rows:
        tv = TaskVocab.model_validate(r)
        assert tv.task and tv.phase_id and tv.phase_description
        for gid, gtext in tv.gesture_vocab.items():
            assert not g_prefix.match(gtext), \
                f"{tv.task}::gesture[{gid}] still carries Gx— prefix: {gtext!r}"


def test_manifest_round_trip():
    m = Manifest(
        schema_version=SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        data_version="2026-05-21",
        files={
            "meta/dataset.json": ManifestFile(sha256="abc", size_bytes=1234),
        },
        total_files=1,
        total_size_bytes=1234,
    )
    j = m.model_dump_json()
    m2 = Manifest.model_validate_json(j)
    assert m2 == m
