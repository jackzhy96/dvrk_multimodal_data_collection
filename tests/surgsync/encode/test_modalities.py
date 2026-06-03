"""modalities.json — topic enumeration tests.

Builds a tiny per-modality parquet trio (timestamp + ECM + PSM1 + PSM2
+ annotation) and verifies that `collect_modalities` enumerates every
expected topic with the right `{"present": bool, ...}` shape.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dvrk_data_processing.surgsync.encode.modalities import (
    ANNOTATION_TOPIC_COLS,
    ECM_TOPIC_COLS,
    PSM_TOPIC_COLS,
    TIMESTAMP_TOPIC_COLS,
    collect_modalities,
    write_modalities_json,
)
from dvrk_data_processing.surgsync.schema import (
    build_annotation_schema,
    build_ecm_schema,
    build_psm_schema,
    build_timestamp_schema,
)


def _write_empty_parquet(path: Path, schema: pa.Schema, n_rows: int) -> None:
    """Write a parquet with N rows where every nullable column is NULL
    and required columns have placeholder values. Used to exercise the
    "topic not present" branch."""
    arrays = []
    for field in schema:
        if field.nullable:
            arrays.append(pa.nulls(n_rows, type=field.type))
            continue
        # Required columns get safe placeholders.
        if pa.types.is_integer(field.type):
            arrays.append(pa.array([0] * n_rows, type=field.type))
        elif pa.types.is_floating(field.type):
            arrays.append(pa.array([0.0] * n_rows, type=field.type))
        elif pa.types.is_boolean(field.type):
            arrays.append(pa.array([False] * n_rows, type=field.type))
        elif pa.types.is_string(field.type):
            arrays.append(pa.array([""] * n_rows, type=field.type))
        else:
            arrays.append(pa.nulls(n_rows, type=field.type))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def _write_populated_parquet(path: Path, schema: pa.Schema, n_rows: int) -> None:
    """Write a parquet with every column populated with placeholder
    non-NULL data. Float lists are length-3."""
    arrays = []
    for field in schema:
        t = field.type
        if pa.types.is_integer(t):
            arrays.append(pa.array(list(range(n_rows)), type=t))
        elif pa.types.is_floating(t):
            arrays.append(pa.array([float(i) for i in range(n_rows)], type=t))
        elif pa.types.is_boolean(t):
            arrays.append(pa.array([True] * n_rows, type=t))
        elif pa.types.is_string(t):
            arrays.append(pa.array([str(i) for i in range(n_rows)], type=t))
        elif pa.types.is_list(t):
            arrays.append(pa.array([[1.0, 2.0, 3.0]] * n_rows, type=t))
        else:
            arrays.append(pa.nulls(n_rows, type=t))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def test_topics_block_marks_null_columns_not_present(tmp_path: Path):
    """A parquet with every nullable column = NULL should produce
    `present: false` for every topic in the manifest."""
    n = 5
    _write_empty_parquet(tmp_path / "PSM1.parquet", build_psm_schema(), n)
    _write_empty_parquet(tmp_path / "PSM2.parquet", build_psm_schema(), n)
    _write_empty_parquet(tmp_path / "ECM.parquet", build_ecm_schema(), n)
    _write_empty_parquet(tmp_path / "annotation.parquet", build_annotation_schema(), n)
    _write_empty_parquet(tmp_path / "timestamp.parquet", build_timestamp_schema(), n)

    mods = collect_modalities(tmp_path, expected_frames=n, episode_id="test")

    # Every kinematic arm has its topics block, all marked not present.
    for arm in ("ECM", "PSM1", "PSM2"):
        arm_block = mods["kinematic"][arm]
        assert arm_block["present"] is True   # parquet exists
        assert arm_block["topics_present_count"] == 0
        expected_cols = ECM_TOPIC_COLS if arm == "ECM" else PSM_TOPIC_COLS
        assert set(arm_block["topics"].keys()) == set(expected_cols)
        for topic, status in arm_block["topics"].items():
            assert status == {"present": False}, f"{arm}.{topic}"

    # Annotation topics — all not present.
    ann = mods["annotation"]
    assert ann["topics_present_count"] == 0
    assert set(ann["topics"].keys()) == set(ANNOTATION_TOPIC_COLS)

    # Timestamp deltas — all not present.
    ts = mods["timestamp"]
    assert set(ts["topics"].keys()) == set(TIMESTAMP_TOPIC_COLS)


def test_topics_block_marks_populated_columns_present(tmp_path: Path):
    """A parquet with every column populated should produce
    `{"present": true, "populated_frames": N, "coverage_ratio": 1.0}`."""
    n = 4
    _write_populated_parquet(tmp_path / "PSM1.parquet", build_psm_schema(), n)
    _write_populated_parquet(tmp_path / "PSM2.parquet", build_psm_schema(), n)
    _write_populated_parquet(tmp_path / "ECM.parquet", build_ecm_schema(), n)
    _write_populated_parquet(tmp_path / "annotation.parquet", build_annotation_schema(), n)
    _write_populated_parquet(tmp_path / "timestamp.parquet", build_timestamp_schema(), n)

    mods = collect_modalities(tmp_path, expected_frames=n, episode_id="test")

    psm1 = mods["kinematic"]["PSM1"]
    assert psm1["topics_present_count"] == len(PSM_TOPIC_COLS)
    for topic, status in psm1["topics"].items():
        assert status["present"] is True
        assert status["populated_frames"] == n
        assert status["coverage_ratio"] == 1.0


def test_missing_parquet_still_lists_every_topic(tmp_path: Path):
    """When a parquet file is absent, `topics` should still enumerate
    every expected topic, all marked `{"present": false}`."""
    mods = collect_modalities(tmp_path, expected_frames=10, episode_id="test")
    # No parquets exist at all.
    psm1 = mods["kinematic"]["PSM1"]
    assert psm1["present"] is False
    assert set(psm1["topics"].keys()) == set(PSM_TOPIC_COLS)
    for status in psm1["topics"].values():
        assert status == {"present": False}


def test_image_streams_always_enumerated(tmp_path: Path):
    """`video` and `video_raw` enumerate the three canonical cameras
    even when no video file is present."""
    mods = collect_modalities(tmp_path, expected_frames=5, episode_id="test")
    for cat in ("video", "video_raw"):
        block = mods[cat]
        assert set(block.keys()) == {"stereo_left", "stereo_right", "side"}
        for status in block.values():
            assert status == {"present": False}


def test_write_modalities_json_atomic(tmp_path: Path):
    """write_modalities_json writes the file atomically (no .tmp left
    behind on success)."""
    _write_empty_parquet(tmp_path / "PSM1.parquet", build_psm_schema(), 3)
    _write_empty_parquet(tmp_path / "PSM2.parquet", build_psm_schema(), 3)
    _write_empty_parquet(tmp_path / "ECM.parquet", build_ecm_schema(), 3)
    _write_empty_parquet(tmp_path / "annotation.parquet", build_annotation_schema(), 3)
    _write_empty_parquet(tmp_path / "timestamp.parquet", build_timestamp_schema(), 3)
    dst = write_modalities_json(tmp_path, expected_frames=3, episode_id="x")
    assert dst.is_file()
    assert not dst.with_suffix(".json.tmp").exists()
    payload = json.loads(dst.read_text())
    assert payload["episode_id"] == "x"
    assert payload["expected_frames"] == 3
    assert payload["summary"]["kinematic_arms"] == 3   # parquets present
