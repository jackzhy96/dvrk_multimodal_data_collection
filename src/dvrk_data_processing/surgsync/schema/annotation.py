"""pyarrow.Schema for `<episode>/annotation.parquet`.

One row per master-clock frame (stereo-left timestamp = master_timestamp_ns).
Annotation rows are aligned by source frame index — the operator's
per-frame labels at the stereo-left capture time.

Annotation ids stay as strings end-to-end (matches the GUI's output).
Verbalization happens at consumer time via `meta/tasks.jsonl`.
"""
from __future__ import annotations
import pyarrow as pa


def build_annotation_schema() -> pa.Schema:
    return pa.schema([
        # Identity & sync — `master_timestamp_ns` is the stereo-left
        # capture timestamp this annotation row corresponds to.
        pa.field("frame_index",          pa.int32(),  nullable=False),
        pa.field("master_timestamp_ns",  pa.int64(),  nullable=False),
        # Per-arm contact flags from the capacitive sensor + annotator.
        pa.field("contact.PSM1",         pa.int8(),   nullable=True),
        pa.field("contact.PSM2",         pa.int8(),   nullable=True),
        # Per-arm gesture id (string, may be NULL where the per-frame
        # gesture file is missing — see "Frame-count divergence" in the
        # raw spec).
        pa.field("gesture.PSM1",         pa.string(), nullable=True),
        pa.field("gesture.PSM2",         pa.string(), nullable=True),
        # Per-frame phase + step ids.
        pa.field("phase",                pa.string(), nullable=True),
        pa.field("step",                 pa.string(), nullable=True),
    ])
