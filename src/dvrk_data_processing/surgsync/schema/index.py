"""pyarrow.Schema for `meta/index.parquet/task=*/part-*.parquet`.

Frame-level cross-episode view. Deliberately excludes per-arm
kinematic columns — those live in per-episode `ECM.parquet` /
`PSM*.parquet`. Consumers filter at the index (cheap) then load the
per-episode kinematic parquets for full payload.

Column names match the per-modality parquets they're projected from:
- `master_timestamp_ns` + `is_contiguous_to_prev` ← `timestamp.parquet`
- `contact.*` / `gesture.*` / `phase` / `step`  ← `annotation.parquet`
"""
from __future__ import annotations
import pyarrow as pa


def build_index_schema() -> pa.Schema:
    """Return the pyarrow.Schema for the frame-level index."""
    return pa.schema([
        pa.field("episode_id",            pa.string(),  nullable=False),
        pa.field("task",                  pa.string(),  nullable=False),
        pa.field("frame_index",           pa.int32(),   nullable=False),
        pa.field("master_timestamp_ns",   pa.int64(),   nullable=False),
        pa.field("is_contiguous_to_prev", pa.bool_(),   nullable=False),
        pa.field("contact.PSM1",          pa.int8(),    nullable=True),
        pa.field("contact.PSM2",          pa.int8(),    nullable=True),
        pa.field("phase",                 pa.string(),  nullable=True),
        pa.field("step",                  pa.string(),  nullable=True),
        pa.field("gesture.PSM1",          pa.string(),  nullable=True),
        pa.field("gesture.PSM2",          pa.string(),  nullable=True),
    ])
