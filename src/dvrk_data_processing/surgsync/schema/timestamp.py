"""pyarrow.Schema for `<episode>/timestamp.parquet`.

The image-side master timeline. One row per stereo-left frame. Carries:
- master clock (clip-relative ns since master_t0_ns),
- contiguity flags,
- **every per-topic delta-to-master column** for the modalities listed
  in `align.topics.TIMESTAMP_TOPICS`.

The delta columns previously lived under `ECM.parquet` / `PSM*.parquet`;
consolidating them in timestamp.parquet keeps the arm parquets focused
on robot-state values and makes the cross-modal alignment story a
single-file read.
"""
from __future__ import annotations
import pyarrow as pa

from dvrk_data_processing.surgsync.align.topics import TIMESTAMP_TOPICS


def build_timestamp_schema() -> pa.Schema:
    fields = [
        pa.field("frame_index",        pa.int32(),  nullable=False),
        pa.field("source_frame_index", pa.int32(),  nullable=False),
        pa.field("master_timestamp_ns", pa.int64(), nullable=False),
    ]
    # One `delta_to_master.<topic>_ns` column per topic in the catalog.
    # int32 because deltas are bounded by the master frame period (~few
    # 100 ms), nullable because some recorders don't publish certain
    # topics (e.g. offline setpoint_cp).
    for topic in TIMESTAMP_TOPICS:
        fields.append(pa.field(
            f"delta_to_master.{topic.name}_ns",
            pa.int32(),
            nullable=True,
        ))
    fields += [
        pa.field("is_contiguous_to_prev", pa.bool_(), nullable=False),
        pa.field("drop_count_since_prev", pa.int8(),  nullable=False),
    ]
    return pa.schema(fields)
