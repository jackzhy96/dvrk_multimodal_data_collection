"""pyarrow.Schema for `meta/stats.parquet`.

Per-column statistics over the full dataset, used by consumers for
normalization presets (`code_design.md` § 2.4). List columns are
expanded by index (e.g. `state.PSM1.joint_position[0]` through `[5]`).
"""
from __future__ import annotations
import pyarrow as pa


def build_stats_schema() -> pa.Schema:
    return pa.schema([
        pa.field("column_name", pa.string(),  nullable=False),
        pa.field("dtype",       pa.string(),  nullable=False),
        pa.field("count",       pa.int64(),   nullable=False),
        pa.field("null_count",  pa.int64(),   nullable=False),
        pa.field("min",         pa.float64(), nullable=True),
        pa.field("max",         pa.float64(), nullable=True),
        pa.field("mean",        pa.float64(), nullable=True),
        pa.field("std",         pa.float64(), nullable=True),
        pa.field("q01",         pa.float64(), nullable=True),
        pa.field("q99",         pa.float64(), nullable=True),
        pa.field("vocab_size",  pa.int32(),   nullable=True),
    ])
