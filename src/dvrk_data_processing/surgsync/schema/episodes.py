"""pyarrow.Schema for `meta/episodes.parquet/task=*/part-*.parquet`.

This is the Hive-partitioned episode-level index (`code_design.md` § 2.2).
One row per episode, queryable across episodes without opening per-episode
files. The `task` column is partition-encoded by directory; it's still
included in the schema so the parquet is portable when moved out of the
Hive layout.
"""
from __future__ import annotations
import pyarrow as pa


def build_episodes_schema() -> pa.Schema:
    """Return the pyarrow.Schema for the episode-level index."""
    return pa.schema([
        pa.field("episode_id",                          pa.string(),  nullable=False),
        pa.field("task",                                pa.string(),  nullable=False),
        pa.field("length_frames",                       pa.int32(),   nullable=False),
        pa.field("duration_s",                          pa.float32(), nullable=False),
        pa.field("recorder_variant",                    pa.string(),  nullable=False),
        pa.field("sync_policy",                         pa.string(),  nullable=False),
        pa.field("source_clip",                         pa.string(),  nullable=False),
        pa.field("operator_skill_level",                pa.string(),  nullable=True),
        pa.field("case_type",                           pa.string(),  nullable=True),
        pa.field("tool.PSM1",                           pa.string(),  nullable=True),
        pa.field("tool.PSM2",                           pa.string(),  nullable=True),
        pa.field("failure_episodes_json",               pa.string(),  nullable=True),
        pa.field("recovery_episodes_json",              pa.string(),  nullable=True),
        pa.field("image_size.width",                    pa.int32(),   nullable=False),
        pa.field("image_size.height",                   pa.int32(),   nullable=False),
        pa.field("has_preprocess",                      pa.bool_(),   nullable=False),
        pa.field("has_preview",                         pa.bool_(),   nullable=False),
        pa.field("has_video_raw",                       pa.bool_(),   nullable=False),
        pa.field("has_calibrated_kinematic",            pa.bool_(),   nullable=False),
        pa.field("pipeline_versions.rectify_resize",    pa.string(),  nullable=True),
        pa.field("pipeline_versions.kinematic_handeye", pa.string(),  nullable=True),
        pa.field("pipeline_versions.depth_estimation",  pa.string(),  nullable=True),
        pa.field("pipeline_versions.optical_flow_raft", pa.string(),  nullable=True),
        pa.field("built_at_utc_ns",                     pa.int64(),   nullable=False),
        pa.field("frames_sha256",                       pa.string(),  nullable=False),
    ])
