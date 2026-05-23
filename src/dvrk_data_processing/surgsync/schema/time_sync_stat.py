"""pydantic model for `<episode>/time_sync_stat.json`.

Per-topic alignment latency detail. One row per synced modality
(image streams + per-arm kinematic topics), aggregated over rows whose
source stamp was present. Sits alongside `episode_meta.json` — the
lightweight cross-modal summary lives there; the detailed per-topic
breakdown lives here.

Per-topic fields:
- `median_delta_ms` / `mean_delta_ms` / `std_delta_ms` / `max_delta_ms`:
  summary statistics over `|delta_to_master|` for the topic, in
  milliseconds. `std` is ddof=0 (population std), matching the
  cross-modal summary in `episode_meta.json`.
- `max_delta_frame_idx`: the master-timeline frame index (0-based)
  at which the maximum `|delta|` occurred. Useful for spot-checking
  the worst-aligned frame in a viewer or for debugging recorder
  glitches.
- `n_present`: how many master frames had a present stamp for this
  topic. Topics with `n_present == 0` (e.g. offline recorder's
  `setpoint_cp_*` modalities) surface every other field as `null`.

Topic keys mirror `episode_meta.json.sync_stats.out_of_tol_counts`
so the two files can be joined on topic name.
"""
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PerTopicLatency(BaseModel):
    """Latency summary for one synced modality."""
    model_config = ConfigDict(extra="forbid")

    median_delta_ms: Optional[float]
    mean_delta_ms: Optional[float]
    std_delta_ms: Optional[float]
    max_delta_ms: Optional[float]
    # Master-frame index (0-based) of the row that produced the maximum
    # |delta|. None when n_present == 0.
    max_delta_frame_idx: Optional[int]
    # Number of master frames for which this topic had a present stamp.
    # Counter, never NULL — explicit zero means "no data for this topic".
    n_present: int


class TimeSyncStat(BaseModel):
    """`time_sync_stat.json` for one finalized episode."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    episode_id: str
    # Topic name (e.g. "PSM1.measured_cp", "image_right") → its latency
    # summary. Keys match the per-modality keys in `episode_meta.json.
    # sync_stats.out_of_tol_counts` for clean joining.
    per_topic: dict[str, PerTopicLatency] = Field(default_factory=dict)
