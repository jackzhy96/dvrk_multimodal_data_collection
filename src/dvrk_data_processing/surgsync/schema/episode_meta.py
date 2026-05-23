"""pydantic model for `<episode_id>/episode_meta.json` (`code_design.md` § 3.2).

Every field is `extra = "forbid"` so a typo in the encoder raises at
construction time. `Tool`, `SyncStats`, and `PipelineVersions` are nested
strict models for the same reason.

Schema version is **not** defaulted here — it must be passed explicitly
by the encoder so all episode files in one release agree.
"""
from __future__ import annotations
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    PSM1: Optional[str] = None
    PSM2: Optional[str] = None


class SyncStats(BaseModel):
    """Surface alignment quality to consumers (Risk A-8).

    Lightweight per-episode sync summary. Detailed per-topic latency
    (median / mean / std / max + max_delta_frame_idx + n_present for
    each modality) is in the sibling `time_sync_stat.json`, not here.

    `median_kin_delta_ms` / `max_kin_delta_ms` cover the four PSM
    measured topics only (legacy kinematic-only summary).
    `cross_modal_{median,mean,std,max}_delta_ms` aggregate over
    **every** synced modality (image_right, image_side, ECM/PSM1/PSM2
    × {measured_cp, measured_js, setpoint_cp, setpoint_js}) pooled
    into one |delta| array — same population, four summary stats.
    `std` is `ddof=0` (population standard deviation).

    **Null semantics**: every `*_delta_ms` field is `Optional[float]`.
    A modality with no present stamps contributes nothing to the
    pool; when every modality is empty the four cross-modal aggregates
    and the legacy kin-only aggregates are `null`. Consumers should
    distinguish `null` ("no data") from `0.0` ("zero latency").

    `episode_length_s` is the clip duration in seconds, mirrored
    from `EpisodeMeta.duration_s` for callers reading `sync_stats`
    in isolation. The earlier `expected_frame_period_ns` field was
    removed — derive it as `length_frames / episode_length_s` if
    needed.
    """
    model_config = ConfigDict(extra="forbid")

    episode_length_s: float
    median_kin_delta_ms: Optional[float]
    max_kin_delta_ms: Optional[float]
    cross_modal_median_delta_ms: Optional[float]
    cross_modal_mean_delta_ms: Optional[float]
    cross_modal_std_delta_ms: Optional[float]
    cross_modal_max_delta_ms: Optional[float]
    frames_dropped: int
    contiguity_ratio: float
    # Per-modality out-of-tolerance counters. Keys are dotted modality
    # names (e.g. "PSM1.measured_cp", "image_right"). Values are NULL
    # counts in the corresponding `delta_to_master.*_ns` column.
    out_of_tol_counts: dict[str, int] = Field(default_factory=dict)


class PipelineVersions(BaseModel):
    """Git SHAs of the four preprocessing stages that produced this episode."""
    model_config = ConfigDict(extra="forbid")

    rectify_resize: Optional[str] = None
    kinematic_handeye: Optional[str] = None
    depth_estimation: Optional[str] = None
    optical_flow_raft: Optional[str] = None


class EpisodeMeta(BaseModel):
    """`episode_meta.json` for one finalized episode."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    episode_id: str
    task: str
    length_frames: int
    duration_s: float
    # Absolute (since-epoch) ns timestamp of frame 0. Every
    # `master_timestamp_ns` column in this episode's parquet files is
    # stored *relative* to this offset (row 0 == 0). To recover the
    # original absolute stamp of row i: `master_t0_ns + master_timestamp_ns[i]`.
    master_t0_ns: int
    recorder_variant: str   # "offline" | "online"
    sync_policy: str        # "strict" | "nearest_interp"
    source_clip: str

    operator_skill_level: Optional[str] = None
    case_type: Optional[str] = None
    tool: Tool = Field(default_factory=Tool)

    failure_episodes: list[list[int]] = Field(default_factory=list)
    recovery_episodes: list[list[int]] = Field(default_factory=list)

    image_size: list[int]   # [width, height]; pydantic refuses a tuple, list is fine
    sync_stats: SyncStats
    pipeline_versions: PipelineVersions = Field(default_factory=PipelineVersions)

    # `has_preprocess` is the canonical name; `has_geometry` is the legacy
    # alias accepted on read so episode_meta.json files written before the
    # rename still deserialize. New builds emit `has_preprocess` only.
    has_preprocess: bool = Field(
        default=False,
        validation_alias=AliasChoices("has_preprocess", "has_geometry"),
    )
    has_preview: bool = False
    has_video_raw: bool = False
    # Per-frame PSM tool-tip pose in the left-rectified camera frame,
    # produced by preprocessing's hand-eye stage. Optional — true iff at least one
    # frame had a usable `calibrated_kinematic/<i>.json` ingested.
    has_calibrated_kinematic: bool = False

    built_at_utc: str
