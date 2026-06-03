"""Write `<staging>/episode_meta.json` via the EpisodeMeta pydantic model.

Atomic temp + rename so a kill mid-write leaves no partial file.
Uses the SCHEMA_VERSION from `schema/__init__.py` as the single source
of truth.
"""
from __future__ import annotations
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dvrk_data_processing.surgsync.align.aligner import AlignedClip
from dvrk_data_processing.surgsync.ingest.calibration import CalibrationBundle
from dvrk_data_processing.surgsync.schema import (
    SCHEMA_VERSION,
    EpisodeMeta,
    SyncStats as MetaSyncStats,
    PipelineVersions,
    Tool,
)
from dvrk_data_processing.surgsync.serde.meta_io import ClipMeta


log = logging.getLogger(__name__)


def compute_episode_id(dataset_name: str, clip_index: str, master_t0_ns: int) -> str:
    """Deterministic time-based composite id.

    Format: `<dataset_name>_<clip_idx>_<master_t0_ns>`
    Example: `online_data_2_1754609707325800839`

    Replaces the prior UUID5 scheme. The id encodes the same three
    inputs that the UUID5 mixed together, but stays human-readable and
    keeps the master_t0_ns directly inspectable (useful when
    cross-referencing against ROS bags).

    Deterministic across re-runs because all three inputs come from
    source data, not from build-time randomness.
    """
    return f"{dataset_name}_{clip_index}_{master_t0_ns}"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(path.parent), suffix=".tmp", delete=False,
    ) as tmp:
        # `Optional[float]` fields serialize as JSON `null` when None;
        # there is no NaN-to-zero fallback here. The aligner returns
        # None for "no present stamps" cases — see SyncStats docstring.
        json.dump(payload, tmp, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_episode_meta(
    *,
    clip_meta: ClipMeta,
    aligned: AlignedClip,
    task: str,
    source_clip: str,
    recorder_variant: str,
    dataset_name: str,
    clip_index: str,
    image_size: list[int],
    has_preprocess: bool,
    has_preview: bool,
    has_video_raw: bool,
    has_calibrated_kinematic: bool = False,
    dst_path: Path,
    pipeline_versions: Optional[dict[str, str]] = None,
) -> EpisodeMeta:
    """Build and atomically write episode_meta.json.

    Returns the EpisodeMeta object for the caller's structured log.
    """
    n = aligned.n_frames
    if n == 0:
        raise ValueError("cannot write episode_meta for an empty AlignedClip")

    # `aligned.master_ns` is now clip-relative (master_ns[0] == 0), so
    # the duration is just the last entry. The absolute t0 lives on
    # `aligned.master_t0_ns` and is preserved into the JSON below.
    duration_s = float(aligned.master_ns[-1] / 1e9)

    sync_policy = "strict" if recorder_variant == "online" else "nearest_interp"

    pv = PipelineVersions(**(pipeline_versions or {}))

    em = EpisodeMeta(
        schema_version=SCHEMA_VERSION,
        # episode_id still encodes the absolute t0 so two clips that
        # both start at relative 0 have different ids.
        episode_id=compute_episode_id(dataset_name, clip_index, aligned.master_t0_ns),
        task=task,
        length_frames=n,
        duration_s=duration_s,
        master_t0_ns=aligned.master_t0_ns,
        recorder_variant=recorder_variant,
        sync_policy=sync_policy,
        source_clip=source_clip,
        operator_skill_level=clip_meta.operator_skill_level,
        case_type=clip_meta.case_type,
        tool=Tool(PSM1=clip_meta.tool.get("PSM1"), PSM2=clip_meta.tool.get("PSM2")),
        failure_episodes=list(clip_meta.failure),
        recovery_episodes=list(clip_meta.recovery),
        image_size=image_size,
        sync_stats=MetaSyncStats(
            episode_length_s=aligned.sync_stats.episode_length_s,
            median_kin_delta_ms=aligned.sync_stats.median_kin_delta_ms,
            max_kin_delta_ms=aligned.sync_stats.max_kin_delta_ms,
            cross_modal_median_delta_ms=aligned.sync_stats.cross_modal_median_delta_ms,
            cross_modal_mean_delta_ms=aligned.sync_stats.cross_modal_mean_delta_ms,
            cross_modal_std_delta_ms=aligned.sync_stats.cross_modal_std_delta_ms,
            cross_modal_max_delta_ms=aligned.sync_stats.cross_modal_max_delta_ms,
            frames_dropped=aligned.sync_stats.frames_dropped,
            contiguity_ratio=aligned.sync_stats.contiguity_ratio,
            out_of_tol_counts=aligned.sync_stats.out_of_tol_counts,
        ),
        pipeline_versions=pv,
        has_preprocess=has_preprocess,
        has_preview=has_preview,
        has_video_raw=has_video_raw,
        has_calibrated_kinematic=has_calibrated_kinematic,
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    _atomic_write_json(dst_path, em.model_dump())
    return em
