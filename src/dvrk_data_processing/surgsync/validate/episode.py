"""Per-episode post-pack validator.

Checks one finalized (or staged) episode for:
- episode_meta.json schema integrity
- every per-modality parquet exists and its row count == episode_meta.json.length_frames
- MKV frame counts match (where applicable)
- contiguity ratio above the configured floor (when supplied)
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.encode.codec import probe_frame_count
from dvrk_data_processing.surgsync.schema import EpisodeMeta, SCHEMA_VERSION
from dvrk_data_processing.surgsync.validate.types import ValidationIssue


log = logging.getLogger(__name__)


def validate_episode(
    episode_dir: Path,
    *,
    contiguity_floor: Optional[float] = None,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    episode_dir = Path(episode_dir)

    # The completion manifest is the contract for "this episode
    # finished a successful pack". Missing manifest → either never
    # started, in-flight (running marker present), or crashed
    # (failed marker present). Validators don't try to recover; they
    # just refuse to ship the episode.
    sentinel = episode_dir / ".surgsync_complete.json"
    if not sentinel.is_file():
        running = (episode_dir / ".surgsync_running.json").is_file()
        failed  = (episode_dir / ".surgsync_failed.json").is_file()
        status = (
            "in-flight (running marker present)" if running and not failed
            else "previously failed" if failed
            else "never finished"
        )
        return [ValidationIssue(
            "ERROR", "ep_incomplete",
            f".surgsync_complete.json missing under {episode_dir} — {status}. "
            "Re-run with force=true.",
        )]

    em_path = episode_dir / "episode_meta.json"
    if not em_path.is_file():
        return [ValidationIssue("ERROR", "ep_missing_meta",
                                f"episode_meta.json missing under {episode_dir}")]

    try:
        with open(em_path) as f:
            em_dict = json.load(f)
        em = EpisodeMeta.model_validate(em_dict)
    except Exception as e:
        return [ValidationIssue("ERROR", "ep_schema_drift",
                                f"episode_meta.json fails schema: {e}")]

    if em.schema_version != SCHEMA_VERSION:
        issues.append(ValidationIssue(
            "ERROR", "ep_schema_version_mismatch",
            f"episode.schema_version={em.schema_version} != {SCHEMA_VERSION}",
        ))

    # Every per-modality parquet exists + has length == length_frames.
    REQUIRED_PARQUETS = (
        "timestamp.parquet",
        "ECM.parquet",
        "PSM1.parquet",
        "PSM2.parquet",
        "annotation.parquet",
    )
    for name in REQUIRED_PARQUETS:
        p = episode_dir / name
        if not p.is_file():
            issues.append(ValidationIssue("ERROR", "ep_missing_parquet",
                                          f"{name} missing under {episode_dir}"))
            continue
        try:
            n_rows = pq.read_metadata(p).num_rows
        except Exception as e:
            issues.append(ValidationIssue("ERROR", "ep_parquet_broken",
                                          f"failed to read {name}: {e}"))
            continue
        if n_rows != em.length_frames:
            issues.append(ValidationIssue(
                "ERROR", "ep_length_mismatch",
                f"{name} has {n_rows} rows; episode_meta.json says {em.length_frames}",
            ))

    # Per-stream frame counts where present.
    # `video/` is MP4 (H.264 — lossy), `video_raw/` and `preprocess/` are MKV
    # (FFV1 — bit-exact). Glob the right extension per subdir.
    sub_to_glob = {"video": "*.mp4", "video_raw": "*.mkv", "preprocess": "*.mkv"}
    for sub, pattern in sub_to_glob.items():
        sub_dir = episode_dir / sub
        if not sub_dir.is_dir():
            continue
        for stream_path in sorted(sub_dir.glob(pattern)):
            try:
                n = probe_frame_count(stream_path)
            except Exception as e:
                issues.append(ValidationIssue("ERROR", "ep_video_undecodable",
                                              f"{stream_path.relative_to(episode_dir)}: {e}"))
                continue
            if n != em.length_frames:
                issues.append(ValidationIssue(
                    "ERROR", "ep_video_length_mismatch",
                    f"{stream_path.relative_to(episode_dir)}: {n} frames; expected {em.length_frames}",
                ))

    # Contiguity floor (optional).
    if contiguity_floor is not None and em.sync_stats.contiguity_ratio < contiguity_floor:
        issues.append(ValidationIssue(
            "ERROR", "ep_low_contiguity",
            f"contiguity_ratio={em.sync_stats.contiguity_ratio:.3f} "
            f"< floor={contiguity_floor}",
        ))

    # Sanity-only: has_video_raw must be true per the packer invertibility
    # contract. Demoted to WARNING since the build CLI exposes a flag
    # that can disable raw video — operator opt-out is their call.
    if not em.has_video_raw:
        issues.append(ValidationIssue(
            "WARNING", "ep_no_video_raw",
            "has_video_raw=false — raw image PNGs cannot be reconstructed by decompose",
        ))

    return issues
