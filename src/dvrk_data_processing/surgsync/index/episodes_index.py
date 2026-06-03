"""Build `meta/episodes.parquet/task=*/part-*.parquet` + `meta/episodes.jsonl`.

Walks every `<dataset_root>/<dataset_name>/episodes/<task>/<idx>/episode_meta.json`,
parses into EpisodeMeta, groups by task, writes Hive-partitioned parquet
plus a JSONL convenience copy for humans + grep.
"""
from __future__ import annotations
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.schema import build_episodes_schema, EpisodeMeta


log = logging.getLogger(__name__)


# Episode dirs sit at <root>/<dataset>/episodes/<task>/<clip_idx>/.
# Walk three levels under any first-level <dataset> dir that contains
# an "episodes" subdir. Skip top-level meta/, .logs/, .staging/.
def _iter_episode_dirs(dataset_root: Path) -> Iterable[Path]:
    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        episodes_root = dataset_dir / "episodes"
        if not episodes_root.is_dir():
            continue
        for task_dir in episodes_root.iterdir():
            if not task_dir.is_dir():
                continue
            for ep_dir in task_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                # Only ship completed episodes into the cross-episode
                # index — skip dirs without the `.surgsync_complete.json`
                # sentinel (in-flight or crashed prior pack).
                if ((ep_dir / "episode_meta.json").exists()
                        and (ep_dir / ".surgsync_complete.json").is_file()):
                    yield ep_dir


def _row_for_episode(ep_dir: Path) -> dict[str, Any]:
    """Read episode_meta.json + a stat of frames.parquet for the parquet row.

    Mirrors the episode-level index schema (with the additional
    `has_video_raw` field we added for the packer invertibility contract).
    """
    import hashlib
    with open(ep_dir / "episode_meta.json") as f:
        em_dict = json.load(f)
    em = EpisodeMeta.model_validate(em_dict)

    # frames_sha256 — cheap streaming hash for spot-checks. We hash
    # `timestamp.parquet` as the canonical row-count parquet now that
    # the monolithic frames.parquet is gone. Renaming the field in the
    # index would be a breaking change for downstream consumers, so
    # the column name stays the same.
    h = hashlib.sha256()
    canonical_path = ep_dir / "timestamp.parquet"
    with open(canonical_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return {
        "episode_id":                          em.episode_id,
        "task":                                em.task,
        "length_frames":                       em.length_frames,
        "duration_s":                          em.duration_s,
        "recorder_variant":                    em.recorder_variant,
        "sync_policy":                         em.sync_policy,
        "source_clip":                         em.source_clip,
        "operator_skill_level":                em.operator_skill_level,
        "case_type":                           em.case_type,
        "tool.PSM1":                           em.tool.PSM1,
        "tool.PSM2":                           em.tool.PSM2,
        "failure_episodes_json":               json.dumps(em.failure_episodes),
        "recovery_episodes_json":              json.dumps(em.recovery_episodes),
        "image_size.width":                    em.image_size[0],
        "image_size.height":                   em.image_size[1],
        "has_preprocess":                      em.has_preprocess,
        "has_preview":                         em.has_preview,
        "has_video_raw":                       em.has_video_raw,
        "has_calibrated_kinematic":            em.has_calibrated_kinematic,
        "pipeline_versions.rectify_resize":    em.pipeline_versions.rectify_resize,
        "pipeline_versions.kinematic_handeye": em.pipeline_versions.kinematic_handeye,
        "pipeline_versions.depth_estimation":  em.pipeline_versions.depth_estimation,
        "pipeline_versions.optical_flow_raft": em.pipeline_versions.optical_flow_raft,
        "built_at_utc_ns":                     _iso_to_ns(em.built_at_utc),
        "frames_sha256":                       h.hexdigest(),
    }


_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?")


def _iso_to_ns(iso: str) -> int:
    """Parse a UTC ISO 8601 timestamp into int64 nanoseconds.

    We accept either fractional seconds or none; missing trailing
    timezone info is treated as UTC.
    """
    import datetime as dt
    # Python's fromisoformat doesn't handle the "Z" suffix until 3.11; strip it.
    s = iso.rstrip("Z")
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        # Fallback for weird formats — best-effort.
        m = _ISO_RE.match(s)
        if not m:
            return 0
        y, mo, d, h, mi, sec = (int(x) for x in m.groups()[:6])
        frac = m.group(7) or "0"
        t = dt.datetime(y, mo, d, h, mi, sec, int(frac[:6].ljust(6, "0")))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1_000_000_000)


def build_episodes_index(dataset_root: Path) -> dict[str, Any]:
    """Walk every episode, write Hive-partitioned parquet + JSONL convenience copy.

    Returns a small summary dict for the structured log.
    """
    dataset_root = Path(dataset_root)
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []

    for ep_dir in _iter_episode_dirs(dataset_root):
        row = _row_for_episode(ep_dir)
        rows_by_task[row["task"]].append(row)
        all_rows.append(row)

    if not all_rows:
        log.warning("build_episodes_index found no episodes under %s", dataset_root)
        return {"n_episodes": 0, "n_tasks": 0}

    schema = build_episodes_schema()
    meta_dir = dataset_root / "meta" / "episodes.parquet"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for task, rows in rows_by_task.items():
        # One part file per task — for our scale this fits comfortably.
        part_dir = meta_dir / f"task={task}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            table, part_dir / "part-00000.parquet",
            compression="zstd", compression_level=3,
        )

    # JSONL convenience copy — one row per episode, sorted by id for
    # deterministic output (helps git diff if anyone checks this in).
    jsonl_path = dataset_root / "meta" / "episodes.jsonl"
    all_rows.sort(key=lambda r: r["episode_id"])
    with open(jsonl_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, default=str) + "\n")

    log.info("build_episodes_index: %d episodes across %d task(s) → %s",
             len(all_rows), len(rows_by_task), meta_dir)
    return {"n_episodes": len(all_rows), "n_tasks": len(rows_by_task)}
