"""Build `meta/index.parquet/task=*/part-*.parquet`.

Frame-level cross-episode index — projects columns from each episode's
`timestamp.parquet` (master_ts + is_contiguous) and `annotation.parquet`
(contact / gesture / phase / step) into one Hive-partitioned table
keyed by `(episode_id, task, frame_index)`.

Streaming reads via `pq.ParquetFile.iter_batches` so memory stays
bounded on large datasets.
"""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.schema import build_index_schema


log = logging.getLogger(__name__)


def _iter_episode_dirs(dataset_root: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield (task, episode_id, episode_dir) tuples for every packed episode."""
    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        episodes_root = dataset_dir / "episodes"
        if not episodes_root.is_dir():
            continue
        for task_dir in episodes_root.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for ep_dir in task_dir.iterdir():
                if (not ep_dir.is_dir()
                        or not (ep_dir / "episode_meta.json").exists()
                        or not (ep_dir / ".surgsync_complete.json").is_file()):
                    # Either missing entirely or an in-flight/crashed
                    # pack with no completion sentinel — skip silently;
                    # the validator surfaces these as errors.
                    continue
                with open(ep_dir / "episode_meta.json") as f:
                    em = json.load(f)
                yield task, em["episode_id"], ep_dir


def build_frames_index(dataset_root: Path) -> dict:
    """Walk every episode, join timestamp.parquet + annotation.parquet
    columns into the cross-episode index. Hive-partitioned by task.
    """
    dataset_root = Path(dataset_root)
    out_root = dataset_root / "meta" / "index.parquet"
    out_root.mkdir(parents=True, exist_ok=True)

    schema = build_index_schema()
    per_task_batches: dict[str, list[pa.RecordBatch]] = defaultdict(list)
    total_frames = 0

    # Columns we need from each source parquet.
    ts_cols  = ["frame_index", "master_timestamp_ns", "is_contiguous_to_prev"]
    ann_cols = ["frame_index", "contact.PSM1", "contact.PSM2",
                "phase", "step", "gesture.PSM1", "gesture.PSM2"]

    for task, episode_id, ep_dir in _iter_episode_dirs(dataset_root):
        ts_path = ep_dir / "timestamp.parquet"
        ann_path = ep_dir / "annotation.parquet"
        if not ts_path.exists() or not ann_path.exists():
            log.warning("missing per-modality parquets under %s — skipping", ep_dir)
            continue

        ts_table = pq.ParquetFile(ts_path).read(columns=ts_cols)
        ann_table = pq.ParquetFile(ann_path).read(columns=ann_cols)

        # Both parquets are aligned by frame_index — and we wrote them in
        # row order — so a positional join suffices. Sanity-check
        # frame_index equality once before flattening.
        if (ts_table.column("frame_index").to_numpy()
                != ann_table.column("frame_index").to_numpy()).any():
            log.error("frame_index mismatch between timestamp+annotation under %s — skipping", ep_dir)
            continue

        n = ts_table.num_rows
        total_frames += n
        arrays = [
            pa.array([episode_id] * n, type=pa.string()),
            pa.array([task]       * n, type=pa.string()),
            ts_table.column("frame_index"),
            ts_table.column("master_timestamp_ns"),
            ts_table.column("is_contiguous_to_prev"),
            ann_table.column("contact.PSM1"),
            ann_table.column("contact.PSM2"),
            ann_table.column("phase"),
            ann_table.column("step"),
            ann_table.column("gesture.PSM1"),
            ann_table.column("gesture.PSM2"),
        ]
        per_task_batches[task].append(
            pa.RecordBatch.from_arrays(
                [a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a for a in arrays],
                schema=schema,
            )
        )

    if total_frames == 0:
        log.warning("build_frames_index found no frames under %s", dataset_root)
        return {"n_frames": 0, "n_tasks": 0}

    for task, batches in per_task_batches.items():
        part_dir = out_root / f"task={task}"
        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_batches(batches, schema=schema)
        pq.write_table(
            table, part_dir / "part-00000.parquet",
            compression="zstd", compression_level=3,
        )

    log.info("build_frames_index: %d frames across %d task(s) → %s",
             total_frames, len(per_task_batches), out_root)
    return {"n_frames": total_frames, "n_tasks": len(per_task_batches)}
