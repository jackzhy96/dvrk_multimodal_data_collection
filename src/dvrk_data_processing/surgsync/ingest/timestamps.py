"""Parse `time_syn/<frame>.json` into a flat per-modality timestamp table.

The raw timestamp schema is per-frame and deeply nested (see
`specs/raw_data_spec.md` § Time-sync schema notes). This module
flattens every topic listed in `align.topics.TIMESTAMP_TOPICS` into a
numpy-friendly dict-of-arrays — one int64 array per topic. NULL stamps
(missing files / missing keys / offline-recorder gaps) are encoded as
`NULL_TS = np.iinfo(np.int64).min` so the dtype stays int64 with no
parallel mask array; the align stage maps `NULL_TS` to Arrow NULL.

The canonical topic catalog lives in `align/topics.py`. Adding or
removing a topic touches that file, not this one.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dvrk_data_processing.surgsync.align.topics import (
    MASTER_STAMP_PATH,
    TIMESTAMP_TOPICS,
    extract_stamp,
)


# Sentinel: NULL timestamp. Using INT64_MIN keeps the dtype int64 (and
# Arrow nullability simple) without needing a parallel mask array.
NULL_TS = np.iinfo(np.int64).min


def _stamp_to_ns(stamp: dict | None) -> int:
    """Convert `{"sec": ..., "nsec": ...}` to int64 ns. Missing → NULL."""
    if stamp is None:
        return NULL_TS
    sec = stamp.get("sec")
    nsec = stamp.get("nsec")
    if sec is None or nsec is None:
        return NULL_TS
    return int(sec) * 1_000_000_000 + int(nsec)


@dataclass
class TimestampTable:
    """Flat per-topic timestamp arrays.

    `master_ns` is the stereo-left capture timestamp (canonical master
    clock per `specs/code_design.md` § 4.1). `topic_stamps[<name>]`
    is the per-frame stamp for that topic — keys come from
    `align.topics.TOPIC_NAMES`. All arrays are length-N int64 with
    `NULL_TS` marking missing values.
    """
    source_frame_indices: np.ndarray   # int64, shape (N,)
    master_ns: np.ndarray              # int64 (NULL_TS where missing)
    topic_stamps: dict[str, np.ndarray] = field(default_factory=dict)


def load_timestamps(time_syn_dir: Path) -> TimestampTable:
    """Parse every `time_syn/<frame>.json` under `time_syn_dir` into a
    `TimestampTable`.

    Frame indices come from the filenames, sorted numerically. The
    order of the returned arrays follows that sort, so `master_ns[i]`
    is the stereo-left stamp of source frame `source_frame_indices[i]`.

    Topics not present in a given clip's JSON (e.g. offline recorder's
    `setpoint_cp_stamp`) surface as all-`NULL_TS` arrays, which the
    align stage treats as "no data for this topic" and the encoder
    converts to Arrow NULL columns.
    """
    from dvrk_data_processing.surgsync.ingest.clip import sorted_frames

    files = sorted_frames(time_syn_dir, suffix=".json")
    if not files:
        raise FileNotFoundError(f"No time_syn JSON files under {time_syn_dir}")

    N = len(files)
    source_indices = np.zeros(N, dtype=np.int64)
    master_ns = np.full(N, NULL_TS, dtype=np.int64)
    # Pre-allocate one array per topic.
    topic_stamps: dict[str, np.ndarray] = {
        t.name: np.full(N, NULL_TS, dtype=np.int64) for t in TIMESTAMP_TOPICS
    }

    for i, p in enumerate(files):
        source_indices[i] = int(p.stem)
        with open(p) as f:
            ts = json.load(f)

        # Master clock first.
        master_ns[i] = _stamp_to_ns(extract_stamp(ts, MASTER_STAMP_PATH))

        # Every other topic, table-driven.
        for topic in TIMESTAMP_TOPICS:
            topic_stamps[topic.name][i] = _stamp_to_ns(extract_stamp(ts, topic.path))

    return TimestampTable(
        source_frame_indices=source_indices,
        master_ns=master_ns,
        topic_stamps=topic_stamps,
    )
