"""Master clock derivation.

The master clock is the **stereo-left camera capture timestamp** — i.e.
the camera driver's ROS header.stamp, NOT the rosbag record time.
Encoded in `time_syn/<frame>.json::image_left_stamp` per the raw spec.

This module is a thin wrapper around `ingest.timestamps.TimestampTable`
that fishes out the master clock and the source frame indices into a
shape the rest of `align/` expects.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from dvrk_data_processing.surgsync.ingest.timestamps import TimestampTable, NULL_TS


@dataclass
class MasterTimeline:
    """The master clock and the source frame indices it aligns to.

    Both arrays have shape (N,) int64. `master_ns[i]` is the stereo-left
    capture stamp of source frame `source_frame_indices[i]`. Both are
    sorted by source frame index ascending — that's the convention
    `align_clip` and the encoders rely on.
    """
    source_frame_indices: np.ndarray   # int64
    master_ns: np.ndarray              # int64


def build_master_timeline(ts: TimestampTable) -> MasterTimeline:
    """Build the master timeline from a TimestampTable.

    Drops any frame whose `master_ns` is NULL (sentinel value) — those
    shouldn't exist in well-formed clips, but guarding here prevents
    downstream matchers from blowing up on a malformed input.

    Side effect: also filters every array in `ts.topic_stamps` to the
    same keep mask + sort order, so downstream callers can pair
    `ts.topic_stamps[name][i]` with `master.master_ns[i]` without
    worrying about row alignment. Mutates the input `ts` in place.
    """
    keep = ts.master_ns != NULL_TS
    if not keep.all():
        # Document but don't raise; the validator will surface this if
        # it's actually a concern.
        pass
    src_idx = ts.source_frame_indices[keep]
    master = ts.master_ns[keep]
    # Sort by source frame index — TimestampTable already does, but be
    # defensive in case a future ingest change reorders.
    order = np.argsort(src_idx, kind="stable")

    # Apply the same keep+order to every topic stamp array, so callers
    # iterating `ts.topic_stamps[name]` see a row-aligned view.
    ts.source_frame_indices = src_idx[order]
    ts.master_ns = master[order]
    ts.topic_stamps = {
        name: arr[keep][order] for name, arr in ts.topic_stamps.items()
    }

    return MasterTimeline(
        source_frame_indices=ts.source_frame_indices,
        master_ns=ts.master_ns,
    )
