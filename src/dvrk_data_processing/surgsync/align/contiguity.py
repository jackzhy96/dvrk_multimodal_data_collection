"""Contiguity / frame-drop detection (`code_design.md` § 4.4).

The master timeline can have drops (missed frames in the camera record).
We label each frame `is_contiguous_to_prev` and count how many source
frames were skipped, so consumers know which frames are temporally
adjacent vs. which sit after a drop.
"""
from __future__ import annotations

import numpy as np


def detect_contiguity(
    master_ns: np.ndarray,
    *,
    period_multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect drops in the master timeline.

    expected_period_ns = median(diff(master_ns))
    is_contig[0] = False               (no previous frame)
    is_contig[i] = diff[i-1] < period_multiplier * expected_period_ns
    drop_count[i] = round(diff[i-1] / expected_period_ns) - 1
                    (clamped to >= 0)

    Returns
    -------
    is_contiguous_to_prev : np.ndarray (N,), bool
    drop_count_since_prev : np.ndarray (N,), int8
    """
    master_ns = np.asarray(master_ns, dtype=np.int64)
    N = master_ns.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int8)

    is_contig = np.zeros(N, dtype=bool)
    drop_count = np.zeros(N, dtype=np.int8)
    if N == 1:
        return is_contig, drop_count

    diffs = np.diff(master_ns).astype(np.int64)
    expected = int(np.median(diffs))
    if expected <= 0:
        # Degenerate timeline (constant or zero diff). Mark everything
        # contiguous to avoid spurious drop counts.
        is_contig[1:] = True
        return is_contig, drop_count

    threshold = expected * period_multiplier
    contig = diffs < threshold
    is_contig[1:] = contig

    # drop_count = round(diff / expected) - 1, clamped to 0..127 (int8).
    ratio = (diffs / float(expected)).round().astype(np.int64) - 1
    ratio = np.clip(ratio, 0, 127)
    drop_count[1:] = ratio.astype(np.int8)

    return is_contig, drop_count
