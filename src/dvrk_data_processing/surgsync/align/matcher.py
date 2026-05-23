"""Nearest-within-tolerance matching (`code_design.md` § 4.2).

`match_modality` is the workhorse — for each master frame, it picks the
closest candidate sample within `tol_ns`. Returns `-1` and an arbitrary
delta where no match is in tolerance; the caller is responsible for
NULLing the corresponding state/action column AND the matching
`delta_to_master.*_ns` column (the "no silent NULLs" rule).
"""
from __future__ import annotations

import numpy as np


def match_modality(
    master_ns: np.ndarray,            # shape (N,) int64; strictly increasing
    candidate_ns: np.ndarray,         # shape (M,) int64; sorted ascending
    tol_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each master timestamp, return the index of the nearest
    candidate within `tol_ns`, plus the signed delta.

    Returns
    -------
    indices : np.ndarray (N,), int32
        Index into `candidate_ns` of the matched sample, or -1 when no
        candidate is within tolerance.
    deltas : np.ndarray (N,), int32
        `candidate_ns[indices[i]] - master_ns[i]` when indices[i] >= 0;
        arbitrary (typically 0) where indices[i] == -1. Callers should
        mask deltas using indices.

    Implementation: `np.searchsorted` to find the insertion point of
    each master_ns in candidate_ns, then pick min(left_neighbor,
    right_neighbor) by absolute distance. O((N + M) log M).
    """
    master_ns = np.asarray(master_ns, dtype=np.int64)
    candidate_ns = np.asarray(candidate_ns, dtype=np.int64)
    N = master_ns.shape[0]
    M = candidate_ns.shape[0]

    if M == 0:
        return np.full(N, -1, dtype=np.int32), np.zeros(N, dtype=np.int32)

    # Insertion point of each master_ns into candidate_ns (sorted).
    # `right` so duplicates favor the rightmost match — doesn't matter for
    # the typical case where candidate timestamps are unique.
    ins = np.searchsorted(candidate_ns, master_ns, side="left")

    # Neighbors: ins-1 (left of insertion) and ins (right of insertion).
    left_idx = np.clip(ins - 1, 0, M - 1)
    right_idx = np.clip(ins, 0, M - 1)

    left_delta = candidate_ns[left_idx] - master_ns
    right_delta = candidate_ns[right_idx] - master_ns

    # Pick whichever neighbor has smaller |delta|. When ins==0 only
    # right is valid; when ins==M only left is valid; clip handles
    # both implicitly by setting both neighbors to the boundary index,
    # so the |delta| comparison degenerates correctly.
    pick_left = np.abs(left_delta) < np.abs(right_delta)
    pick_left &= (ins > 0)          # left is invalid at ins==0
    # When ins==M, both left_idx and right_idx point to M-1; pick_left
    # ends up False and we use the (correct) left neighbor. To be
    # explicit, when ins==M force pick_left=True.
    pick_left |= (ins == M)

    chosen_idx = np.where(pick_left, left_idx, right_idx).astype(np.int32)
    chosen_delta = np.where(pick_left, left_delta, right_delta)

    # Tolerance filter — anything outside tol_ns is -1.
    out_of_tol = np.abs(chosen_delta) > int(tol_ns)
    chosen_idx[out_of_tol] = -1
    chosen_delta_i32 = np.where(out_of_tol, 0, chosen_delta).astype(np.int32, copy=False)

    return chosen_idx, chosen_delta_i32
