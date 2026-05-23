"""Synthetic tests for the nearest-within-tolerance matcher.

Includes the jitter fuzzing acceptance test.
"""
from __future__ import annotations

import numpy as np
import pytest

from dvrk_data_processing.surgsync.align.matcher import match_modality


def test_perfect_match_zero_delta():
    """When candidate == master, every delta is zero."""
    master = np.arange(0, 10) * 33_333_333
    candidate = master.copy()
    indices, deltas = match_modality(master, candidate, tol_ns=1_000_000)
    assert (indices == np.arange(10)).all()
    assert (deltas == 0).all()


def test_out_of_tolerance_returns_minus_one():
    """A candidate 10 ms away with 2 ms tolerance → -1."""
    master = np.array([0, 1_000_000_000], dtype=np.int64)
    candidate = np.array([10_000_000, 1_010_000_000], dtype=np.int64)  # +10 ms shift
    indices, deltas = match_modality(master, candidate, tol_ns=2_000_000)
    assert (indices == -1).all()


def test_picks_nearest_neighbor():
    """Master falls between two candidates; the closer one wins."""
    master = np.array([100], dtype=np.int64)
    candidate = np.array([90, 105], dtype=np.int64)
    indices, deltas = match_modality(master, candidate, tol_ns=100)
    # |105-100|=5 < |90-100|=10
    assert indices[0] == 1
    assert deltas[0] == 5


def test_empty_candidate_returns_minus_one_array():
    master = np.array([100, 200], dtype=np.int64)
    indices, deltas = match_modality(master, np.zeros(0, dtype=np.int64), tol_ns=100)
    assert (indices == -1).all()
    assert deltas.shape == (2,)


def test_jitter_fuzzing_preserves_alignment():
    """With ±0.5 ms jitter on a 30 Hz signal at 2 ms tolerance, every
    master frame should still find its match."""
    rng = np.random.default_rng(seed=0)
    N = 200
    period_ns = 33_333_333   # 30 Hz
    master = (np.arange(N) * period_ns).astype(np.int64)
    jitter = rng.integers(-500_000, 500_001, size=N, dtype=np.int64)
    candidate = master + jitter
    indices, deltas = match_modality(master, candidate, tol_ns=2_000_000)
    assert (indices >= 0).all(), "All matches should be in-tolerance with ±0.5 ms jitter at 2 ms tol"
    # Indices are monotonic across master since the candidate is just
    # jittered around master itself.
    diffs = np.diff(indices.astype(np.int64))
    assert (diffs >= 0).all()


def test_jitter_fuzzing_with_drops():
    """Injecting 5 drops into a candidate array — those master frames
    should fall back to neighbors; some may go out of tolerance."""
    rng = np.random.default_rng(seed=1)
    N = 100
    period_ns = 33_333_333
    master = (np.arange(N) * period_ns).astype(np.int64)
    candidate = master.copy()
    # Drop 5 random candidate entries — the rest must still align.
    drop_idx = rng.choice(N, size=5, replace=False)
    candidate = np.delete(candidate, drop_idx)
    indices, deltas = match_modality(master, candidate, tol_ns=1_000_000)
    # Master frames at the drop positions will pick a neighbor; if the
    # neighbor is >1ms away the index goes -1.
    # We expect at least 90% of frames still match (with 1 ms tolerance
    # on 33 ms grid, only the dropped frame itself goes out of tol).
    matched = int((indices >= 0).sum())
    assert matched >= N - 5
