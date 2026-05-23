from __future__ import annotations
import numpy as np

from dvrk_data_processing.surgsync.align.contiguity import detect_contiguity


def test_uniform_timeline_has_no_drops():
    master = (np.arange(10) * 33_333_333).astype(np.int64)
    is_contig, drops = detect_contiguity(master)
    assert not is_contig[0]   # first frame has no prev
    assert is_contig[1:].all()
    assert (drops == 0).all()


def test_single_drop_flagged():
    master = (np.arange(10) * 33_333_333).astype(np.int64)
    # Skip one frame at index 5 — the gap is 2 * period.
    master[5:] += 33_333_333
    is_contig, drops = detect_contiguity(master)
    assert not is_contig[5]
    assert drops[5] == 1
    # Other frames still contiguous.
    assert is_contig[1:5].all()
    assert is_contig[6:].all()


def test_empty_timeline():
    is_contig, drops = detect_contiguity(np.zeros(0, dtype=np.int64))
    assert is_contig.shape == (0,)
    assert drops.shape == (0,)


def test_single_frame_timeline():
    is_contig, drops = detect_contiguity(np.array([0], dtype=np.int64))
    assert not is_contig[0]
    assert drops[0] == 0
