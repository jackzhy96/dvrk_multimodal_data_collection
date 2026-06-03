from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from dvrk_data_processing.surgsync.ingest.timestamps import (
    load_timestamps,
    NULL_TS,
)


REPO = Path(__file__).resolve().parents[3]
ONLINE = REPO / "data" / "online_data" / "2" / "time_syn"
OFFLINE = REPO / "data" / "offline_data" / "3" / "time_syn"


@pytest.mark.skipif(not ONLINE.exists(), reason="sample data not present")
def test_online_master_ns_strictly_increasing():
    ts = load_timestamps(ONLINE)
    # Sanity: 886 frames present, master clock non-null for all.
    assert len(ts.master_ns) > 0
    assert not np.any(ts.master_ns == NULL_TS), "master clock has NULLs — should never happen"
    diffs = np.diff(ts.master_ns)
    assert (diffs > 0).all(), "master_ns not strictly increasing"


@pytest.mark.skipif(not ONLINE.exists(), reason="sample data not present")
def test_online_setpoint_cp_ns_populated():
    """Online recorder publishes setpoint_cp stamps for both PSMs.
    `topic_stamps` is dict-keyed by canonical topic name (see
    `align.topics`)."""
    ts = load_timestamps(ONLINE)
    assert (ts.topic_stamps["PSM1.setpoint_cp"] != NULL_TS).all()
    assert (ts.topic_stamps["PSM2.setpoint_cp"] != NULL_TS).all()


@pytest.mark.skipif(not OFFLINE.exists(), reason="sample data not present")
def test_offline_setpoint_cp_ns_all_null():
    """Offline recorder doesn't publish setpoint_cp_stamp — every
    slot is NULL_TS, and the encoder will write Arrow NULL."""
    ts = load_timestamps(OFFLINE)
    assert (ts.topic_stamps["PSM1.setpoint_cp"] == NULL_TS).all()
    assert (ts.topic_stamps["PSM2.setpoint_cp"] == NULL_TS).all()


@pytest.mark.skipif(not ONLINE.exists(), reason="sample data not present")
def test_topic_catalog_fully_covered():
    """Every topic in the canonical catalog has an array of length N
    in `ts.topic_stamps` — the ingester never drops a topic on the
    floor. Topics with no source stamp in the JSON show up as
    all-NULL_TS arrays."""
    from dvrk_data_processing.surgsync.align.topics import TOPIC_NAMES
    ts = load_timestamps(ONLINE)
    n = len(ts.master_ns)
    assert set(ts.topic_stamps.keys()) == set(TOPIC_NAMES)
    for name, arr in ts.topic_stamps.items():
        assert arr.shape == (n,), f"{name} wrong shape: {arr.shape}"


def test_empty_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_timestamps(tmp_path)
