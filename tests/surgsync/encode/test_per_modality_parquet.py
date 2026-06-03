"""Per-modality parquet writer tests — verify each parquet's schema +
row count + a few column projections against real sample data."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from dvrk_data_processing.surgsync.align.aligner import align_clip
from dvrk_data_processing.surgsync.align.policy import TolerancePolicy
from dvrk_data_processing.surgsync.encode.per_modality_parquet import (
    write_all_per_modality,
    write_ecm_parquet, write_psm_parquet,
    write_annotation_parquet, write_timestamp_parquet,
)
from dvrk_data_processing.surgsync.ingest.annotations import load_annotations
from dvrk_data_processing.surgsync.ingest.kinematics import load_arm
from dvrk_data_processing.surgsync.ingest.timestamps import load_timestamps


REPO = Path(__file__).resolve().parents[3]
ONLINE_RAW = REPO / "data" / "online_data" / "2"


def _policy():
    return TolerancePolicy.for_variant(
        "online",
        tol_ms_image_right=2.0, tol_ms_image_side=33.0, tol_ms_kinematic=100.0,
    )


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_write_all_per_modality_against_sample(tmp_path: Path):
    ts = load_timestamps(ONLINE_RAW / "time_syn")
    kin_ECM  = load_arm(ONLINE_RAW / "kinematic", "ECM")
    kin_PSM1 = load_arm(ONLINE_RAW / "kinematic", "PSM1")
    kin_PSM2 = load_arm(ONLINE_RAW / "kinematic", "PSM2")
    ann = load_annotations(ONLINE_RAW / "annotation")
    aligned = align_clip(
        ts, kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann, policy=_policy(),
    )

    write_all_per_modality(aligned, tmp_path)

    # Every parquet present + row count matches.
    for name in ("timestamp.parquet", "ECM.parquet", "PSM1.parquet",
                 "PSM2.parquet", "annotation.parquet"):
        path = tmp_path / name
        assert path.exists(), name
        t = pq.read_table(path)
        assert t.num_rows == aligned.n_frames, name

    # timestamp.parquet carries master clock + image deltas.
    ts_tab = pq.read_table(tmp_path / "timestamp.parquet")
    assert "master_timestamp_ns" in ts_tab.column_names
    assert ts_tab.column("master_timestamp_ns").null_count == 0
    # `master_timestamp_ns` is now clip-relative: row 0 is exactly 0,
    # later rows are positive ns since clip start. Absolute timestamps
    # are recoverable via the `master_t0_ns` field in episode_meta.json.
    master_col = ts_tab.column("master_timestamp_ns").to_numpy()
    assert int(master_col[0]) == 0
    assert (master_col[1:] > 0).all()
    # Same column must show up identically across every per-modality
    # parquet so cross-table joins on master_timestamp_ns work.
    for name in ("ECM.parquet", "PSM1.parquet", "PSM2.parquet",
                 "annotation.parquet"):
        other = pq.read_table(tmp_path / name).column("master_timestamp_ns").to_numpy()
        assert (other == master_col).all(), name

    # ECM.parquet has every JSON field.
    ecm_tab = pq.read_table(tmp_path / "ECM.parquet")
    for col in ("local_measured_cp.position", "measured_cp.velocity",
                "measured_cv.linear", "measured_cv.angular",
                "measured_js.position", "setpoint_js.position"):
        assert col in ecm_tab.column_names, col

    # PSM1.parquet has jaw + freq + setpoint_cp (online → populated).
    psm1_tab = pq.read_table(tmp_path / "PSM1.parquet")
    for col in ("jaw.measured_position", "jaw.setpoint_position",
                "source_frequency_hz",
                "setpoint_cp.position", "setpoint_cp.orientation",
                "measured_cp.velocity", "measured_cv.linear"):
        assert col in psm1_tab.column_names, col
    # Online recorder → setpoint_cp populated for most frames.
    assert psm1_tab.column("setpoint_cp.position").null_count < aligned.n_frames

    # annotation.parquet has master_timestamp_ns aligned to stereo-left.
    ann_tab = pq.read_table(tmp_path / "annotation.parquet")
    assert "master_timestamp_ns" in ann_tab.column_names
    # Same master_timestamps as timestamp.parquet (row-aligned).
    assert (ann_tab.column("master_timestamp_ns").to_numpy()
            == ts_tab.column("master_timestamp_ns").to_numpy()).all()


@pytest.mark.skipif(not ONLINE_RAW.exists(), reason="sample data not present")
def test_psm1_measured_cp_velocity_populated(tmp_path: Path):
    """measured_cp.velocity (the 6-twist) should be populated for at
    least frame 0 on online_data/2 — the source JSON has this field."""
    ts = load_timestamps(ONLINE_RAW / "time_syn")
    kin_ECM  = load_arm(ONLINE_RAW / "kinematic", "ECM")
    kin_PSM1 = load_arm(ONLINE_RAW / "kinematic", "PSM1")
    kin_PSM2 = load_arm(ONLINE_RAW / "kinematic", "PSM2")
    ann = load_annotations(ONLINE_RAW / "annotation")
    aligned = align_clip(
        ts, kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann, policy=_policy(),
    )
    write_psm_parquet(aligned, "PSM1", tmp_path / "PSM1.parquet")
    t = pq.read_table(tmp_path / "PSM1.parquet")
    vel = t.column("measured_cp.velocity")[0].as_py()
    assert vel is not None
    assert len(vel) == 6   # 6-twist [vx, vy, vz, ωx, ωy, ωz]


@pytest.mark.skipif(not (REPO / "data" / "offline_data" / "3").exists(),
                    reason="offline sample not present")
def test_offline_psm_setpoint_cp_is_null(tmp_path: Path):
    """Offline recorder has no setpoint_cp — `setpoint_cp.position`
    column should be all-NULL after packing."""
    OFFLINE = REPO / "data" / "offline_data" / "3"
    ts = load_timestamps(OFFLINE / "time_syn")
    kin_ECM  = load_arm(OFFLINE / "kinematic", "ECM")
    kin_PSM1 = load_arm(OFFLINE / "kinematic", "PSM1")
    kin_PSM2 = load_arm(OFFLINE / "kinematic", "PSM2")
    ann = load_annotations(OFFLINE / "annotation")
    aligned = align_clip(
        ts, kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann,
        policy=TolerancePolicy.for_variant(
            "offline",
            tol_ms_image_right=2.0, tol_ms_image_side=33.0, tol_ms_kinematic=100.0,
        ),
    )
    write_psm_parquet(aligned, "PSM1", tmp_path / "PSM1.parquet")
    t = pq.read_table(tmp_path / "PSM1.parquet")
    assert t.column("setpoint_cp.position").null_count == aligned.n_frames
    assert t.column("setpoint_cp.orientation").null_count == aligned.n_frames
