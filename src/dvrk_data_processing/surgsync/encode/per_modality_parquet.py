"""Per-modality parquet writers — one parquet per arm / annotation /
timestamp instead of the old monolithic `frames.parquet`.

Each writer takes the same `AlignedClip` and projects its columns into
one of:
- `<episode>/timestamp.parquet`   — master clock + image deltas + contiguity
- `<episode>/ECM.parquet`         — ECM kinematics (every JSON field)
- `<episode>/PSM1.parquet`        — PSM1 kinematics + jaw + freq + setpoint_cp
- `<episode>/PSM2.parquet`        — PSM2 kinematics, same shape as PSM1
- `<episode>/annotation.parquet`  — contact/gesture/phase/step, aligned to stereo-left timestamp

All five parquets have the same row count (`N` = master timeline length)
and the same `frame_index` values 0..N-1, so consumers join them by row
position (or by the explicit `frame_index` column).

Parquet write options: row group size 8192, Zstd compression level 3 —
mirrors the per-`code_design.md` § 2.1 defaults from the prior
monolithic encoder.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.align.aligner import AlignedClip, NULL_DELTA
from dvrk_data_processing.surgsync.schema import (
    build_timestamp_schema,
    build_ecm_schema,
    build_psm_schema,
    build_annotation_schema,
)
from dvrk_data_processing.surgsync.serde.kinematic_io import (
    KinematicSample, CartesianSnapshot, TwistSnapshot, JointSnapshot,
)
from dvrk_data_processing.surgsync.serde.annotation_io import AnnotationSample
from dvrk_data_processing.surgsync.serde.workflow_text import (
    verbalize_phase, verbalize_step, verbalize_gesture,
)


log = logging.getLogger(__name__)


_PARQUET_OPTS = dict(compression="zstd", compression_level=3, row_group_size=8192)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _delta_to_arrow(arr: np.ndarray) -> pa.Array:
    """int64-with-NULL_DELTA → int32 Arrow with proper nulls."""
    mask = arr == NULL_DELTA
    safe = np.where(mask, 0, arr).astype(np.int32)
    return pa.array(safe, type=pa.int32(), mask=mask)


def _opt_list_f32(values: list) -> pa.Array:
    """list<float32> tolerating None entries (→ Arrow NULL row)."""
    return pa.array(values, type=pa.list_(pa.float32()))


def _opt_f32_scalar(values: list) -> pa.Array:
    return pa.array(values, type=pa.float32())


def _project_to_schema(
    cols: dict[str, pa.Array],
    schema: pa.Schema,
    n: int,
) -> pa.Table:
    """Build a pa.Table from a dict of arrays, in the schema's column
    order. Missing columns are filled with all-NULL of the right type
    (defensive — shouldn't happen but keeps the writer robust)."""
    arrays: list[pa.Array] = []
    for field in schema:
        if field.name in cols:
            arrays.append(cols[field.name])
        else:
            log.warning("missing column %s in projection; writing all-NULL", field.name)
            arrays.append(pa.nulls(n, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Kinematic column factories (shared between ECM and PSM)
# ---------------------------------------------------------------------------

def _kin_cartesian_cols(
    arm_samples: list[Optional[KinematicSample]],
    extract,        # callable: sample → Optional[CartesianSnapshot]
    prefix: str,
    *,
    include_velocity: bool,
) -> dict[str, pa.Array]:
    """Build position / orientation [/ velocity] columns for one
    Cartesian snapshot kind (local_measured_cp / measured_cp /
    setpoint_cp / *_calibrated)."""
    positions: list = []
    orientations: list = []
    velocities: list = []
    for s in arm_samples:
        snap = None if s is None else extract(s)
        if snap is None:
            positions.append(None)
            orientations.append(None)
            velocities.append(None)
        else:
            positions.append(snap.position)
            orientations.append(snap.orientation)
            velocities.append(snap.velocity)
    out = {
        f"{prefix}.position":    _opt_list_f32(positions),
        f"{prefix}.orientation": _opt_list_f32(orientations),
    }
    if include_velocity:
        out[f"{prefix}.velocity"] = _opt_list_f32(velocities)
    return out


def _kin_joint_cols(
    arm_samples: list[Optional[KinematicSample]],
    extract,        # callable: sample → JointSnapshot
    prefix: str,
) -> dict[str, pa.Array]:
    positions, velocities, efforts = [], [], []
    for s in arm_samples:
        snap = None if s is None else extract(s)
        if snap is None:
            positions.append(None); velocities.append(None); efforts.append(None)
            continue
        positions.append(snap.position)
        velocities.append(snap.velocity)
        efforts.append(snap.effort)
    return {
        f"{prefix}.position": _opt_list_f32(positions),
        f"{prefix}.velocity": _opt_list_f32(velocities),
        f"{prefix}.effort":   _opt_list_f32(efforts),
    }


def _kin_twist_cols(
    arm_samples: list[Optional[KinematicSample]],
    extract,        # callable: sample → TwistSnapshot
    prefix: str,
) -> dict[str, pa.Array]:
    linear, angular = [], []
    for s in arm_samples:
        snap = None if s is None else extract(s)
        if snap is None:
            linear.append(None); angular.append(None)
            continue
        linear.append(snap.linear)
        angular.append(snap.angular)
    return {
        f"{prefix}.linear":  _opt_list_f32(linear),
        f"{prefix}.angular": _opt_list_f32(angular),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_timestamp_parquet(aligned: AlignedClip, dst_path: Path) -> None:
    """Write `<episode>/timestamp.parquet`.

    Master clock + contiguity flags + **one `delta_to_master.<topic>_ns`
    column per topic in `align.topics.TIMESTAMP_TOPICS`**. The arm
    parquets (ECM, PSM1, PSM2) no longer carry delta columns —
    everything is centralized here.
    """
    n = aligned.n_frames
    cols: dict[str, pa.Array] = {
        "frame_index":           pa.array(np.arange(n, dtype=np.int32)),
        "source_frame_index":    pa.array(aligned.source_frame_index, type=pa.int32()),
        "master_timestamp_ns":   pa.array(aligned.master_ns, type=pa.int64()),
        "is_contiguous_to_prev": pa.array(aligned.is_contiguous_to_prev, type=pa.bool_()),
        "drop_count_since_prev": pa.array(aligned.drop_count_since_prev, type=pa.int8()),
    }
    # Every topic in the catalog → one delta column. Order doesn't
    # matter at write time (the schema is the source of truth); the
    # projection step reorders.
    for topic_name, signed_delta in aligned.topic_deltas.items():
        cols[f"delta_to_master.{topic_name}_ns"] = _delta_to_arrow(signed_delta)
    table = _project_to_schema(cols, build_timestamp_schema(), n)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path, **_PARQUET_OPTS)


def write_ecm_parquet(aligned: AlignedClip, dst_path: Path) -> None:
    """Write `<episode>/ECM.parquet` — every field from `kinematic/ECM/<i>.json`."""
    n = aligned.n_frames
    samples = aligned.ECM_aligned
    cols: dict[str, pa.Array] = {
        "frame_index":         pa.array(np.arange(n, dtype=np.int32)),
        "master_timestamp_ns": pa.array(aligned.master_ns, type=pa.int64()),
    }
    cols.update(_kin_cartesian_cols(samples, lambda s: s.local_measured_cp, "local_measured_cp",
                                    include_velocity=False))
    cols.update(_kin_cartesian_cols(samples, lambda s: s.measured_cp, "measured_cp",
                                    include_velocity=True))
    cols.update(_kin_twist_cols(samples, lambda s: s.measured_cv, "measured_cv"))
    cols.update(_kin_joint_cols(samples, lambda s: s.measured_js, "measured_js"))
    cols.update(_kin_joint_cols(samples, lambda s: s.setpoint_js, "setpoint_js"))

    table = _project_to_schema(cols, build_ecm_schema(), n)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path, **_PARQUET_OPTS)


def write_psm_parquet(
    aligned: AlignedClip,
    arm: str,                  # "PSM1" | "PSM2"
    dst_path: Path,
    *,
    calibrated_measured: Optional[list] = None,
    calibrated_setpoint: Optional[list] = None,
) -> None:
    """Write `<episode>/<arm>.parquet` — every field from `kinematic/<arm>/<i>.json`
    plus jaw + measured_frequency + (online-only) setpoint_cp.

    `calibrated_measured` and `calibrated_setpoint` are optional per-frame
    `(position, orientation) | None` lists from preprocessing's `calibrated_kinematic/`
    JSON outputs. Currently always passed as None — the ingester for
    those files is in the deferred list (`tasks/data_format.md` § 6).
    """
    if arm == "PSM1":
        samples = aligned.PSM1_aligned
    elif arm == "PSM2":
        samples = aligned.PSM2_aligned
    else:
        raise ValueError(f"unknown PSM arm: {arm!r}")

    n = aligned.n_frames

    cols: dict[str, pa.Array] = {
        "frame_index":         pa.array(np.arange(n, dtype=np.int32)),
        "master_timestamp_ns": pa.array(aligned.master_ns, type=pa.int64()),
    }
    cols.update(_kin_cartesian_cols(samples, lambda s: s.local_measured_cp,
                                    "local_measured_cp", include_velocity=False))
    cols.update(_kin_cartesian_cols(samples, lambda s: s.measured_cp,
                                    "measured_cp", include_velocity=True))
    cols.update(_kin_twist_cols(samples, lambda s: s.measured_cv, "measured_cv"))
    cols.update(_kin_joint_cols(samples, lambda s: s.measured_js, "measured_js"))
    cols.update(_kin_joint_cols(samples, lambda s: s.setpoint_js, "setpoint_js"))
    cols.update(_kin_cartesian_cols(samples, lambda s: s.setpoint_cp,
                                    "setpoint_cp", include_velocity=False))

    # measured_cp_calibrated / setpoint_cp_calibrated — currently NULL.
    cm_pos, cm_orient = [], []
    cs_pos, cs_orient = [], []
    for i in range(n):
        if calibrated_measured and calibrated_measured[i] is not None:
            cm_pos.append(calibrated_measured[i][0]); cm_orient.append(calibrated_measured[i][1])
        else:
            cm_pos.append(None); cm_orient.append(None)
        if calibrated_setpoint and calibrated_setpoint[i] is not None:
            cs_pos.append(calibrated_setpoint[i][0]); cs_orient.append(calibrated_setpoint[i][1])
        else:
            cs_pos.append(None); cs_orient.append(None)
    cols["measured_cp_calibrated.position"]    = _opt_list_f32(cm_pos)
    cols["measured_cp_calibrated.orientation"] = _opt_list_f32(cm_orient)
    cols["setpoint_cp_calibrated.position"]    = _opt_list_f32(cs_pos)
    cols["setpoint_cp_calibrated.orientation"] = _opt_list_f32(cs_orient)

    # Jaw + provenance
    jaw_meas, jaw_set, freq = [], [], []
    for s in samples:
        if s is None:
            jaw_meas.append(None); jaw_set.append(None); freq.append(None); continue
        jaw_meas.append(s.measured_jaw_position)
        jaw_set.append(s.setpoint_jaw_position)
        freq.append(s.source_frequency_hz)
    cols["jaw.measured_position"]  = _opt_f32_scalar(jaw_meas)
    cols["jaw.setpoint_position"]  = _opt_f32_scalar(jaw_set)
    cols["source_frequency_hz"]    = _opt_f32_scalar(freq)

    table = _project_to_schema(cols, build_psm_schema(), n)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path, **_PARQUET_OPTS)


def write_annotation_parquet(
    aligned: AlignedClip,
    dst_path: Path,
    *,
    task: Optional[str] = None,
) -> None:
    """Write `<episode>/annotation.parquet` — annotation rows aligned to
    the stereo-left image timestamp.

    The `phase`, `step`, and `gesture.PSM*` columns store the **text
    description** for each annotation id (looked up via
    `serde/workflow_text.py`), not the raw numeric id. The mapping for
    gestures is task-specific (suturing vs. dissection); pass the
    clip's task name so the writer picks the right gesture table.
    Unknown ids pass through unchanged so the column stays populated.
    """
    n = aligned.n_frames

    contact_p1, contact_p2 = [], []
    gesture_p1, gesture_p2 = [], []
    phase, step = [], []
    for a in aligned.annotations:
        if a is None:
            contact_p1.append(None); contact_p2.append(None)
            gesture_p1.append(None); gesture_p2.append(None)
            phase.append(None); step.append(None)
            continue
        contact_p1.append(a.contact_PSM1)
        contact_p2.append(a.contact_PSM2)
        # Verbalize at write time — see the module docstring for the
        # rationale. The clip's task disambiguates gesture vocabularies
        # (suturing vs. dissection). Step + phase are task-agnostic
        # here for backward compatibility with the v1.0 release; the
        # task-aware step lookup wired into `verbalize_step` is
        # available but not yet engaged in the packer.
        gesture_p1.append(verbalize_gesture(a.gesture_PSM1, task))
        gesture_p2.append(verbalize_gesture(a.gesture_PSM2, task))
        phase.append(verbalize_phase(a.phase))
        step.append(verbalize_step(a.step))

    cols = {
        "frame_index":         pa.array(np.arange(n, dtype=np.int32)),
        "master_timestamp_ns": pa.array(aligned.master_ns, type=pa.int64()),
        "contact.PSM1": pa.array(contact_p1, type=pa.int8()),
        "contact.PSM2": pa.array(contact_p2, type=pa.int8()),
        "gesture.PSM1": pa.array(gesture_p1, type=pa.string()),
        "gesture.PSM2": pa.array(gesture_p2, type=pa.string()),
        "phase":        pa.array(phase, type=pa.string()),
        "step":         pa.array(step,  type=pa.string()),
    }

    table = _project_to_schema(cols, build_annotation_schema(), n)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path, **_PARQUET_OPTS)


# ---------------------------------------------------------------------------
# Convenience — write all five at once
# ---------------------------------------------------------------------------

def write_all_per_modality(
    aligned: AlignedClip,
    episode_dir: Path,
    *,
    task: Optional[str] = None,
    calibrated_psm1_measured: Optional[list] = None,
    calibrated_psm1_setpoint: Optional[list] = None,
    calibrated_psm2_measured: Optional[list] = None,
    calibrated_psm2_setpoint: Optional[list] = None,
) -> None:
    """Write all five per-modality parquets under `episode_dir`.

    `task` is forwarded to the annotation writer so it can pick the
    right gesture vocabulary (suturing gestures use a different table
    than dissection gestures). Without it, gesture cells fall back to
    the raw id.
    """
    episode_dir = Path(episode_dir)
    write_timestamp_parquet(aligned, episode_dir / "timestamp.parquet")
    write_ecm_parquet(aligned, episode_dir / "ECM.parquet")
    write_psm_parquet(aligned, "PSM1", episode_dir / "PSM1.parquet",
                      calibrated_measured=calibrated_psm1_measured,
                      calibrated_setpoint=calibrated_psm1_setpoint)
    write_psm_parquet(aligned, "PSM2", episode_dir / "PSM2.parquet",
                      calibrated_measured=calibrated_psm2_measured,
                      calibrated_setpoint=calibrated_psm2_setpoint)
    write_annotation_parquet(aligned, episode_dir / "annotation.parquet",
                             task=task)
