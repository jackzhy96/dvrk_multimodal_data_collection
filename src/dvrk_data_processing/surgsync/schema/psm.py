"""pyarrow.Schema for `<episode>/PSM{1,2}.parquet`.

One row per master-clock frame. Superset of the ECM schema, with jaw,
`measured_cp_calibrated`, `setpoint_cp` (online-only), `setpoint_cp_calibrated`
(online-only), and `source_frequency_hz` (the publish rate of the ROS
topic at recording time).

PSM1 and PSM2 share this schema. The arm identity is encoded in the
filename (PSM1.parquet vs PSM2.parquet); we don't add an `arm` column
to keep the schema compact.

`*_calibrated` columns are populated when the preprocessing calibrated_kinematic
JSON output is ingested (currently deferred — see `tasks/data_format.md`
§ 6 Deferred items).
"""
from __future__ import annotations
import pyarrow as pa


def _list_f32() -> pa.DataType:
    return pa.list_(pa.float32())


def build_psm_schema() -> pa.Schema:
    return pa.schema([
        # Identity & sync — per-topic deltas live in timestamp.parquet
        # only (`delta_to_master.PSM{1,2}.<topic>_ns`); they're not
        # duplicated here.
        pa.field("frame_index",                                pa.int32(),   nullable=False),
        pa.field("master_timestamp_ns",                        pa.int64(),   nullable=False),
        # local_measured_cp — pose in PSM base frame
        pa.field("local_measured_cp.position",                 _list_f32(),  nullable=True),
        pa.field("local_measured_cp.orientation",              _list_f32(),  nullable=True),
        # measured_cp — pose in ROS world frame, with twist
        pa.field("measured_cp.position",                       _list_f32(),  nullable=True),
        pa.field("measured_cp.orientation",                    _list_f32(),  nullable=True),
        pa.field("measured_cp.velocity",                       _list_f32(),  nullable=True),  # 6-twist
        # measured_cv — split-form twist
        pa.field("measured_cv.linear",                         _list_f32(),  nullable=True),
        pa.field("measured_cv.angular",                        _list_f32(),  nullable=True),
        # measured_cp_calibrated — pose in left-rectified camera frame
        # (currently NULL — preprocessing calibrated_kinematic ingestion deferred)
        pa.field("measured_cp_calibrated.position",            _list_f32(),  nullable=True),
        pa.field("measured_cp_calibrated.orientation",         _list_f32(),  nullable=True),
        # measured_js — 6-DOF joints
        pa.field("measured_js.position",                       _list_f32(),  nullable=True),
        pa.field("measured_js.velocity",                       _list_f32(),  nullable=True),
        pa.field("measured_js.effort",                         _list_f32(),  nullable=True),
        # setpoint_js — 6-DOF commands
        pa.field("setpoint_js.position",                       _list_f32(),  nullable=True),
        pa.field("setpoint_js.velocity",                       _list_f32(),  nullable=True),
        pa.field("setpoint_js.effort",                         _list_f32(),  nullable=True),
        # setpoint_cp — Cartesian commands (online recorder only — NULL on offline)
        pa.field("setpoint_cp.position",                       _list_f32(),  nullable=True),
        pa.field("setpoint_cp.orientation",                    _list_f32(),  nullable=True),
        # setpoint_cp_calibrated — (deferred — currently NULL)
        pa.field("setpoint_cp_calibrated.position",            _list_f32(),  nullable=True),
        pa.field("setpoint_cp_calibrated.orientation",         _list_f32(),  nullable=True),
        # Jaw — PSM-only single-DOF gripper
        pa.field("jaw.measured_position",                      pa.float32(), nullable=True),
        pa.field("jaw.setpoint_position",                      pa.float32(), nullable=True),
        # Provenance
        pa.field("source_frequency_hz",                        pa.float32(), nullable=True),
    ])
