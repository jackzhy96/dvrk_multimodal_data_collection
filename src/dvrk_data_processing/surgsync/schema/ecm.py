"""pyarrow.Schema for `<episode>/ECM.parquet`.

One row per master-clock frame, carrying every field that the raw
`kinematic/ECM/<frame>.json` provides. ECM has 4 joints, no jaw, and
no `setpoint_cp` (the ECM publishes setpoints in joint space only).

`master_timestamp_ns` is duplicated here (also in `timestamp.parquet`)
so this file is self-contained for time-aware analysis. The
per-topic `delta_to_master.*_ns` columns live in `timestamp.parquet`
only — no longer carried in arm parquets.
"""
from __future__ import annotations
import pyarrow as pa


def _list_f32() -> pa.DataType:
    return pa.list_(pa.float32())


def build_ecm_schema() -> pa.Schema:
    return pa.schema([
        # Identity & sync
        pa.field("frame_index",                                pa.int32(),   nullable=False),
        pa.field("master_timestamp_ns",                        pa.int64(),   nullable=False),
        # local_measured_cp — pose in ECM base frame
        pa.field("local_measured_cp.position",                 _list_f32(),  nullable=True),
        pa.field("local_measured_cp.orientation",              _list_f32(),  nullable=True),
        # measured_cp — pose in ROS world frame, with twist velocity
        pa.field("measured_cp.position",                       _list_f32(),  nullable=True),
        pa.field("measured_cp.orientation",                    _list_f32(),  nullable=True),
        pa.field("measured_cp.velocity",                       _list_f32(),  nullable=True),  # 6-twist [vx,vy,vz,ωx,ωy,ωz]
        # measured_cv — split-form twist
        pa.field("measured_cv.linear",                         _list_f32(),  nullable=True),
        pa.field("measured_cv.angular",                        _list_f32(),  nullable=True),
        # measured_js — joint-space state
        pa.field("measured_js.position",                       _list_f32(),  nullable=True),
        pa.field("measured_js.velocity",                       _list_f32(),  nullable=True),
        pa.field("measured_js.effort",                         _list_f32(),  nullable=True),
        # setpoint_js — joint-space command
        pa.field("setpoint_js.position",                       _list_f32(),  nullable=True),
        pa.field("setpoint_js.velocity",                       _list_f32(),  nullable=True),
        pa.field("setpoint_js.effort",                         _list_f32(),  nullable=True),
    ])
