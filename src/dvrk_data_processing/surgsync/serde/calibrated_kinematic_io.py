"""Per-frame calibrated-kinematic JSON ↔ record.

The preprocessing hand-eye stage emits one JSON per PSM frame at:
    `processed_dir/kinematic_reproject/<PSM>/calibrated_kinematic/<i>.json`

Schema (from `specs/interm_data_spec.md` § calibrated_kinematic):

    {
      "frame": 123,
      "arm_name": "PSM1",
      "measured_cp_calibrated": {"position": [3], "orientation": [4]},
      "setpoint_cp_calibrated": {"position": [3], "orientation": [4]}   # omit on offline
    }

Positions in meters, quaternions xyzw, all in the **left-rectified
camera frame**.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CalibratedKinematicSample:
    """One PSM's calibrated pose at one frame.

    `setpoint_cp_calibrated` is None on the offline recorder (which
    has no Cartesian setpoint to calibrate).
    """
    frame: int
    arm_name: str

    # (position [3], orientation xyzw [4]) tuples — None when the
    # whole block is missing.
    measured_cp_calibrated: Optional[tuple[list[float], list[float]]] = None
    setpoint_cp_calibrated: Optional[tuple[list[float], list[float]]] = None


def _pair_from_block(block: Optional[dict]) -> Optional[tuple[list[float], list[float]]]:
    if block is None:
        return None
    pos = block.get("position")
    orient = block.get("orientation")
    if pos is None or orient is None:
        return None
    return ([float(x) for x in pos], [float(x) for x in orient])


def load_calibrated_frame(json_path: Path, arm_name: str, frame_idx: int) -> CalibratedKinematicSample:
    """Parse one `calibrated_kinematic/<frame>.json` into a sample.

    Raises FileNotFoundError if the file is missing. Tolerates the
    offline-omits-`setpoint_cp_calibrated` case (returns None for that
    field).
    """
    with open(json_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"calibrated_kinematic JSON must be an object at {json_path}")
    # The JSON's "frame" / "arm_name" fields are advisory — we trust the
    # caller-supplied identity, since the file naming is the source of truth.
    return CalibratedKinematicSample(
        frame=frame_idx,
        arm_name=arm_name,
        measured_cp_calibrated=_pair_from_block(payload.get("measured_cp_calibrated")),
        setpoint_cp_calibrated=_pair_from_block(payload.get("setpoint_cp_calibrated")),
    )


def calibrated_sample_to_dict(sample: CalibratedKinematicSample) -> dict:
    """Inverse for unpack — produce a dict ready for json.dump.

    `setpoint_cp_calibrated` key is **omitted** (not present-but-null)
    when the field is None, matching the offline-recorder semantics in
    the spec.
    """
    out: dict = {"frame": sample.frame, "arm_name": sample.arm_name}
    if sample.measured_cp_calibrated is not None:
        pos, orient = sample.measured_cp_calibrated
        out["measured_cp_calibrated"] = {"position": pos, "orientation": orient}
    if sample.setpoint_cp_calibrated is not None:
        pos, orient = sample.setpoint_cp_calibrated
        out["setpoint_cp_calibrated"] = {"position": pos, "orientation": orient}
    return out
