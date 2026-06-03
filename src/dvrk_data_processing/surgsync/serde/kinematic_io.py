"""Per-arm kinematic JSON ↔ in-memory record.

Captures **every** field that the raw `kinematic/<arm>/<frame>.json`
carries. The on-disk JSON layout (which differs slightly from the
schema diagram) is:

    [{
      "arm": {
        "measured_data": {
          "local_measured_cp":  {"position": [3], "orientation": [4]},
          "measured_cp":        {"position": [3], "orientation": [4],
                                  "velocity": [vx, vy, vz, ωx, ωy, ωz]},  # 6-twist
          "measured_cv":        {"linear":   [3], "angular":     [3]},    # split twist
          "measured_js":        {"position": [N], "velocity": [N], "effort": [N]},
        },
        "setpoint_data": {
          "setpoint_js":        {"position": [N], "velocity": [N], "effort": [N]},
          "setpoint_cp":        {"position": [3], "orientation": [4]},    # online only
        }
      },
      # PSM-only siblings of "arm":
      "jaw": {"measured_data": {"position": [1]}, "setpoint_data": {"position": [1]}},
      "measured_frequency": <Hz>,
    }]

Reader tolerances:
- Offline arms have no `setpoint_cp` (Cartesian setpoint). Online arms do.
- ECM has no jaw block. PSM1/PSM2 do.
- `measured_frequency` is per-arm (PSM) but absent on ECM.

All length values are in meters, angles in radians, quaternions in
xyzw order (dVRK convention). The 6-twist `velocity` order is
`[linear_xyz, angular_xyz]` matching the dVRK CRTK convention.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class JointSnapshot:
    """Joint-space data for one arm at one frame.

    `position` length depends on the arm: 6 for PSMs, 4 for ECM.
    `velocity` and `effort` mirror `position`'s length. A missing block
    in the JSON manifests as None at the snapshot level.
    """
    position: Optional[list[float]] = None
    velocity: Optional[list[float]] = None
    effort: Optional[list[float]] = None


@dataclass
class CartesianSnapshot:
    """Cartesian pose at a single instant.

    `velocity` is the optional 6-twist `[vx, vy, vz, ωx, ωy, ωz]` that
    accompanies `measured_cp` in the raw JSON. `local_measured_cp` and
    `setpoint_cp` typically don't carry a velocity field.
    """
    position: Optional[list[float]] = None       # [x, y, z]
    orientation: Optional[list[float]] = None    # [qx, qy, qz, qw]
    velocity: Optional[list[float]] = None       # [vx, vy, vz, ωx, ωy, ωz]


@dataclass
class TwistSnapshot:
    """Split-form twist (the `measured_cv` block in raw JSON).

    Carries the same information as the 6-twist `velocity` field on
    `measured_cp`, but split into `linear` (3-vec) and `angular` (3-vec)
    for callers that prefer the split form. We preserve both because
    they're both in the source JSON and the conversion isn't loss-free
    in the presence of rounding.
    """
    linear: Optional[list[float]] = None    # [vx, vy, vz]
    angular: Optional[list[float]] = None   # [ωx, ωy, ωz]


@dataclass
class KinematicSample:
    """One arm's kinematic state at one frame. Mirrors every field of
    the raw JSON so the inverse direction is bit-equivalent."""
    frame: int
    arm_name: str             # "ECM" | "PSM1" | "PSM2"

    measured_js: JointSnapshot = field(default_factory=JointSnapshot)
    setpoint_js: JointSnapshot = field(default_factory=JointSnapshot)
    measured_cp: CartesianSnapshot = field(default_factory=CartesianSnapshot)
    local_measured_cp: CartesianSnapshot = field(default_factory=CartesianSnapshot)
    measured_cv: TwistSnapshot = field(default_factory=TwistSnapshot)
    setpoint_cp: Optional[CartesianSnapshot] = None   # None when raw is missing (offline)

    # PSM-only.
    measured_jaw_position: Optional[float] = None
    setpoint_jaw_position: Optional[float] = None
    source_frequency_hz: Optional[float] = None


# ---------------------------------------------------------------------------
# Forward direction — JSON → KinematicSample
# ---------------------------------------------------------------------------

def _quat_to_xyzw(q):
    """Pass-through. dVRK already emits xyzw; convert here if any future
    upstream schema starts emitting wxyz."""
    if q is None:
        return None
    if len(q) != 4:
        raise ValueError(f"quaternion must have length 4, got {len(q)}")
    return list(q)


def _maybe_list(value) -> Optional[list[float]]:
    """Copy a JSON list to a Python list of floats, or return None."""
    if value is None:
        return None
    return [float(x) for x in value]


def _cartesian_from_block(block: Optional[dict]) -> Optional[CartesianSnapshot]:
    """Build a CartesianSnapshot from a raw `*_cp` dict.

    Returns None when the block itself is missing (e.g. offline
    `setpoint_cp`). Recognizes the optional `velocity` 6-twist that
    accompanies `measured_cp`.
    """
    if block is None:
        return None
    return CartesianSnapshot(
        position=_maybe_list(block.get("position")),
        orientation=_quat_to_xyzw(block.get("orientation")),
        velocity=_maybe_list(block.get("velocity")),
    )


def _twist_from_block(block: Optional[dict]) -> TwistSnapshot:
    """Build a TwistSnapshot from a raw `measured_cv` dict."""
    if block is None:
        return TwistSnapshot()
    return TwistSnapshot(
        linear=_maybe_list(block.get("linear")),
        angular=_maybe_list(block.get("angular")),
    )


def _joint_from_block(block: Optional[dict]) -> JointSnapshot:
    if block is None:
        return JointSnapshot()
    return JointSnapshot(
        position=_maybe_list(block.get("position")),
        velocity=_maybe_list(block.get("velocity")),
        effort=_maybe_list(block.get("effort")),
    )


def load_arm_frame_json(json_path: Path, arm_name: str, frame_idx: int) -> KinematicSample:
    """Parse one `kinematic/<arm>/<frame>.json` into a KinematicSample.

    Raises FileNotFoundError if the file is missing. Returns a sample
    with NULL-valued optional blocks for any sub-section the raw JSON
    omits (offline setpoint_cp, ECM jaw, etc.).
    """
    with open(json_path) as f:
        payload = json.load(f)

    # Raw JSON is a 1-element list. The top-level dict carries `arm` plus
    # optional siblings `jaw` and `measured_frequency` (PSM-only; ECM has
    # neither). The on-disk layout differs from the schema diagram,
    # which colocates `jaw` inside `arm` — the
    # real data places jaw/measured_frequency at the top level, so we
    # accept either location for forward-compatibility.
    if not isinstance(payload, list) or not payload or "arm" not in payload[0]:
        raise ValueError(f"Unexpected kinematic JSON shape at {json_path}")
    top = payload[0]
    arm = top["arm"]

    measured = arm.get("measured_data", {})
    setpoint = arm.get("setpoint_data", {})
    # Jaw / frequency may live either at top-level (real data) or inside `arm`
    # (spec diagram). Try both, top-level first since that's what's on disk.
    jaw = top.get("jaw", arm.get("jaw"))
    measured_frequency = top.get("measured_frequency", arm.get("measured_frequency"))

    sample = KinematicSample(
        frame=frame_idx,
        arm_name=arm_name,
        measured_js=_joint_from_block(measured.get("measured_js")),
        setpoint_js=_joint_from_block(setpoint.get("setpoint_js")),
        measured_cp=_cartesian_from_block(measured.get("measured_cp")) or CartesianSnapshot(),
        local_measured_cp=_cartesian_from_block(measured.get("local_measured_cp")) or CartesianSnapshot(),
        measured_cv=_twist_from_block(measured.get("measured_cv")),
        setpoint_cp=_cartesian_from_block(setpoint.get("setpoint_cp")),
        source_frequency_hz=measured_frequency,
    )

    if jaw is not None:
        m_jaw = jaw.get("measured_data", {}).get("position")
        s_jaw = jaw.get("setpoint_data", {}).get("position")
        if isinstance(m_jaw, list) and m_jaw:
            sample.measured_jaw_position = float(m_jaw[0])
        if isinstance(s_jaw, list) and s_jaw:
            sample.setpoint_jaw_position = float(s_jaw[0])

    return sample


# ---------------------------------------------------------------------------
# Inverse direction — KinematicSample → JSON (used by unpack)
# ---------------------------------------------------------------------------

def kinematic_sample_to_raw_dict(
    sample: KinematicSample,
    *,
    jaw_freq_location: str = "top_level",
) -> list[dict]:
    """Reverse of `load_arm_frame_json` — produces the raw `[{arm: ...,
    jaw: ..., measured_frequency: ...}]` structure.

    `jaw_freq_location` controls where the PSM-only `jaw` block and
    `measured_frequency` scalar are emitted:
      - `"top_level"` (default) — siblings of `arm` at the top of the
        outer dict. This matches the **online recorder**'s actual
        on-disk layout (e.g. `data/online_data/2/kinematic/PSM1/`).
      - `"inside_arm"` — nested under `arm`. This matches the
        **offline recorder**'s on-disk layout (e.g.
        `data/offline_data/3/kinematic/PSM1/`) and also the schema
        diagram.

    Callers (the decomposer) should pick the location based on the
    episode's `recorder_variant` so the inverse reproduces the original
    raw layout per clip rather than picking one canonical form for all.

    Now emits every field that's loadable on the forward side:
    `local_measured_cp`, `measured_cv`, and `measured_cp.velocity` —
    so the decompose round-trip is field-complete (modulo floating-
    point repr drift in JSON).
    """
    measured_data: dict = {
        "measured_js": _joint_to_dict(sample.measured_js),
    }
    if _has_cartesian(sample.local_measured_cp):
        measured_data["local_measured_cp"] = _cartesian_to_dict(sample.local_measured_cp)
    if _has_cartesian(sample.measured_cp):
        measured_data["measured_cp"] = _cartesian_to_dict(sample.measured_cp)
    if _has_twist(sample.measured_cv):
        measured_data["measured_cv"] = _twist_to_dict(sample.measured_cv)

    setpoint_data: dict = {
        "setpoint_js": _joint_to_dict(sample.setpoint_js),
    }
    if sample.setpoint_cp is not None and _has_cartesian(sample.setpoint_cp):
        setpoint_data["setpoint_cp"] = _cartesian_to_dict(sample.setpoint_cp)

    arm: dict = {"measured_data": measured_data, "setpoint_data": setpoint_data}
    top: dict = {"arm": arm}

    # PSM-only siblings — placed where the caller requests so the
    # round-trip reproduces either the online (top_level) or offline
    # (inside_arm) recorder's actual layout.
    placement = arm if jaw_freq_location == "inside_arm" else top
    if sample.measured_jaw_position is not None or sample.setpoint_jaw_position is not None:
        jaw: dict = {}
        if sample.measured_jaw_position is not None:
            jaw["measured_data"] = {"position": [sample.measured_jaw_position]}
        if sample.setpoint_jaw_position is not None:
            jaw["setpoint_data"] = {"position": [sample.setpoint_jaw_position]}
        placement["jaw"] = jaw

    if sample.source_frequency_hz is not None:
        placement["measured_frequency"] = sample.source_frequency_hz

    return [top]


def _joint_to_dict(snap: JointSnapshot) -> Optional[dict]:
    if snap.position is None and snap.velocity is None and snap.effort is None:
        return None
    return {
        "position": snap.position,
        "velocity": snap.velocity,
        "effort": snap.effort,
    }


def _has_cartesian(snap: Optional[CartesianSnapshot]) -> bool:
    return snap is not None and (
        snap.position is not None or snap.orientation is not None or snap.velocity is not None
    )


def _has_twist(snap: TwistSnapshot) -> bool:
    return snap.linear is not None or snap.angular is not None


def _cartesian_to_dict(snap: CartesianSnapshot) -> dict:
    out: dict = {}
    if snap.position is not None:
        out["position"] = snap.position
    if snap.orientation is not None:
        out["orientation"] = snap.orientation
    if snap.velocity is not None:
        out["velocity"] = snap.velocity
    return out


def _twist_to_dict(snap: TwistSnapshot) -> dict:
    out: dict = {}
    if snap.linear is not None:
        out["linear"] = snap.linear
    if snap.angular is not None:
        out["angular"] = snap.angular
    return out
