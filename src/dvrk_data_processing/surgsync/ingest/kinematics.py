"""Bulk-load per-arm kinematic JSONs into per-frame KinematicSample lists.

Thin wrapper over `serde/kinematic_io.py` — for the packer's ingest pass we
load every frame's kinematic file into memory. The samples are then
joined to the master timeline by the align stage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from dvrk_data_processing.surgsync.serde.kinematic_io import (
    KinematicSample,
    load_arm_frame_json,
)
from dvrk_data_processing.surgsync.ingest.clip import sorted_frames


@dataclass
class ArmKinematics:
    arm_name: str
    samples: list[KinematicSample] = field(default_factory=list)


def load_arm(kinematic_root: Path, arm_name: str) -> ArmKinematics:
    """Read every `<kinematic_root>/<arm>/<frame>.json` into a list.

    The returned list is sorted by source frame index. The align stage
    handles missing frames by reading the source index from the sample
    object; gaps just produce shorter lists.
    """
    arm_dir = kinematic_root / arm_name
    files = sorted_frames(arm_dir, suffix=".json")
    samples = [
        load_arm_frame_json(p, arm_name, int(p.stem))
        for p in files
    ]
    return ArmKinematics(arm_name=arm_name, samples=samples)
