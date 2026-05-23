"""Bulk-load per-frame calibrated_kinematic JSONs (optional modality).

The preprocessing hand-eye stage emits per-frame JSON for PSM1/PSM2 at:
    `processed_dir/kinematic_reproject/<PSM>/calibrated_kinematic/<i>.json`

When `enable: true` was set in preprocessing, every frame in the raw
clip's range has a corresponding JSON. When the preprocessing
calibrated_kinematic stage wasn't
run (or the user used the dVRK variant which doesn't emit these files),
the folder is absent — this ingest layer surfaces that as "empty dict"
so the per-clip orchestrator can treat the stream as optional.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dvrk_data_processing.surgsync.ingest.clip import sorted_frames
from dvrk_data_processing.surgsync.serde.calibrated_kinematic_io import (
    CalibratedKinematicSample, load_calibrated_frame,
)


log = logging.getLogger(__name__)


@dataclass
class CalibratedKinematicTable:
    """Per-PSM sparse-indexed table: source frame index → sample."""
    arm_name: str
    samples: dict[int, CalibratedKinematicSample] = field(default_factory=dict)


def load_calibrated_arm(processed_dir: Path, arm: str) -> CalibratedKinematicTable:
    """Read every `calibrated_kinematic/<frame>.json` under
    `processed_dir/kinematic_reproject/<arm>/calibrated_kinematic/`.

    Returns an empty table when the directory doesn't exist — the
    caller treats absence as "this modality wasn't produced".
    """
    folder = processed_dir / "kinematic_reproject" / arm / "calibrated_kinematic"
    if not folder.exists():
        log.info("calibrated_kinematic not present for %s under %s — optional, skipping",
                 arm, processed_dir)
        return CalibratedKinematicTable(arm_name=arm, samples={})

    samples: dict[int, CalibratedKinematicSample] = {}
    for p in sorted_frames(folder, suffix=".json"):
        idx = int(p.stem)
        samples[idx] = load_calibrated_frame(p, arm, idx)
    return CalibratedKinematicTable(arm_name=arm, samples=samples)


def aligned_to_master(
    table: CalibratedKinematicTable,
    source_frame_indices,
) -> tuple[list[Optional[tuple[list[float], list[float]]]],
           list[Optional[tuple[list[float], list[float]]]]]:
    """Project a CalibratedKinematicTable onto the master timeline.

    Returns two parallel lists (one per master frame), each entry being
    `(position, orientation)` or None. The first list is
    `measured_cp_calibrated`, the second is `setpoint_cp_calibrated`.
    """
    measured: list = []
    setpoint: list = []
    for src in source_frame_indices:
        s = table.samples.get(int(src))
        measured.append(None if s is None else s.measured_cp_calibrated)
        setpoint.append(None if s is None else s.setpoint_cp_calibrated)
    return measured, setpoint
