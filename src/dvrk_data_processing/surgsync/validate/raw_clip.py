"""Pre-ingest sanity check on a raw clip.

Checks the documented raw-clip structure — every required subdirectory
present, every per-frame index space matches (within the documented
"gesture may be partial" tolerance).
"""
from __future__ import annotations
from pathlib import Path
from typing import List

from dvrk_data_processing.surgsync.ingest.clip import sorted_frame_indices
from dvrk_data_processing.surgsync.validate.types import ValidationIssue


# Subdirs that must be present in any raw clip.
_REQUIRED_SUBDIRS = (
    "annotation/contact_detection",
    "annotation/phase",
    "annotation/step",
    "camera_calibration",
    "hand_eye_calibration",
    "image/left",
    "image/right",
    "kinematic/ECM",
    "kinematic/PSM1",
    "kinematic/PSM2",
    "time_syn",
)

# Optional but commonly present.
_OPTIONAL_SUBDIRS = (
    "annotation/gesture",   # may be partial
    "image/side",           # online recorder
    "image/side1",          # offline recorder
)


def validate_raw_clip(clip_dir: Path) -> List[ValidationIssue]:
    """Return a list of issues found in `clip_dir`. Empty list = clean."""
    issues: List[ValidationIssue] = []
    clip_dir = Path(clip_dir)

    if not clip_dir.is_dir():
        return [ValidationIssue("ERROR", "raw_clip_missing",
                                f"raw clip directory does not exist: {clip_dir}")]

    # meta_data.json
    if not (clip_dir / "meta_data.json").is_file():
        issues.append(ValidationIssue(
            "ERROR", "raw_missing_meta",
            f"meta_data.json missing under {clip_dir}",
        ))

    # required subdirs
    for sub in _REQUIRED_SUBDIRS:
        if not (clip_dir / sub).is_dir():
            issues.append(ValidationIssue(
                "ERROR", "raw_missing_subdir",
                f"required subdir missing: {sub}",
            ))

    # frame-count parity across the primary frame-indexed dirs.
    # Per the raw spec, image/{left,right} + kinematic/{ECM,PSM1,PSM2}
    # + time_syn + annotation/{contact_detection,phase,step} are expected
    # equinumerous. gesture is allowed partial.
    counts: dict[str, int] = {}
    for sub in (
        "image/left", "image/right",
        "kinematic/ECM", "kinematic/PSM1", "kinematic/PSM2",
        "time_syn",
        "annotation/contact_detection", "annotation/phase", "annotation/step",
    ):
        d = clip_dir / sub
        if d.is_dir():
            counts[sub] = len(sorted_frame_indices(d, ".json" if "annotation" in sub or "kinematic" in sub or sub == "time_syn" else ".png"))

    if counts:
        ref = max(counts.values())
        for sub, n in counts.items():
            if n != ref:
                issues.append(ValidationIssue(
                    "ERROR", "raw_frame_count_mismatch",
                    f"{sub} has {n} frames; expected {ref} (max across required dirs)",
                ))

    # Optional dirs: log INFO for ones that are missing so the operator
    # knows the clip is stereo-only / has no gesture.
    for sub in _OPTIONAL_SUBDIRS:
        if not (clip_dir / sub).is_dir():
            issues.append(ValidationIssue(
                "INFO", "raw_missing_optional",
                f"optional subdir absent: {sub}",
            ))

    return issues
