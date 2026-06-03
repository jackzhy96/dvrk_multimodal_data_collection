"""Per-frame annotation JSON ↔ in-memory record.

The four annotation tasks each have their own per-frame file under
`raw_dir/annotation/<task>/<frame>.json`. Schemas:

  contact_detection/<frame>.json  → {"PSM1": 0/1, "PSM2": 0/1}
  gesture/<frame>.json            → {"gesture": {"PSM1": "<id>", "PSM2": "<id>"}}
  phase/<frame>.json              → {"phase": "<id>"}
  step/<frame>.json               → {"step": "<id>"}

All class-id fields are JSON strings (intentional and matches the
annotation GUI). Some clips have partial gesture coverage (online_data/2
has 821 gesture files for 886 frames).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AnnotationSample:
    """All annotation labels for one frame.

    All fields are nullable to tolerate per-task missing files (gesture
    is the common case, but other tasks may also have gaps).
    """
    frame: int
    contact_PSM1: Optional[int] = None
    contact_PSM2: Optional[int] = None
    gesture_PSM1: Optional[str] = None
    gesture_PSM2: Optional[str] = None
    phase: Optional[str] = None
    step: Optional[str] = None


# ---------------------------------------------------------------------------
# Forward direction — JSON → AnnotationSample
# ---------------------------------------------------------------------------

# Raw annotation cells sometimes carry the string sentinels "None",
# "null", "NaN", or "" instead of being JSON null. Normalize all of
# those to a real None at ingest so downstream code (parquet writer,
# validator, vocab joins) sees a single canonical "missing" value.
# The annotation GUI / earlier preprocessing pipeline writes Python's
# repr "None" in some clips — that's the bug we're papering over here,
# at the only ingest entry point.
_MISSING_SENTINELS = frozenset({"none", "null", "nan", ""})


def _normalize_id(value) -> Optional[str]:
    """Coerce a raw annotation id to a clean Optional[str], collapsing
    string-form "None"/"null"/etc. to a real None."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _MISSING_SENTINELS:
        return None
    return s


def load_annotation_frame(
    *,
    contact_path: Optional[Path],
    gesture_path: Optional[Path],
    phase_path: Optional[Path],
    step_path: Optional[Path],
    frame_idx: int,
) -> AnnotationSample:
    """Load all four annotation files for one frame. Missing files
    (or files carrying string-form "None"/"null") leave the
    corresponding fields as None."""
    s = AnnotationSample(frame=frame_idx)

    if contact_path is not None and contact_path.exists():
        with open(contact_path) as f:
            d = json.load(f)
        s.contact_PSM1 = int(d["PSM1"]) if "PSM1" in d else None
        s.contact_PSM2 = int(d["PSM2"]) if "PSM2" in d else None

    if gesture_path is not None and gesture_path.exists():
        with open(gesture_path) as f:
            d = json.load(f)
        gesture = d.get("gesture", {})
        s.gesture_PSM1 = _normalize_id(gesture.get("PSM1"))
        s.gesture_PSM2 = _normalize_id(gesture.get("PSM2"))

    if phase_path is not None and phase_path.exists():
        with open(phase_path) as f:
            d = json.load(f)
        s.phase = _normalize_id(d.get("phase"))

    if step_path is not None and step_path.exists():
        with open(step_path) as f:
            d = json.load(f)
        s.step = _normalize_id(d.get("step"))

    return s


# ---------------------------------------------------------------------------
# Inverse direction — AnnotationSample → JSON files (unpack)
# ---------------------------------------------------------------------------

def annotation_sample_to_files(
    sample: AnnotationSample,
    *,
    contact_path: Path,
    gesture_path: Optional[Path],
    phase_path: Path,
    step_path: Path,
) -> None:
    """Write the per-frame annotation JSONs back to disk.

    `gesture_path` may be None — when the original clip had no gesture
    file for this frame (common in online_data/2 — a known frame-count
    divergence), the decomposer should pass None for that
    one frame to skip writing the gesture file.
    """
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(contact_path, "w") as f:
        json.dump({"PSM1": sample.contact_PSM1 or 0, "PSM2": sample.contact_PSM2 or 0}, f)

    if gesture_path is not None and (sample.gesture_PSM1 is not None or sample.gesture_PSM2 is not None):
        gesture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gesture_path, "w") as f:
            json.dump({"gesture": {
                "PSM1": sample.gesture_PSM1,
                "PSM2": sample.gesture_PSM2,
            }}, f)

    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with open(phase_path, "w") as f:
        json.dump({"phase": sample.phase}, f)

    step_path.parent.mkdir(parents=True, exist_ok=True)
    with open(step_path, "w") as f:
        json.dump({"step": sample.step}, f)
