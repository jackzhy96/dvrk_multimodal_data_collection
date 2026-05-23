"""Bulk-load per-frame annotation JSONs.

Annotations are organized by task (`contact_detection`, `gesture`,
`phase`, `step`) — one folder per task, one file per frame per task.
Per `specs/raw_data_spec.md`, gesture coverage can be partial; we
tolerate gaps and record per-task presence stats so the validator can
surface them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from dvrk_data_processing.surgsync.serde.annotation_io import (
    AnnotationSample,
    load_annotation_frame,
)
from dvrk_data_processing.surgsync.ingest.clip import sorted_frame_indices


@dataclass
class AnnotationTable:
    """Sparse-indexed per-frame annotation table.

    `samples` is a dict keyed by source frame index → AnnotationSample.
    The align stage looks samples up by source index; frames with no
    annotation for any task land here with `samples.get(i)` returning
    None.

    `partials` records which annotation tasks had < N coverage (where
    N is the number of frames in the master timeline). The encoder
    surfaces this via `episode_meta.json.sync_stats` so consumers can branch.
    """
    samples: dict[int, AnnotationSample] = field(default_factory=dict)
    gesture_partial: bool = False
    counts: dict[str, int] = field(default_factory=dict)


def load_annotations(annotation_root: Path) -> AnnotationTable:
    """Load every annotation file under `annotation_root/{task}/`.

    Returns a sparse table — only frames that have at least one
    annotation file land in the dict.
    """
    tasks = {
        "contact_detection": annotation_root / "contact_detection",
        "gesture":           annotation_root / "gesture",
        "phase":             annotation_root / "phase",
        "step":              annotation_root / "step",
    }
    # Build the union of source frame indices across all four tasks.
    indices: set[int] = set()
    counts: dict[str, int] = {}
    for task, folder in tasks.items():
        idx = set(sorted_frame_indices(folder, suffix=".json"))
        counts[task] = len(idx)
        indices |= idx

    samples: dict[int, AnnotationSample] = {}
    for frame_idx in sorted(indices):
        s = load_annotation_frame(
            contact_path=tasks["contact_detection"] / f"{frame_idx}.json",
            gesture_path=tasks["gesture"] / f"{frame_idx}.json",
            phase_path=tasks["phase"] / f"{frame_idx}.json",
            step_path=tasks["step"] / f"{frame_idx}.json",
            frame_idx=frame_idx,
        )
        samples[frame_idx] = s

    # Partial gesture coverage is the common case (online_data/2 has 821
    # of 886 expected). We flag it for downstream surfacing.
    if counts["gesture"] and counts["gesture"] < max(counts.values()):
        gesture_partial = True
    else:
        gesture_partial = False

    return AnnotationTable(
        samples=samples,
        gesture_partial=gesture_partial,
        counts=counts,
    )
