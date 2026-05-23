"""Discover + describe one raw clip.

`RawClip` carries every path the rest of the pipeline needs and a
record of which preprocessing outputs are present for this clip. Per
the preprocessing → packing contract, missing preprocessing outputs
do **not** trigger preprocessing to run — the encoder skips or fails
fast based on the operator's request.
"""
from __future__ import annotations
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


# Numeric sort helper. Raw files are unpadded integer stems; lexicographic
# sort gives 1, 10, 100, 2, ... which is wrong. The `utils.utility`
# `glob_sorted_frame` does the same job but is in a different package —
# we keep this lightweight version local to avoid the upstream coupling.
_FRAME_STEM_RE = re.compile(r"^(\d+)\.\w+$")


def sorted_frames(folder: Path, suffix: str = ".png") -> list[Path]:
    """Return every `<int>{suffix}` under `folder`, sorted numerically.

    Empty folder → empty list. Names that don't parse as integer-stem
    `<int>{suffix}` are skipped (with no warning — that's a raw-data
    layout error and the validator will catch it).
    """
    if not folder.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file() or not p.name.endswith(suffix):
            continue
        m = _FRAME_STEM_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def sorted_frame_indices(folder: Path, suffix: str = ".png") -> list[int]:
    """Same as `sorted_frames` but returns the integer indices."""
    return [int(_FRAME_STEM_RE.match(p.name).group(1)) for p in sorted_frames(folder, suffix)]


@dataclass
class RawClip:
    """One raw clip on disk.

    `recorder_variant` is derived from the parent directory name
    (`offline_data` / `online_data`). `side_dir_name` differs by variant
    (`side1` vs `side`) — the spec calls this out and `resolve_side_dir`
    handles it.

    `intermediate_present` and `processed_present` are preprocessing-output probes
    — surfaced here for the per-clip orchestrator's skip-with-message
    behavior when preprocessing outputs are missing.
    """
    # Identification
    dataset_name: str          # "offline_data" | "online_data"
    clip_index: str            # the per-clip integer-string id
    recorder_variant: str      # "offline" | "online"

    # Paths
    raw_dir: Path
    intermediate_dir: Path     # may not exist if preprocessing hasn't been run
    processed_dir: Path        # may not exist if preprocessing hasn't been run

    # Probed presence
    intermediate_present: bool = False
    processed_present: dict[str, bool] = field(default_factory=dict)

    # Convenience accessors
    @property
    def source_clip_str(self) -> str:
        """Stable identifier used to derive episode_id (`source_clip` field
        of episode_meta.json + UUID5 input)."""
        return f"data/{self.dataset_name}/{self.clip_index}/"

    @property
    def side_dir_name(self) -> str:
        """Side-camera folder name per recorder variant."""
        return "side1" if self.recorder_variant == "offline" else "side"

    @property
    def annotation_dir(self) -> Path:
        return self.raw_dir / "annotation"

    @property
    def kinematic_dir(self) -> Path:
        return self.raw_dir / "kinematic"

    @property
    def time_syn_dir(self) -> Path:
        return self.raw_dir / "time_syn"

    @property
    def meta_path(self) -> Path:
        return self.raw_dir / "meta_data.json"

    @property
    def intermediate_image_left_dir(self) -> Path:
        return self.intermediate_dir / "image" / "left"

    @property
    def intermediate_image_right_dir(self) -> Path:
        return self.intermediate_dir / "image" / "right"

    @property
    def intermediate_camera_calibration_dir(self) -> Path:
        return self.intermediate_dir / "camera_calibration"

    @property
    def hand_eye_dir(self) -> Path:
        return self.raw_dir / "hand_eye_calibration"


def discover_clip(
    data_dir: Path,
    dataset_name: str,
    clip_index: str,
) -> RawClip:
    """Build a RawClip for one `<data_dir>/<dataset>/<clip>` location.

    Probes for preprocessing outputs at the canonical paths from
    `config/path_config/jack_local_release.yaml`. Preprocessing now
    writes everything **inside the clip's own directory**:

        raw_dir          = <data_dir>/<dataset>/<clip>/
        intermediate_dir = <raw_dir>/preprocess/rectify_resize/   (preprocessing stage 1)
        processed_dir    = <raw_dir>/preprocess/                  (preprocessing stages 2-4 are siblings)

    Stage subfolders found under `processed_dir` are
    `rectify_resize/`, `kinematic_reproject/`,
    `kinematic_reproject_drawframe/`, `depth_estimation/`,
    `optical_flow/`.
    """
    raw_dir = data_dir / dataset_name / clip_index
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    if dataset_name.startswith("offline"):
        recorder = "offline"
    elif dataset_name.startswith("online"):
        recorder = "online"
    else:
        raise ValueError(
            f"Cannot infer recorder variant from dataset_name={dataset_name!r}; "
            "expected 'offline_data' or 'online_data'."
        )

    # New per-clip layout: preprocessing outputs all live under the
    # clip's own preprocess/ subtree. Old layouts where they hung off
    # the data-root (`<data_dir>/intermediate/...`) are no longer
    # supported — the release path config also reflects this.
    intermediate_dir = raw_dir / "preprocess" / "rectify_resize"
    processed_dir = raw_dir / "preprocess"

    # Probe presence of preprocessing outputs that the encoder will read.
    intermediate_present = (intermediate_dir / "image" / "left").exists()
    processed_present = {
        "kinematic_reproject": (processed_dir / "kinematic_reproject").exists(),
        "depth_estimation":    (processed_dir / "depth_estimation").exists(),
        "optical_flow":        (processed_dir / "optical_flow").exists(),
    }

    return RawClip(
        dataset_name=dataset_name,
        clip_index=clip_index,
        recorder_variant=recorder,
        raw_dir=raw_dir,
        intermediate_dir=intermediate_dir,
        processed_dir=processed_dir,
        intermediate_present=intermediate_present,
        processed_present=processed_present,
    )


def infer_task_from_phase(raw_dir: Path) -> Optional[str]:
    """Read `annotation/phase/<frame>.json` and pick the canonical task
    whose phase id is the most common across the clip's annotated
    frames.

    Returns `None` when the phase folder is missing or all frames are
    unphased — callers should fall back to an explicit task override
    or fail loudly. Doesn't read any other annotation modality (gesture
    / step / contact); phase alone is the right signal for routing.

    Mapping phase id → task name comes from
    `workflow_description.json:_task_routing` via
    `serde.workflow_text.phase_to_task`. Multiple tasks may route to
    the same phase (cold_cut_dissection variants share phase 2/5/6);
    we pick the first task listed in the JSON for that phase, which
    lets the JSON author choose the canonical name by listing order.
    """
    # Late import — the workflow_text module reads the JSON at import
    # time and shouldn't be loaded at clip-discovery time when not
    # needed (e.g. for code paths that pass an explicit task).
    from dvrk_data_processing.surgsync.serde.workflow_text import phase_to_task

    phase_dir = raw_dir / "annotation" / "phase"
    if not phase_dir.is_dir():
        return None
    counts: Counter[str] = Counter()
    for f in phase_dir.iterdir():
        if not f.is_file() or f.suffix != ".json":
            continue
        try:
            p = json.loads(f.read_text()).get("phase")
        except (json.JSONDecodeError, OSError) as e:
            log.warning("infer_task_from_phase: skip %s (%s)", f, e)
            continue
        if p is None:
            continue
        counts[str(p)] += 1
    if not counts:
        return None
    dominant_phase, n = counts.most_common(1)[0]
    if len(counts) > 1:
        # Mixed-phase clip — still go with the mode but record the
        # split so the operator can audit it later.
        log.info(
            "infer_task_from_phase(%s): mixed phases %s — picking dominant %r (%d frames)",
            raw_dir, dict(counts), dominant_phase, n,
        )
    return phase_to_task(dominant_phase)


def discover_clips(
    data_dir: Path,
    *,
    datasets: Optional[list[str]] = None,
) -> list[RawClip]:
    """Sweep `<data_dir>/<dataset>/*` for every per-clip subdirectory.

    `datasets` filters to a subset of top-level names (e.g.
    `["online_data"]`). When None, both standard names are tried —
    missing ones are silently skipped.

    Subdirs whose names parse as integers are sorted numerically; non-
    integer names are still included but sorted lexicographically after
    the numeric ones (so a hand-named clip doesn't break the sweep).
    """
    if datasets is None:
        datasets = ["offline_data", "online_data"]

    out: list[RawClip] = []
    for ds in datasets:
        ds_root = data_dir / ds
        if not ds_root.is_dir():
            continue
        subdirs = [p for p in ds_root.iterdir() if p.is_dir()]
        # Numeric clip indices first, lexicographic fallback.
        def _key(p):
            try:
                return (0, int(p.name))
            except ValueError:
                return (1, p.name)
        subdirs.sort(key=_key)
        for sub in subdirs:
            out.append(discover_clip(data_dir, ds, sub.name))
    return out
