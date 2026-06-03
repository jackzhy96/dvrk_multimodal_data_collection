"""Dataclass configs for the `surgsync` CLI, registered with Hydra.

The structured config hierarchy mirrors `config/surgsync/`:

  build.yaml          → SurgSyncBuildCfg          (top-level, used by `surgsync build`)
  encode/h264_crf18.yaml → H264EncodeCfg
  encode/ffv1.yaml       → FFV1EncodeCfg
  encode/preview_h264.yaml → PreviewEncodeCfg
  align/online.yaml      → AlignCfg (strict variant)
  align/offline.yaml     → AlignCfg (nearest-interp variant)
  path_config/jack_local.yaml → SurgSyncPathCfg

Dataclasses use field(default_factory=...) so Hydra can override
individual fields from the CLI (`encode.h264.crf=20`).

We deliberately do NOT register `SurgSyncBuildCfg` as a structured-config
group via ConfigStore — Hydra 1.3 + OmegaConf has a documented
default_factory issue with nested dataclasses (the same bug worked around
in `scripts/run_all_stages.py`). Plain YAML composition via `defaults:`
works correctly and we only need the dataclasses as a typed surface for
the modules that consume them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Encoding sub-configs
# ---------------------------------------------------------------------------

@dataclass
class H264EncodeCfg:
    """Processed RGB video — visually lossless at CRF 18.

    fps defaults to 10 Hz to match the dVRK capture rate used in this
    dataset. Override on the CLI for clips recorded at a different rate.
    """
    crf: int = 18
    fps: float = 10.0
    preset: str = "medium"


@dataclass
class FFV1EncodeCfg:
    """Raw RGB + preprocess visualization streams — bit-exact FFV1."""
    fps: float = 10.0


@dataclass
class PreviewEncodeCfg:
    """Option-C preview videos — H.264 8-bit visualization."""
    enable: bool = False
    crf: int = 23
    fps: float = 10.0


@dataclass
class EncodeCfg:
    h264: H264EncodeCfg = field(default_factory=H264EncodeCfg)
    ffv1: FFV1EncodeCfg = field(default_factory=FFV1EncodeCfg)
    preview: PreviewEncodeCfg = field(default_factory=PreviewEncodeCfg)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignCfg:
    """Tolerance policy and contiguity detection knobs.

    `tol_ms_*` are millisecond windows for the nearest-within-tolerance
    matcher. The defaults are set for the typical 30 Hz capture / 1000 Hz
    kinematic rates we see in practice (the kinematic stream runs at the
    recorded running rate, nominally 1000 Hz).
    """
    # Online (strict): 2 ms image_right; 100 ms kinematic by default to
    # accommodate standard rosbag-recorded data (camera+PSM topics
    # commonly land 10–30 ms apart in the time_syn file when recording
    # isn't on a tight real-time loop). Operators recording in real-time
    # mode can tighten this back to 2 ms via the CLI override.
    tol_ms_image_right_online: float = 2.0
    tol_ms_image_side_online: float = 33.0
    tol_ms_kinematic_online: float = 100.0

    # Offline (nearest-interp): wider for inter-modality, narrower
    # within-modality.
    tol_ms_image_right_offline: float = 2.0
    tol_ms_image_side_offline: float = 33.0     # ~1 frame at 30 fps
    tol_ms_kinematic_offline: float = 100.0     # see online comment above

    # Contiguity detection multiplier — diff > N * expected_period is a drop.
    contiguity_period_multiplier: float = 1.5


# ---------------------------------------------------------------------------
# Path config
# ---------------------------------------------------------------------------

@dataclass
class SurgSyncPathCfg:
    """Locations of the inputs (preprocessing outputs) and the output dataset root."""
    data_dir: str = "data"
    # Datasets we sweep into one release. Each name is a top-level
    # directory under `data_dir`.
    datasets: List[str] = field(default_factory=lambda: ["offline_data", "online_data"])
    # Output dataset root. The release tag is carried by
    # `meta/dataset.json.data_version`, not the directory name — the
    # output sits directly under `surgsync_release/`.
    dataset_root: str = "data/surgsync_release"


# ---------------------------------------------------------------------------
# Build / clip selection
# ---------------------------------------------------------------------------

@dataclass
class ClipSelector:
    """How to pick which clips to pack.

    Mirrors the discovery modes of `scripts/run_all_stages.py` so
    preprocessing and packing sweeps share a vocabulary.
    """
    # "list"   — use the `list` field below (e.g. ["online_data/2"])
    # "all"    — sweep every <dataset>/<clip> under data_dir
    # "dataset"— sweep every clip under one dataset (`dataset_name`)
    source: str = "all"
    list: List[str] = field(default_factory=list)
    dataset_name: str = ""


@dataclass
class TaskMapping:
    """Per-clip task labels (task names go into the Hive partition column).

    Since meta_data.json doesn't carry a `task` field, the packer
    either infers it from `annotation/phase/*.json` (the default,
    `default_task="auto"`) or accepts an explicit `tasks.overrides`
    mapping. Keys in `overrides` are `<dataset>/<clip_idx>` strings.

    Task vocab itself (the `meta/tasks.jsonl` shipped with each
    release) is auto-generated by the build from
    `workflow_description.json`; there is no `vocab_jsonl_source`
    config knob anymore.

    Example:
        default_task: "auto"
        overrides:
          online_data/2: "single_interrupted_stitch"
          offline_data/3: "tissue_manipulation"
    """
    default_task: str = "auto"
    overrides: dict = field(default_factory=dict)


@dataclass
class SurgSyncBuildCfg:
    """Top-level config bound by `surgsync build`."""
    path_config: SurgSyncPathCfg = field(default_factory=SurgSyncPathCfg)
    encode: EncodeCfg = field(default_factory=EncodeCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    clips: ClipSelector = field(default_factory=ClipSelector)
    tasks: TaskMapping = field(default_factory=TaskMapping)

    # Dataset-level metadata
    data_version: str = "1.0"
    release_option: str = "B"   # "A" | "B" | "C" — Option B (preprocess) is the default

    # Build behavior
    force: bool = False           # overwrite already-finalized episodes
    clean_staging: bool = False   # rmtree any leftover staging dirs at start
    parallelism: int = 1          # MVP: 1 worker (CPU-bound; raise for parallelism)
    log_dir: str = ""             # JSONL log dir; defaults to <dataset_root>/.logs/
    fps: float = 10.0             # dVRK capture rate used by every video encoder

    # Optional ingredient subsets — turn off for faster iteration
    include_video_processed: bool = True
    include_video_raw: bool = True       # MANDATORY per the packer invertibility contract
    include_preprocess: bool = True
    include_preview: bool = False
