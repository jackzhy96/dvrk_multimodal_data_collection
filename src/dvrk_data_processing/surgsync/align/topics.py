"""Canonical catalog of every per-frame timestamp topic the packer tracks.

Single source of truth for:
- `ingest/timestamps.py` — extracts each stamp from `time_syn/<frame>.json`
- `align/aligner.py`     — computes `delta_to_master.<topic>_ns` per row
- `schema/timestamp.py`  — declares every `delta_to_master.<topic>_ns` column
- `encode/per_modality_parquet.py` — writes the delta columns into
  `timestamp.parquet`

Every topic shows up exactly once across the parquet schemas (in
timestamp.parquet only — `delta_to_master.*_ns` is **not** duplicated
under `ECM.parquet` / `PSM{1,2}.parquet` anymore).

The "extractor path" is the list of nested keys to follow inside
`time_syn/<frame>.json` to reach a `{sec, nsec}` block. Missing path
anywhere along the way → NULL stamp (sentinel) for that frame.

Master clock: `image_left_stamp` at the top level — it's not in the
table below; it's separate via `MASTER_STAMP_PATH`.
"""
from __future__ import annotations
from dataclasses import dataclass


# Master clock — the stereo-left capture timestamp. Every delta below
# is measured against this.
MASTER_STAMP_PATH: tuple[str, ...] = ("image_left_stamp",)


@dataclass(frozen=True)
class TopicSpec:
    """One synced modality.

    `name` is the canonical topic name — used as the dict key in
    `TimestampTable.topic_stamps` / `AlignedClip.topic_deltas`, and as
    the column suffix `delta_to_master.<name>_ns` in timestamp.parquet.

    `path` is the nested-dict extractor into `time_syn/<frame>.json`.

    `kinematic_topic` is True for any kinematic-bus topic (per-arm
    PSM/ECM stamp). The legacy kinematic-only stats summary
    (`median_kin_delta_ms`/`max_kin_delta_ms`) aggregates these.

    `optional` flags topics that are normally absent on the offline
    recorder (`setpoint_cp_stamp` on ECM / PSM). They're never
    required, just informational.
    """
    name: str
    path: tuple[str, ...]
    kinematic_topic: bool = False
    optional: bool = False


# Image streams (image_left is master; only the others get deltas).
_IMAGE_TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec("image_right", ("image_right_stamp",)),
    TopicSpec("image_side",  ("side_image_1_stamp",)),
)


def _per_arm_topics(arm: str, has_jaw: bool) -> tuple[TopicSpec, ...]:
    """All kinematic-bus topics for one arm.

    Topic list (per the packing spec): measured_js, measured_cp,
    measured_cv, local_measured_cp, setpoint_js, setpoint_cp
    (optional — absent on offline recorder), and for PSMs only the
    two jaw stamps. `header_cv_stamp` and `reference_js_stamp` are
    intentionally **not** tracked — they're internal to the bus and
    don't correspond to a published modality the consumer cares about.
    """
    base = ("Kinematics_set_1", arm)
    md   = base + ("measured_data",)
    sd   = base + ("setpoint_data",)
    out = [
        TopicSpec(f"{arm}.measured_js",       md + ("measured_js_stamp",),       kinematic_topic=True),
        TopicSpec(f"{arm}.measured_cp",       md + ("measured_cp_stamp",),       kinematic_topic=True),
        TopicSpec(f"{arm}.measured_cv",       md + ("measured_cv_stamp",),       kinematic_topic=True),
        TopicSpec(f"{arm}.local_measured_cp", md + ("local_measured_cp_stamp",), kinematic_topic=True),
        TopicSpec(f"{arm}.setpoint_js",       sd + ("setpoint_js_stamp",),       kinematic_topic=True),
        TopicSpec(f"{arm}.setpoint_cp",       sd + ("setpoint_cp_stamp",),       kinematic_topic=True, optional=True),
    ]
    if has_jaw:
        jaw = base + ("jaw",)
        out += [
            TopicSpec(f"{arm}.jaw_measured", jaw + ("measured_stamp",), kinematic_topic=True),
            TopicSpec(f"{arm}.jaw_setpoint", jaw + ("setpoint_stamp",), kinematic_topic=True),
        ]
    return tuple(out)


# Final canonical list — order matters: it's the iteration order for
# delta computation, the column order in timestamp.parquet, and the
# topic-key order in the per-topic latency JSON.
TIMESTAMP_TOPICS: tuple[TopicSpec, ...] = (
    *_IMAGE_TOPICS,
    *_per_arm_topics("ECM",  has_jaw=False),
    *_per_arm_topics("PSM1", has_jaw=True),
    *_per_arm_topics("PSM2", has_jaw=True),
)


# Convenience views over the catalog. These all derive from
# TIMESTAMP_TOPICS — changing the catalog updates them automatically.

TOPIC_NAMES:        tuple[str, ...]      = tuple(t.name for t in TIMESTAMP_TOPICS)
TOPIC_BY_NAME:      dict[str, TopicSpec] = {t.name: t for t in TIMESTAMP_TOPICS}
KINEMATIC_TOPICS:   tuple[str, ...]      = tuple(t.name for t in TIMESTAMP_TOPICS if t.kinematic_topic)
DELTA_COLUMN_NAMES: tuple[str, ...]      = tuple(f"delta_to_master.{t.name}_ns" for t in TIMESTAMP_TOPICS)


def extract_stamp(time_syn: dict, path: tuple[str, ...]) -> Optional[dict]:  # noqa: F821 (Optional defined below for forward-ref)
    """Walk the nested JSON to reach a `{sec, nsec}` stamp dict.

    Returns the leaf dict (or None if any step along the path is
    missing). Callers convert the leaf to int64 ns via
    `ingest.timestamps._stamp_to_ns`.
    """
    cur: object = time_syn
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur if isinstance(cur, dict) else None


# Forward reference for Optional used in extract_stamp.
from typing import Optional  # noqa: E402
