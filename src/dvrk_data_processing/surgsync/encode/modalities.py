"""Per-episode modalities manifest.

Walks a staged or finalized episode directory and emits a JSON
manifest describing every modality that ended up in the pack —
which streams are present, what codec/format they use, how many
frames they carry, and how big they are.

The manifest enumerates **every expected topic** explicitly (camera
streams, kinematic topics per arm, annotation topics, timestamp
deltas), so a consumer can iterate the dict and know up front what
the dataset is supposed to provide. Topics whose source was absent
get `{"present": false}`; present ones get `{"present": true,
"populated_frames": N, "coverage_ratio": N/expected_frames}`.

This is a deliberate complement to the boolean `has_*` flags on
`episode_meta.json`: those tell consumers which broad categories shipped;
`modalities.json` gives the per-topic detail (e.g. side camera missing
but stereo present; PSM2 has measured_cp populated but setpoint_cp
NULL because the clip is offline; gesture annotations are partial).

Schema is a free-form dict (no pydantic — the nesting is deep and the
field set varies by what preprocessing emitted for a given clip). Stable enough
for downstream tooling; future-proofed by carrying a `schema_version`
string.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.encode.codec import probe_frame_count
from dvrk_data_processing.surgsync.schema import SCHEMA_VERSION


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topic tables — the **expected** set of fields per parquet. Topics that
# aren't in the source parquet (e.g. an offline clip's setpoint_cp) end
# up with `{"present": false}` in the manifest, satisfying the
# "if not present, just say not present" contract.
# ---------------------------------------------------------------------------

# ECM has 4 joints, no jaw, no setpoint_cp, no calibrated_cp.
# Delta-to-master columns no longer live in the arm parquets — they
# all moved to timestamp.parquet (`TIMESTAMP_TOPIC_COLS` below).
ECM_TOPIC_COLS: tuple[str, ...] = (
    "local_measured_cp.position",
    "local_measured_cp.orientation",
    "measured_cp.position",
    "measured_cp.orientation",
    "measured_cp.velocity",            # 6-twist [vx, vy, vz, ωx, ωy, ωz]
    "measured_cv.linear",
    "measured_cv.angular",
    "measured_js.position",
    "measured_js.velocity",
    "measured_js.effort",
    "setpoint_js.position",
    "setpoint_js.velocity",
    "setpoint_js.effort",
)

# PSM superset: ECM topics + jaw + setpoint_cp + calibrated_cp + freq.
PSM_TOPIC_COLS: tuple[str, ...] = (
    "local_measured_cp.position",
    "local_measured_cp.orientation",
    "measured_cp.position",
    "measured_cp.orientation",
    "measured_cp.velocity",
    "measured_cv.linear",
    "measured_cv.angular",
    "measured_js.position",
    "measured_js.velocity",
    "measured_js.effort",
    "setpoint_js.position",
    "setpoint_js.velocity",
    "setpoint_js.effort",
    "setpoint_cp.position",            # NULL on offline recorder
    "setpoint_cp.orientation",
    "measured_cp_calibrated.position",     # NULL when preprocessing didn't emit calibrated_kinematic/
    "measured_cp_calibrated.orientation",
    "setpoint_cp_calibrated.position",     # NULL on offline; NULL when preprocessing didn't emit
    "setpoint_cp_calibrated.orientation",
    "jaw.measured_position",
    "jaw.setpoint_position",
    "source_frequency_hz",
)

ANNOTATION_TOPIC_COLS: tuple[str, ...] = (
    "contact.PSM1",
    "contact.PSM2",
    "gesture.PSM1",
    "gesture.PSM2",
    "phase",
    "step",
)

# Every `delta_to_master.<topic>_ns` column lives in timestamp.parquet;
# the list is derived from the central topic catalog so adding /
# removing topics needs only one edit (align/topics.py).
def _timestamp_topic_cols() -> tuple[str, ...]:
    from dvrk_data_processing.surgsync.align.topics import DELTA_COLUMN_NAMES
    return DELTA_COLUMN_NAMES

TIMESTAMP_TOPIC_COLS: tuple[str, ...] = _timestamp_topic_cols()

# Canonical camera streams enumerated per video block — `side` absent
# on stereo-only clips (the offline recorder names it `side1/` in raw;
# the pack normalizes to `side` in either case).
CAMERA_STREAMS: tuple[str, ...] = ("stereo_left", "stereo_right", "side")

GEOMETRY_STREAMS: tuple[tuple[str, str], ...] = (
    ("depth",              "bgr24"),
    ("flow_left",          "bgr24"),
    ("flow_right",         "bgr24"),
    ("heatmap_PSM1_left",  "gray8"),
    ("heatmap_PSM1_right", "gray8"),
    ("heatmap_PSM2_left",  "gray8"),
    ("heatmap_PSM2_right", "gray8"),
)


# ---------------------------------------------------------------------------
# Per-stream probes
# ---------------------------------------------------------------------------

def _probe_video(path: Path, *, codec: str, pix_fmt: Optional[str] = None) -> dict[str, Any]:
    """Build a one-entry dict for a single video stream (MKV or MP4).

    `path` must exist. Caller decides whether to probe via this helper
    or emit `{"present": False}` for absent streams.
    """
    out: dict[str, Any] = {
        "present": True,
        "path": path.name,
        "codec": codec,
        "size_bytes": path.stat().st_size,
    }
    if pix_fmt is not None:
        out["pix_fmt"] = pix_fmt
    try:
        out["frames"] = probe_frame_count(path)
    except Exception as e:
        log.warning("probe_frame_count failed for %s: %s", path, e)
        out["frames"] = None
    return out


def _absent() -> dict[str, Any]:
    return {"present": False}


def _probe_parquet(path: Path) -> dict[str, Any]:
    """Per-parquet probe: row count + on-disk size + column count."""
    if not path.is_file():
        return _absent()
    try:
        md = pq.read_metadata(path)
        return {
            "present": True,
            "path": path.name,
            "rows": md.num_rows,
            "columns": md.num_columns,
            "size_bytes": path.stat().st_size,
        }
    except Exception as e:
        log.warning("parquet metadata read failed for %s: %s", path, e)
        return {"present": True, "path": path.name, "rows": None, "columns": None,
                "size_bytes": path.stat().st_size, "error": str(e)}


def _nonnull_count(parquet: Path, column: str) -> Optional[int]:
    """Count non-NULL values in one parquet column. Returns None on
    parquet/column errors (e.g. column doesn't exist) so the caller can
    degrade gracefully — that case is treated as "not present"."""
    if not parquet.is_file():
        return None
    try:
        col = pq.ParquetFile(parquet).read(columns=[column]).column(column)
        return col.length() - col.null_count
    except Exception as e:
        log.debug("nonnull_count failed for %s::%s: %s", parquet, column, e)
        return None


def _topic_status(parquet_path: Path, column: str, expected_frames: int) -> dict[str, Any]:
    """Status entry for one per-frame topic. Returns either
    `{"present": false}` (column missing or all-NULL) or
    `{"present": true, "populated_frames": N, "coverage_ratio": frac}`.
    """
    n = _nonnull_count(parquet_path, column)
    if n is None or n == 0:
        return {"present": False}
    coverage = round(n / expected_frames, 4) if expected_frames else None
    return {
        "present": True,
        "populated_frames": n,
        "coverage_ratio": coverage,
    }


def _topics_block(
    parquet_path: Path, columns: tuple[str, ...], expected_frames: int,
) -> dict[str, dict[str, Any]]:
    """Build a `{topic: status}` dict for every expected column in a
    parquet. Missing parquet → every topic marked not-present."""
    return {col: _topic_status(parquet_path, col, expected_frames) for col in columns}


# ---------------------------------------------------------------------------
# Top-level collector
# ---------------------------------------------------------------------------

def collect_modalities(
    episode_dir: Path,
    *,
    expected_frames: int,
    episode_id: Optional[str] = None,
) -> dict[str, Any]:
    """Walk `episode_dir` and build the modalities manifest.

    Safe to call against a staging dir (before the atomic rename) or a
    finalized episode dir — both layouts are identical at this point.
    """
    episode_dir = Path(episode_dir)
    mods: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "expected_frames": expected_frames,
    }

    # ---- Video (H.264 processed, MP4 container) ---------------------------
    # MP4 — see encode/video_processed.py for why H.264 isn't in MKV.
    video: dict[str, Any] = {}
    for stream in CAMERA_STREAMS:
        p = episode_dir / "video" / f"{stream}.mp4"
        video[stream] = _probe_video(p, codec="h264", pix_fmt="yuv420p") if p.is_file() else _absent()
    mods["video"] = video

    # ---- Video raw (FFV1 bit-exact, MKV container) ------------------------
    video_raw: dict[str, Any] = {}
    for stream in CAMERA_STREAMS:
        p = episode_dir / "video_raw" / f"{stream}.mkv"
        video_raw[stream] = _probe_video(p, codec="ffv1", pix_fmt="bgr24") if p.is_file() else _absent()
    mods["video_raw"] = video_raw

    # ---- Preprocess streams (FFV1 over preprocessing visualization PNGs) ----
    # Folder named `preprocess/` (matches the preprocessing source-side
    # `<raw_dir>/preprocess/` directory the streams came from).
    preprocess_block: dict[str, Any] = {}
    for stream, pix_fmt in GEOMETRY_STREAMS:
        p = episode_dir / "preprocess" / f"{stream}.mkv"
        preprocess_block[stream] = _probe_video(p, codec="ffv1", pix_fmt=pix_fmt) if p.is_file() else _absent()
    mods["preprocess"] = preprocess_block

    # ---- Kinematic parquets — one block per arm, every topic enumerated ---
    kinematic: dict[str, Any] = {}
    for arm in ("ECM", "PSM1", "PSM2"):
        p = episode_dir / f"{arm}.parquet"
        entry = _probe_parquet(p)
        cols = ECM_TOPIC_COLS if arm == "ECM" else PSM_TOPIC_COLS
        # Always emit the topics block — even when the parquet is
        # missing, so consumers can see what was expected.
        entry["topics"] = _topics_block(p, cols, expected_frames)
        entry["topics_present_count"] = sum(
            1 for t in entry["topics"].values() if t.get("present")
        )
        entry["topics_total"] = len(cols)
        kinematic[arm] = entry
    mods["kinematic"] = kinematic

    # ---- Annotation -------------------------------------------------------
    ann_path = episode_dir / "annotation.parquet"
    ann_entry = _probe_parquet(ann_path)
    ann_entry["topics"] = _topics_block(ann_path, ANNOTATION_TOPIC_COLS, expected_frames)
    ann_entry["topics_present_count"] = sum(
        1 for t in ann_entry["topics"].values() if t.get("present")
    )
    ann_entry["topics_total"] = len(ANNOTATION_TOPIC_COLS)
    # Surface gesture partial coverage at the top of the annotation
    # block — it's the most common "not-quite-present" case.
    gesture_topic = ann_entry["topics"].get("gesture.PSM1", {})
    if gesture_topic.get("present") and (gesture_topic.get("coverage_ratio") or 1.0) < 1.0:
        ann_entry["gesture_partial"] = True
    mods["annotation"] = ann_entry

    # ---- Timestamp parquet (image-side deltas) ----------------------------
    ts_path = episode_dir / "timestamp.parquet"
    ts_entry = _probe_parquet(ts_path)
    ts_entry["topics"] = _topics_block(ts_path, TIMESTAMP_TOPIC_COLS, expected_frames)
    mods["timestamp"] = ts_entry

    # ---- Calibration ------------------------------------------------------
    cal_dir = episode_dir / "calibration"
    cal: dict[str, Any] = {}
    for name in ("left.yaml", "right.yaml", "rectify_params.json", "camera.json"):
        present = (cal_dir / name).is_file()
        cal[name] = {"present": present}
        if present:
            cal[name]["size_bytes"] = (cal_dir / name).stat().st_size
    he_dir = cal_dir / "hand_eye"
    cal["hand_eye"] = {}
    for arm in ("PSM1", "PSM2"):
        for convention in ("dVRK", "open-cv"):
            fname = f"{arm}-registration-{convention}.json"
            present = (he_dir / fname).is_file()
            cal["hand_eye"][fname] = {"present": present}
            if present:
                cal["hand_eye"][fname]["size_bytes"] = (he_dir / fname).stat().st_size
    mods["calibration"] = cal

    # ---- Summary (cheap roll-up for filtering) ----------------------------
    psm1_topics = kinematic.get("PSM1", {}).get("topics", {})
    has_calibrated = any(
        psm1_topics.get(k, {}).get("present") for k in (
            "measured_cp_calibrated.position", "setpoint_cp_calibrated.position",
        )
    ) or any(
        kinematic.get("PSM2", {}).get("topics", {}).get(k, {}).get("present") for k in (
            "measured_cp_calibrated.position", "setpoint_cp_calibrated.position",
        )
    )
    mods["summary"] = {
        "video_streams":       sum(1 for v in mods["video"].values() if v.get("present")),
        "video_raw_streams":   sum(1 for v in mods["video_raw"].values() if v.get("present")),
        "preprocess_streams":  sum(1 for v in mods["preprocess"].values() if v.get("present")),
        "kinematic_arms":      sum(1 for v in mods["kinematic"].values() if v.get("present")),
        "has_annotation":      mods["annotation"].get("present", False),
        "has_calibrated_kinematic": has_calibrated,
        "gesture_partial":     mods["annotation"].get("gesture_partial", False),
    }

    return mods


def write_modalities_json(
    episode_dir: Path,
    *,
    expected_frames: int,
    episode_id: Optional[str] = None,
) -> Path:
    """Atomic write of `<episode_dir>/modalities.json`. Returns the path."""
    mods = collect_modalities(episode_dir, expected_frames=expected_frames,
                              episode_id=episode_id)
    dst = Path(episode_dir) / "modalities.json"
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mods, indent=2))
    os.replace(tmp, dst)
    return dst


# ---------------------------------------------------------------------------
# Dataset-level aggregate
# ---------------------------------------------------------------------------

def aggregate_modalities(dataset_root: Path) -> dict[str, Any]:
    """Walk every episode's `modalities.json` and build an aggregate
    summary at `meta/modalities.json`.

    The aggregate gives consumers a quick "which streams are available
    across the release" view without opening every episode dir. It now
    also rolls up per-topic presence across the release for kinematics,
    annotations, and timestamps.
    """
    dataset_root = Path(dataset_root)
    n_episodes = 0
    # Stream-level (file-level) presence counters.
    counters: dict[str, dict[str, int]] = {
        "video":      {s: 0 for s in CAMERA_STREAMS},
        "video_raw":  {s: 0 for s in CAMERA_STREAMS},
        "preprocess": {s: 0 for s, _ in GEOMETRY_STREAMS},
        "kinematic":  {"ECM": 0, "PSM1": 0, "PSM2": 0},
    }
    # Per-topic presence counters.
    topic_counters: dict[str, dict[str, int]] = {
        "kinematic.ECM":   {c: 0 for c in ECM_TOPIC_COLS},
        "kinematic.PSM1":  {c: 0 for c in PSM_TOPIC_COLS},
        "kinematic.PSM2":  {c: 0 for c in PSM_TOPIC_COLS},
        "annotation":      {c: 0 for c in ANNOTATION_TOPIC_COLS},
        "timestamp":       {c: 0 for c in TIMESTAMP_TOPIC_COLS},
    }
    n_with_annotation = 0
    n_with_calibrated_kin = 0
    n_with_gesture_partial = 0
    per_episode: list[dict[str, Any]] = []

    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir() or not (dataset_dir / "episodes").is_dir():
            continue
        for task_dir in (dataset_dir / "episodes").iterdir():
            if not task_dir.is_dir():
                continue
            for ep_dir in task_dir.iterdir():
                # Skip in-flight/crashed episodes — the completion
                # manifest is the only "shippable" signal.
                if not (ep_dir / ".surgsync_complete.json").is_file():
                    continue
                mp = ep_dir / "modalities.json"
                if not mp.is_file():
                    continue
                with open(mp) as f:
                    mods = json.load(f)
                n_episodes += 1

                # Stream-level presence
                for cat in ("video", "video_raw", "preprocess", "kinematic"):
                    block = mods.get(cat, {})
                    for stream, info in block.items():
                        if isinstance(info, dict) and info.get("present"):
                            counters[cat][stream] = counters[cat].get(stream, 0) + 1

                # Topic-level presence (kinematic arms + annotation + timestamp)
                for arm in ("ECM", "PSM1", "PSM2"):
                    topics = mods.get("kinematic", {}).get(arm, {}).get("topics", {})
                    for topic, status in topics.items():
                        if status.get("present"):
                            d = topic_counters[f"kinematic.{arm}"]
                            d[topic] = d.get(topic, 0) + 1
                for cat in ("annotation", "timestamp"):
                    topics = mods.get(cat, {}).get("topics", {})
                    for topic, status in topics.items():
                        if status.get("present"):
                            d = topic_counters[cat]
                            d[topic] = d.get(topic, 0) + 1

                summary = mods.get("summary", {})
                if summary.get("has_annotation"):
                    n_with_annotation += 1
                if summary.get("has_calibrated_kinematic"):
                    n_with_calibrated_kin += 1
                if summary.get("gesture_partial"):
                    n_with_gesture_partial += 1
                per_episode.append({
                    "episode_id": mods.get("episode_id"),
                    "summary":    summary,
                    "path":       str(mp.relative_to(dataset_root)),
                })

    return {
        "schema_version": SCHEMA_VERSION,
        "n_episodes": n_episodes,
        "streams_present_in_n_episodes": counters,
        "topics_present_in_n_episodes": topic_counters,
        "n_with_annotation": n_with_annotation,
        "n_with_calibrated_kinematic": n_with_calibrated_kin,
        "n_with_gesture_partial": n_with_gesture_partial,
        "episodes": sorted(per_episode, key=lambda e: e.get("episode_id") or ""),
    }


def write_aggregate_modalities(dataset_root: Path) -> Path:
    """Write `<dataset_root>/meta/modalities.json`."""
    agg = aggregate_modalities(dataset_root)
    dst = Path(dataset_root) / "meta" / "modalities.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(agg, indent=2))
    os.replace(tmp, dst)
    return dst
