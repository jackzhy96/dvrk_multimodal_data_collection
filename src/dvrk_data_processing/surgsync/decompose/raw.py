"""Raw-domain writer for the decomposer.

Writes the pre-pack raw layout under `<out_clip_dir>/`:
  image/{left,right,side}/<i>.png
  kinematic/{ECM,PSM1,PSM2}/<i>.json
  annotation/{contact_detection,phase,step,gesture}/<i>.json
  time_syn/<i>.json
  camera_calibration/, hand_eye_calibration/, meta_data.json
"""
from __future__ import annotations
import json
import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from dvrk_data_processing.surgsync.load.episode import Episode
from dvrk_data_processing.surgsync.serde.kinematic_io import (
    KinematicSample, CartesianSnapshot, TwistSnapshot, JointSnapshot,
    kinematic_sample_to_raw_dict,
)
from dvrk_data_processing.surgsync.serde.meta_io import ClipMeta, clip_meta_to_dict


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounded PNG writer pool
# ---------------------------------------------------------------------------

_RAW_VIDEO_TO_IMAGE_DIR: dict[str, str] = {
    "stereo_left":  "left",
    "stereo_right": "right",
    "side":         "side",
}

_QUEUE_SENTINEL = object()


def _png_write(args: tuple[Path, np.ndarray]) -> None:
    path, frame = args
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(f"cv2.imwrite failed at {path}")


def _bounded_png_pool(
    items: Iterable[tuple[Path, np.ndarray]],
    *,
    workers: int,
    queue_depth_factor: int = 4,
) -> None:
    """Encode PNGs through a bounded thread pool.

    The queue size caps in-flight frames so memory stays flat on
    multi-thousand-frame clips. Workers drain the queue; the producer
    blocks when full.
    """
    workers = max(1, workers)
    work_q: "queue.Queue[object]" = queue.Queue(maxsize=workers * max(1, queue_depth_factor))
    err_q: "queue.Queue[BaseException]" = queue.Queue()

    def _consumer() -> None:
        while True:
            item = work_q.get()
            try:
                if item is _QUEUE_SENTINEL:
                    return
                try:
                    _png_write(item)
                except BaseException as e:
                    err_q.put(e)
            finally:
                work_q.task_done()

    threads = [threading.Thread(target=_consumer, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()

    producer_exc: Optional[BaseException] = None
    try:
        for item in items:
            work_q.put(item)
    except BaseException as e:
        producer_exc = e
    finally:
        for _ in threads:
            work_q.put(_QUEUE_SENTINEL)
        for t in threads:
            t.join()

    if producer_exc is not None:
        raise producer_exc
    if not err_q.empty():
        first = err_q.get()
        while not err_q.empty():
            log.error("additional png write error: %s", err_q.get())
        raise first


# ---------------------------------------------------------------------------
# image/  ← video_raw/*.mkv
# ---------------------------------------------------------------------------

def write_raw_images(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
    workers: int = 4,
) -> dict[str, int]:
    """Decode each `video_raw/*.mkv` and dump PNGs under `image/<cam>/`.

    Output filenames use `source_indices[i]` so the layout matches the
    original raw clip's frame numbering.
    """
    from time import time as _now

    written: dict[str, int] = {}
    for stream in episode.available_videos()["video_raw"]:
        cam = _RAW_VIDEO_TO_IMAGE_DIR.get(stream, stream)
        cam_dir = out_clip_dir / "image" / cam
        view = episode.video_raw(stream)
        if view is None:
            continue
        cam_dir.mkdir(parents=True, exist_ok=True)
        t0 = _now()
        count = [0]

        def _produce():
            for i, frame in enumerate(view.iter_frames()):
                if i >= len(source_indices):
                    log.warning("%s: more decoded frames than master rows; truncating",
                                view.path)
                    break
                src_idx = int(source_indices[i])
                count[0] += 1
                yield (cam_dir / f"{src_idx}.png", frame)

        _bounded_png_pool(_produce(), workers=workers)
        elapsed = _now() - t0
        log.info("image/%s: %d PNGs in %.1fs (%.1f fps)",
                 cam, count[0], elapsed, count[0] / max(elapsed, 1e-9))
        written[cam] = count[0]
    return written


# ---------------------------------------------------------------------------
# kinematic/  ← per-arm parquet
# ---------------------------------------------------------------------------

_KIN_JSON_BATCH = 1024


def _list_or_none(value):
    if value is None:
        return None
    out = list(value)
    return None if not out else [float(x) for x in out]


def _scalar_or_none(value):
    return None if value is None else float(value)


def _row_to_kinematic_sample(row: dict, arm: str, src_idx: int) -> KinematicSample:
    measured_cp = CartesianSnapshot(
        position=_list_or_none(row.get("measured_cp.position")),
        orientation=_list_or_none(row.get("measured_cp.orientation")),
        velocity=_list_or_none(row.get("measured_cp.velocity")),
    )
    local_measured_cp = CartesianSnapshot(
        position=_list_or_none(row.get("local_measured_cp.position")),
        orientation=_list_or_none(row.get("local_measured_cp.orientation")),
    )
    measured_cv = TwistSnapshot(
        linear=_list_or_none(row.get("measured_cv.linear")),
        angular=_list_or_none(row.get("measured_cv.angular")),
    )
    measured_js = JointSnapshot(
        position=_list_or_none(row.get("measured_js.position")),
        velocity=_list_or_none(row.get("measured_js.velocity")),
        effort=_list_or_none(row.get("measured_js.effort")),
    )
    setpoint_js = JointSnapshot(
        position=_list_or_none(row.get("setpoint_js.position")),
        velocity=_list_or_none(row.get("setpoint_js.velocity")),
        effort=_list_or_none(row.get("setpoint_js.effort")),
    )

    setpoint_cp: Optional[CartesianSnapshot] = None
    sp_pos = _list_or_none(row.get("setpoint_cp.position"))
    sp_ori = _list_or_none(row.get("setpoint_cp.orientation"))
    if sp_pos is not None or sp_ori is not None:
        setpoint_cp = CartesianSnapshot(position=sp_pos, orientation=sp_ori)

    return KinematicSample(
        frame=src_idx,
        arm_name=arm,
        measured_js=measured_js,
        setpoint_js=setpoint_js,
        measured_cp=measured_cp,
        local_measured_cp=local_measured_cp,
        measured_cv=measured_cv,
        setpoint_cp=setpoint_cp,
        measured_jaw_position=_scalar_or_none(row.get("jaw.measured_position")),
        setpoint_jaw_position=_scalar_or_none(row.get("jaw.setpoint_position")),
        source_frequency_hz=_scalar_or_none(row.get("source_frequency_hz")),
    )


def write_kinematic_jsons(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
) -> dict[str, int]:
    """Per-frame `kinematic/<arm>/<src_idx>.json` from the per-arm parquets.

    Layout matches the raw form including the per-recorder difference
    in jaw/measured_frequency placement (online: top-level; offline:
    nested inside `arm`).
    """
    from time import time as _now

    jaw_freq_loc = "inside_arm" if episode.recorder_variant == "offline" else "top_level"

    written: dict[str, int] = {}
    for arm in ("ECM", "PSM1", "PSM2"):
        t0 = _now()
        table = episode.arm(arm)
        arm_dir = out_clip_dir / "kinematic" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        global_i = 0
        for batch in table.to_batches(max_chunksize=_KIN_JSON_BATCH):
            for row in batch.to_pylist():
                src_idx = int(source_indices[global_i])
                global_i += 1
                sample = _row_to_kinematic_sample(row, arm, src_idx)
                payload = kinematic_sample_to_raw_dict(
                    sample, jaw_freq_location=jaw_freq_loc,
                )
                with open(arm_dir / f"{src_idx}.json", "w") as f:
                    json.dump(payload, f, indent=4)
        written[arm] = global_i
        log.info("kinematic/%s: %d JSONs in %.1fs", arm, global_i, _now() - t0)
    return written


# ---------------------------------------------------------------------------
# annotation/  ← annotation parquet (text form)
# ---------------------------------------------------------------------------

def write_annotation_jsons(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
) -> dict[str, int]:
    """Per-frame annotation JSONs. phase/step/gesture cells carry the
    verbalized text (the packer's vocab lookup is one-way; the original
    numeric id is not preserved)."""
    ann = episode.annotation
    cols = {name: ann.column(name).to_pylist() for name in ann.column_names}

    contact_dir = out_clip_dir / "annotation" / "contact_detection"
    gesture_dir = out_clip_dir / "annotation" / "gesture"
    phase_dir   = out_clip_dir / "annotation" / "phase"
    step_dir    = out_clip_dir / "annotation" / "step"
    for d in (contact_dir, gesture_dir, phase_dir, step_dir):
        d.mkdir(parents=True, exist_ok=True)

    written = {"contact_detection": 0, "gesture": 0, "phase": 0, "step": 0}
    n = ann.num_rows

    for i in range(n):
        src_idx = int(source_indices[i])
        c1 = cols.get("contact.PSM1", [None] * n)[i]
        c2 = cols.get("contact.PSM2", [None] * n)[i]
        g1 = cols.get("gesture.PSM1", [None] * n)[i]
        g2 = cols.get("gesture.PSM2", [None] * n)[i]
        ph = cols.get("phase",        [None] * n)[i]
        st = cols.get("step",         [None] * n)[i]

        with open(contact_dir / f"{src_idx}.json", "w") as f:
            json.dump({"PSM1": int(c1 or 0), "PSM2": int(c2 or 0)}, f)
        written["contact_detection"] += 1

        if g1 is not None or g2 is not None:
            with open(gesture_dir / f"{src_idx}.json", "w") as f:
                json.dump({"gesture": {"PSM1": g1, "PSM2": g2}}, f, indent=2)
            written["gesture"] += 1

        with open(phase_dir / f"{src_idx}.json", "w") as f:
            json.dump({"phase": ph}, f, indent=2)
        written["phase"] += 1

        with open(step_dir / f"{src_idx}.json", "w") as f:
            json.dump({"step": st}, f, indent=2)
        written["step"] += 1

    return written


# ---------------------------------------------------------------------------
# time_syn/  ← timestamp parquet
# ---------------------------------------------------------------------------

def _ns_to_stamp_dict(ns: int) -> dict:
    sec = int(ns // 1_000_000_000)
    nsec = int(ns - sec * 1_000_000_000)
    return {"sec": sec, "nsec": nsec}


def _arm_block(
    *,
    arm: str,
    master_ns: int,
    deltas: dict[str, int],
    null_mask: dict[str, bool],
    measured_freq: Optional[float],
    include_jaw: bool,
) -> dict:
    """One arm's per-frame stamp dict matching the raw time_syn layout.

    `header_cv_stamp` and `reference_js_stamp` aren't tracked in the
    packer topic catalog — reconstruct them by mirroring the closest
    semantically-equivalent topic (measured_cp / setpoint_js).
    """
    def stamp(topic_short: str) -> Optional[dict]:
        topic_full = f"{arm}.{topic_short}"
        if null_mask.get(topic_full, True):
            return None
        return _ns_to_stamp_dict(master_ns + deltas[topic_full])

    measured = {}
    for raw_key, topic_short in (
        ("local_measured_cp_stamp", "local_measured_cp"),
        ("measured_cp_stamp",       "measured_cp"),
        ("measured_cv_stamp",       "measured_cv"),
        ("measured_js_stamp",       "measured_js"),
    ):
        s = stamp(topic_short)
        if s is not None:
            measured[raw_key] = s

    setpoint = {}
    for raw_key, topic_short in (
        ("setpoint_cp_stamp", "setpoint_cp"),
        ("setpoint_js_stamp", "setpoint_js"),
    ):
        s = stamp(topic_short)
        if s is not None:
            setpoint[raw_key] = s

    block: dict = {}
    hdr = stamp("measured_cp") or stamp("measured_js")
    if hdr is not None:
        block["header_cv_stamp"] = hdr
    if measured:
        block["measured_data"] = measured
    ref = stamp("setpoint_js") or stamp("measured_js")
    if ref is not None:
        block["reference_js_stamp"] = ref
    if setpoint:
        block["setpoint_data"] = setpoint

    if include_jaw:
        jaw = {}
        m_jaw = stamp("jaw_measured")
        s_jaw = stamp("jaw_setpoint")
        if m_jaw is not None:
            jaw["measured_stamp"] = m_jaw
        if s_jaw is not None:
            jaw["setpoint_stamp"] = s_jaw
        if jaw:
            block["jaw"] = jaw

    if measured_freq is not None:
        block["measured_frequency"] = float(measured_freq)

    return block


def write_time_syn_jsons(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
) -> int:
    """Reconstruct every time_syn JSON from `timestamp.parquet`.

        master_ns_abs[i]  = master_t0_ns + master_timestamp_ns[i]
        <topic>_stamp[i]  = master_ns_abs[i] + delta_to_master.<topic>_ns[i]
    """
    ts = episode.timestamps
    master_t0 = episode.master_t0_ns
    ts_cols = {n: ts.column(n).to_pylist() for n in ts.column_names}

    topic_names: list[str] = []
    delta_lists: dict[str, list[Optional[int]]] = {}
    for cn in ts.column_names:
        if cn.startswith("delta_to_master.") and cn.endswith("_ns"):
            topic = cn[len("delta_to_master."): -len("_ns")]
            topic_names.append(topic)
            delta_lists[topic] = ts_cols[cn]

    arm_freq: dict[str, list[Optional[float]]] = {}
    for arm in ("PSM1", "PSM2"):
        if "source_frequency_hz" in episode.arm(arm).column_names:
            arm_freq[arm] = episode.arm(arm).column("source_frequency_hz").to_pylist()
        else:
            arm_freq[arm] = [None] * episode.length

    time_syn_dir = out_clip_dir / "time_syn"
    time_syn_dir.mkdir(parents=True, exist_ok=True)

    n = ts.num_rows
    for i in range(n):
        master_ns_abs = master_t0 + int(ts_cols["master_timestamp_ns"][i])
        deltas: dict[str, int] = {}
        null_mask: dict[str, bool] = {}
        for t in topic_names:
            v = delta_lists[t][i]
            null_mask[t] = (v is None)
            deltas[t] = int(v) if v is not None else 0
        src_idx = int(source_indices[i])

        payload: dict = {
            "Kinematics_set_1": {
                "ECM":  _arm_block(arm="ECM",  master_ns=master_ns_abs,
                                   deltas=deltas, null_mask=null_mask,
                                   measured_freq=None, include_jaw=False),
                "PSM1": _arm_block(arm="PSM1", master_ns=master_ns_abs,
                                   deltas=deltas, null_mask=null_mask,
                                   measured_freq=arm_freq["PSM1"][i],
                                   include_jaw=True),
                "PSM2": _arm_block(arm="PSM2", master_ns=master_ns_abs,
                                   deltas=deltas, null_mask=null_mask,
                                   measured_freq=arm_freq["PSM2"][i],
                                   include_jaw=True),
            },
            "frame": src_idx,
            "image_left_stamp": _ns_to_stamp_dict(master_ns_abs),
        }
        if not null_mask.get("image_right", True):
            payload["image_right_stamp"] = _ns_to_stamp_dict(
                master_ns_abs + int(deltas["image_right"])
            )
        if not null_mask.get("image_side", True):
            payload["side_image_1_stamp"] = _ns_to_stamp_dict(
                master_ns_abs + int(deltas["image_side"])
            )

        with open(time_syn_dir / f"{src_idx}.json", "w") as f:
            json.dump(payload, f, indent=4, sort_keys=True)

    return n


# ---------------------------------------------------------------------------
# calibration/ + meta_data.json
# ---------------------------------------------------------------------------

def write_calibration_and_meta(episode: Episode, out_clip_dir: Path) -> dict[str, int]:
    """Copy calibration files; reconstruct meta_data.json from episode_meta."""
    written = {"camera_calibration": 0, "hand_eye_calibration": 0, "meta_data": 0}
    cal = episode.calibration

    cam_dir = out_clip_dir / "camera_calibration"
    cam_dir.mkdir(parents=True, exist_ok=True)
    for src in (cal.left_yaml, cal.right_yaml, cal.stereo_calib_json):
        if src is None or not src.exists():
            continue
        shutil.copy2(src, cam_dir / src.name)
        written["camera_calibration"] += 1

    if cal.hand_eye_dir is not None:
        he_dir = out_clip_dir / "hand_eye_calibration"
        he_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(cal.hand_eye_dir.iterdir()):
            if src.is_file():
                shutil.copy2(src, he_dir / src.name)
                written["hand_eye_calibration"] += 1

    em = episode.meta
    cm = ClipMeta(
        user_id=str(em.get("source_clip", "").rstrip("/").rsplit("/", 1)[-1] or None),
        operator_skill_level=em.get("operator_skill_level"),
        case_type=em.get("case_type"),
        tool=em.get("tool") or {"PSM1": None, "PSM2": None},
        failure=list(em.get("failure_episodes", [])),
        recovery=list(em.get("recovery_episodes", [])),
        extra={},
    )
    with open(out_clip_dir / "meta_data.json", "w") as f:
        json.dump(clip_meta_to_dict(cm), f, indent=2)
    written["meta_data"] = 1
    return written


# ---------------------------------------------------------------------------
# Top-level domain writer
# ---------------------------------------------------------------------------

def write_raw_domain(
    episode: Episode,
    out_clip_dir: Path,
    *,
    workers: int = 4,
) -> dict:
    """Write every artifact under the raw-domain inverse layout."""
    out_clip_dir = Path(out_clip_dir)
    out_clip_dir.mkdir(parents=True, exist_ok=True)

    src_idx_col = episode.timestamps.column("source_frame_index").to_pylist()
    source_indices: list[int] = [int(x) for x in src_idx_col]

    image_counts = write_raw_images(
        episode, out_clip_dir, source_indices=source_indices, workers=workers,
    )
    kin_counts = write_kinematic_jsons(
        episode, out_clip_dir, source_indices=source_indices,
    )
    ann_counts = write_annotation_jsons(
        episode, out_clip_dir, source_indices=source_indices,
    )
    ts_n = write_time_syn_jsons(
        episode, out_clip_dir, source_indices=source_indices,
    )
    cal_counts = write_calibration_and_meta(episode, out_clip_dir)

    return {
        "image":       image_counts,
        "kinematic":   kin_counts,
        "annotation":  ann_counts,
        "time_syn":    ts_n,
        "calibration": cal_counts,
    }
