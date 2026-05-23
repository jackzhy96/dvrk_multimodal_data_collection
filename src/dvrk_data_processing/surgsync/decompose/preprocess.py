"""Preprocess-domain writer for the decomposer.

Writes the preprocessing tree under `<out_clip_dir>/preprocess/`:
  rectify_resize/image/{left,right}/<i>.png  + copies of kinematic + time_syn
  depth_estimation/depth_image/<i>.png
  optical_flow/{left,right}/image/<i>.png
  kinematic_reproject/{PSM1,PSM2}/{left,right}/image/<i>.png
  kinematic_reproject/{PSM1,PSM2}/calibrated_kinematic/<i>.json

Raw `.npy` derivatives are not recoverable — re-run preprocessing if you need them.
"""
from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from dvrk_data_processing.surgsync.decompose.raw import _bounded_png_pool
from dvrk_data_processing.surgsync.load.episode import Episode
from dvrk_data_processing.surgsync.serde.calibrated_kinematic_io import (
    CalibratedKinematicSample, calibrated_sample_to_dict,
)


log = logging.getLogger(__name__)


_PROCESSED_TO_IMAGE_DIR: dict[str, str] = {
    "stereo_left":  "left",
    "stereo_right": "right",
}


def _decode_video_to_pngs(
    view,
    out_dir: Path,
    *,
    source_indices: list[int],
    workers: int = 4,
    label: str = "stream",
) -> int:
    """Decode `view`, write PNGs named by source frame index. Returns count."""
    from time import time as _now

    if view is None:
        log.info("%s: video stream absent — skipping", label)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    count = [0]
    t0 = _now()

    def _produce():
        for i, frame in enumerate(view.iter_frames()):
            if i >= len(source_indices):
                log.warning("%s: decoded frame %d exceeds master row count; truncating",
                            label, i)
                break
            src_idx = int(source_indices[i])
            count[0] += 1
            yield (out_dir / f"{src_idx}.png", frame)

    _bounded_png_pool(_produce(), workers=workers)
    elapsed = _now() - t0
    log.info("%s: %d PNGs in %.1fs (%.1f fps)",
             label, count[0], elapsed, count[0] / max(elapsed, 1e-9))
    return count[0]


# ---------------------------------------------------------------------------
# preprocess/rectify_resize/
# ---------------------------------------------------------------------------

def _mirror_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def write_rectify_resize(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
    workers: int = 4,
) -> dict[str, int]:
    """Rectified images from H.264 MP4 + calibration + kinematic/time_syn copies."""
    rr_dir = out_clip_dir / "preprocess" / "rectify_resize"
    rr_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {
        "image_left": 0, "image_right": 0, "calibration": 0,
        "kinematic_copy": 0, "time_syn_copy": 0,
    }

    for stream in ("stereo_left", "stereo_right"):
        v = episode.video(stream)
        cam = _PROCESSED_TO_IMAGE_DIR[stream]
        n = _decode_video_to_pngs(
            v, rr_dir / "image" / cam,
            source_indices=source_indices, workers=workers,
            label=f"rectify_resize/image/{cam}",
        )
        written[f"image_{cam}"] = n

    cal = episode.calibration
    cc_dir = rr_dir / "camera_calibration"
    cc_dir.mkdir(parents=True, exist_ok=True)
    for src in (cal.left_yaml, cal.right_yaml, cal.rectify_params_json):
        if src is not None and src.exists():
            shutil.copy2(src, cc_dir / src.name)
            written["calibration"] += 1

    raw_kin = out_clip_dir / "kinematic"
    raw_ts  = out_clip_dir / "time_syn"
    if raw_kin.is_dir():
        _mirror_tree(raw_kin, rr_dir / "kinematic")
        written["kinematic_copy"] = sum(1 for _ in (rr_dir / "kinematic").rglob("*.json"))
    if raw_ts.is_dir():
        _mirror_tree(raw_ts, rr_dir / "time_syn")
        written["time_syn_copy"] = sum(1 for _ in (rr_dir / "time_syn").rglob("*.json"))

    return written


# ---------------------------------------------------------------------------
# preprocess/depth_estimation/
# ---------------------------------------------------------------------------

def write_depth(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
    workers: int = 4,
) -> int:
    """Decode `preprocess/depth.mkv` → `preprocess/depth_estimation/depth_image/`."""
    return _decode_video_to_pngs(
        episode.preprocess("depth"),
        out_clip_dir / "preprocess" / "depth_estimation" / "depth_image",
        source_indices=source_indices, workers=workers,
        label="preprocess/depth_estimation/depth_image",
    )


# ---------------------------------------------------------------------------
# preprocess/optical_flow/
# ---------------------------------------------------------------------------

def write_optical_flow(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
    workers: int = 4,
) -> dict[str, int]:
    """Decode `preprocess/flow_*.mkv` → `preprocess/optical_flow/<cam>/image/`."""
    out: dict[str, int] = {}
    for cam in ("left", "right"):
        out[cam] = _decode_video_to_pngs(
            episode.preprocess(f"flow_{cam}"),
            out_clip_dir / "preprocess" / "optical_flow" / cam / "image",
            source_indices=source_indices, workers=workers,
            label=f"preprocess/optical_flow/{cam}/image",
        )
    return out


# ---------------------------------------------------------------------------
# preprocess/kinematic_reproject/
# ---------------------------------------------------------------------------

def _write_calibrated_kinematic(
    episode: Episode,
    arm: str,
    out_dir: Path,
    *,
    source_indices: list[int],
) -> int:
    table = episode.arm(arm)
    needed = (
        "measured_cp_calibrated.position",
        "measured_cp_calibrated.orientation",
        "setpoint_cp_calibrated.position",
        "setpoint_cp_calibrated.orientation",
    )
    if not any(c in table.column_names for c in needed):
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = {c: table.column(c).to_pylist() for c in needed if c in table.column_names}
    n_written = 0
    for i in range(table.num_rows):
        m_pos = cols.get("measured_cp_calibrated.position",   [None])[i] if cols.get("measured_cp_calibrated.position") else None
        m_ori = cols.get("measured_cp_calibrated.orientation", [None])[i] if cols.get("measured_cp_calibrated.orientation") else None
        s_pos = cols.get("setpoint_cp_calibrated.position",    [None])[i] if cols.get("setpoint_cp_calibrated.position") else None
        s_ori = cols.get("setpoint_cp_calibrated.orientation", [None])[i] if cols.get("setpoint_cp_calibrated.orientation") else None

        if m_pos is None and s_pos is None:
            continue
        src_idx = int(source_indices[i])
        sample = CalibratedKinematicSample(
            frame=src_idx, arm_name=arm,
            measured_cp_calibrated=None if m_pos is None else (
                [float(x) for x in m_pos], [float(x) for x in (m_ori or [])],
            ),
            setpoint_cp_calibrated=None if s_pos is None else (
                [float(x) for x in s_pos], [float(x) for x in (s_ori or [])],
            ),
        )
        with open(out_dir / f"{src_idx}.json", "w") as f:
            json.dump(calibrated_sample_to_dict(sample), f, indent=2)
        n_written += 1
    return n_written


def write_kinematic_reproject(
    episode: Episode,
    out_clip_dir: Path,
    *,
    source_indices: list[int],
    workers: int = 4,
) -> dict[str, int]:
    """Heatmap PNGs + calibrated kinematic JSONs per PSM × camera."""
    out: dict[str, int] = {}
    for arm in ("PSM1", "PSM2"):
        for cam in ("left", "right"):
            out[f"heatmap_{arm}_{cam}"] = _decode_video_to_pngs(
                episode.preprocess(f"heatmap_{arm}_{cam}"),
                out_clip_dir / "preprocess" / "kinematic_reproject" / arm / cam / "image",
                source_indices=source_indices, workers=workers,
                label=f"preprocess/kinematic_reproject/{arm}/{cam}/image",
            )
        out[f"calibrated_kinematic_{arm}"] = _write_calibrated_kinematic(
            episode, arm,
            out_clip_dir / "preprocess" / "kinematic_reproject" / arm / "calibrated_kinematic",
            source_indices=source_indices,
        )
    return out


# ---------------------------------------------------------------------------
# Top-level domain writer
# ---------------------------------------------------------------------------

def write_preprocess_domain(
    episode: Episode,
    out_clip_dir: Path,
    *,
    workers: int = 4,
) -> dict:
    """Write the `preprocess/` inverse tree. Call after `write_raw_domain`
    so `rectify_resize/{kinematic,time_syn}/` can mirror the raw copies."""
    out_clip_dir = Path(out_clip_dir)
    src_idx_col = episode.timestamps.column("source_frame_index").to_pylist()
    source_indices: list[int] = [int(x) for x in src_idx_col]

    return {
        "rectify_resize":      write_rectify_resize(
            episode, out_clip_dir, source_indices=source_indices, workers=workers),
        "depth_estimation":    write_depth(
            episode, out_clip_dir, source_indices=source_indices, workers=workers),
        "optical_flow":        write_optical_flow(
            episode, out_clip_dir, source_indices=source_indices, workers=workers),
        "kinematic_reproject": write_kinematic_reproject(
            episode, out_clip_dir, source_indices=source_indices, workers=workers),
    }
