"""Encode dense preprocess visualizations into MKV/FFV1 — bit-exact 8-bit.

The preprocessing pipeline emits 8-bit colorized PNGs for every
preprocess stream (depth,
optical flow, kinematic heatmap); this module wraps them in FFV1 so
the on-disk pixels round-trip bit-exact through the pack → unpack
cycle.

Sources:
  depth_estimation/depth_image/<i>.png             (INFERNO colormap)
  optical_flow/<cam>/image/<i>.png                 (color flow viz)
  kinematic_reproject/<PSM>/<cam>/image/<i>.png    (gray8 MinMax)

Outputs (under `<episode>/preprocess/`):
  depth.mkv                FFV1 bgr24
  flow_{left,right}.mkv    FFV1 bgr24
  heatmap_<PSM>_<cam>.mkv  FFV1 gray8
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from dvrk_data_processing.surgsync.encode.codec import encode_mkv_ffv1
from dvrk_data_processing.surgsync.ingest.clip import sorted_frames


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-stream PNG → MKV encoder
# ---------------------------------------------------------------------------

def _png_dir_to_mkv(
    src_dir: Path,
    dst_path: Path,
    *,
    fps: float,
    n_frames_expected: int,
    pix_fmt: str = "bgr24",
    log_label: str,
) -> int:
    """Read every `<i>.png` in `src_dir` (sorted numerically), pad
    missing frames with zeros so the MKV's frame count matches
    `n_frames_expected`, and write FFV1.

    Returns the count of real (non-padded) frames written.
    """
    files = sorted_frames(src_dir, suffix=".png")
    if not files:
        log.info("%s: no PNGs under %s — skipping", log_label, src_dir)
        return 0

    by_idx: dict[int, Path] = {int(p.stem): p for p in files}
    # Probe the first available frame once to know the shape we should
    # pad missing frames to. cv2 returns (H, W, 3) for bgr24 or (H, W)
    # for gray8 — both supported by FFV1.
    first_p = next(iter(by_idx.values()))
    probe = cv2.imread(
        str(first_p),
        cv2.IMREAD_COLOR if pix_fmt == "bgr24" else cv2.IMREAD_GRAYSCALE,
    )
    if probe is None:
        raise IOError(f"failed to decode probe PNG {first_p}")
    ref_shape = probe.shape

    n_real = 0

    def _iter() -> Iterator[np.ndarray]:
        nonlocal n_real
        for i in range(n_frames_expected):
            p = by_idx.get(i)
            if p is None:
                # Missing frame — pad with zeros to keep the MKV frame
                # count aligned with the master timeline.
                yield np.zeros(ref_shape, dtype=np.uint8)
                continue
            img = cv2.imread(
                str(p),
                cv2.IMREAD_COLOR if pix_fmt == "bgr24" else cv2.IMREAD_GRAYSCALE,
            )
            if img is None:
                raise IOError(f"failed to decode PNG {p}")
            if img.shape != ref_shape:
                raise ValueError(
                    f"{log_label} {p}: shape {img.shape} != reference {ref_shape}"
                )
            yield img
            n_real += 1

    encode_mkv_ffv1(_iter(), dst_path, fps=fps, pix_fmt=pix_fmt)
    return n_real


# ---------------------------------------------------------------------------
# Per-stream public writers
# ---------------------------------------------------------------------------

def write_depth(
    processed_dir: Path,
    dst_path: Path,
    *,
    fps: float,
    n_frames_expected: int,
) -> int:
    """Encode `depth_estimation/depth_image/<i>.png` into FFV1 bgr24.

    Returns the count of real frames written (excluding padded gaps).
    Returns 0 (and skips writing the MKV) if the source dir is absent.
    """
    src = processed_dir / "depth_estimation" / "depth_image"
    if not src.exists():
        log.info("depth source not present at %s — skipping", src)
        return 0
    return _png_dir_to_mkv(
        src, dst_path,
        fps=fps, n_frames_expected=n_frames_expected,
        pix_fmt="bgr24", log_label="depth",
    )


def write_flow(
    processed_dir: Path,
    cam: str,
    dst_path: Path,
    *,
    fps: float,
    n_frames_expected: int,
) -> int:
    """Encode `optical_flow/<cam>/image/<i>.png` into FFV1 bgr24.

    Note: preprocessing's optical_flow produces N-1 visualization PNGs for N
    source frames (flow is between consecutive frames). The trailing
    frame is zero-padded so the MKV row count matches `frames.parquet`.
    """
    src = processed_dir / "optical_flow" / cam / "image"
    if not src.exists():
        log.info("flow source not present at %s — skipping", src)
        return 0
    return _png_dir_to_mkv(
        src, dst_path,
        fps=fps, n_frames_expected=n_frames_expected,
        pix_fmt="bgr24", log_label=f"flow_{cam}",
    )


def write_heatmap(
    processed_dir: Path,
    arm: str,
    cam: str,
    dst_path: Path,
    *,
    fps: float,
    n_frames_expected: int,
) -> int:
    """Encode `kinematic_reproject/<arm>/<cam>/image/<i>.png` into FFV1
    gray8 (the PNGs are already 8-bit MinMax-normalized at the source).
    """
    src = processed_dir / "kinematic_reproject" / arm / cam / "image"
    if not src.exists():
        log.info("heatmap source not present at %s — skipping", src)
        return 0
    return _png_dir_to_mkv(
        src, dst_path,
        fps=fps, n_frames_expected=n_frames_expected,
        pix_fmt="gray8", log_label=f"heatmap_{arm}_{cam}",
    )


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def write_preprocess(
    processed_dir: Path,
    dst_dir: Path,
    *,
    fps: float,
    n_frames_expected: int,
) -> dict[str, int]:
    """Encode every available preprocess stream from preprocessing visualization PNGs.

    Up to 7 independent streams (1 depth + 2 flow + 4 heatmaps). Each
    is its own ffmpeg subprocess, so we dispatch them through a
    ThreadPoolExecutor and let ffmpeg saturate the available cores.

    Returns a `{stream_name: n_real_frames}` dict for the structured
    log. Streams whose source is absent are skipped silently and don't
    appear in the returned dict.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Build a job list of (stream_name, callable) pairs; each callable
    # returns the frame count it wrote. write_* helpers internally
    # short-circuit to 0 when the source is missing, so we don't need
    # to probe disk twice — just submit everything.
    jobs: list[tuple[str, callable]] = [
        ("depth",
         lambda: write_depth(processed_dir, dst_dir / "depth.mkv",
                             fps=fps, n_frames_expected=n_frames_expected)),
    ]
    for cam in ("left", "right"):
        jobs.append((
            f"flow_{cam}",
            lambda cam=cam: write_flow(  # `cam=cam` closes over the loop var
                processed_dir, cam, dst_dir / f"flow_{cam}.mkv",
                fps=fps, n_frames_expected=n_frames_expected,
            ),
        ))
    for arm in ("PSM1", "PSM2"):
        for cam in ("left", "right"):
            jobs.append((
                f"heatmap_{arm}_{cam}",
                lambda arm=arm, cam=cam: write_heatmap(
                    processed_dir, arm, cam,
                    dst_dir / f"heatmap_{arm}_{cam}.mkv",
                    fps=fps, n_frames_expected=n_frames_expected,
                ),
            ))

    written: dict[str, int] = {}
    # Cap at the smaller of (job count) and (8 workers). 8 is a
    # reasonable upper bound for a CPU with 8+ cores; higher just
    # oversubscribes ffmpeg subprocesses.
    with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
        future_to_name = {pool.submit(fn): name for name, fn in jobs}
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            n = fut.result()  # surface encoder failures
            if n > 0:
                written[name] = n
    return written
