"""Encode rectified PNGs from `intermediate_dir` into MKV/H.264 CRF 18.

Output: `<staging>/video/stereo_{left,right}.mkv` (and optionally
`video/side.mkv` when a side stream is available, which isn't the case
for stage-1 output today — stage 1 only rectifies the stereo pair).

Visually lossless target — PSNR ≥ 40 dB on natural surgical video at
CRF 18 (the encoder's docstring contract). The end-to-end smoke test
verifies this against the round-trip diff.
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from dvrk_data_processing.surgsync.encode.codec import encode_h264_crf
from dvrk_data_processing.surgsync.ingest.clip import sorted_frames


log = logging.getLogger(__name__)


def _iter_pngs_as_bgr(folder: Path) -> Iterator[np.ndarray]:
    """Read sorted PNGs and yield bgr24 numpy arrays.

    cv2.imread returns BGR by default — that's the pix_fmt our H.264
    encoder consumes, so no conversion needed. Empty / missing folder
    yields nothing.
    """
    for p in sorted_frames(folder, suffix=".png"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"failed to decode {p}")
        yield img


def _encode_one_stream(name: str, src: Path, out: Path, fps: float, crf: int) -> tuple[str, Path]:
    """Run a single H.264 encode. Runs in its own thread."""
    log.info("encoding %s → %s", name, out)
    encode_h264_crf(_iter_pngs_as_bgr(src), out, fps=fps, crf=crf)
    return name, out


def write_processed_videos(
    intermediate_dir: Path,
    dst_dir: Path,
    *,
    fps: float,
    crf: int = 18,
) -> dict[str, Path]:
    """Encode stereo_left and stereo_right MKVs concurrently.

    The two streams are independent ffmpeg subprocesses; a small
    ThreadPoolExecutor lets them run in parallel without the GIL
    getting in the way (the heavy work is in the subprocess).
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    streams = {
        "stereo_left":  intermediate_dir / "image" / "left",
        "stereo_right": intermediate_dir / "image" / "right",
    }
    to_encode: list[tuple[str, Path, Path]] = []
    for name, src in streams.items():
        if not src.exists():
            log.warning("processed video src missing: %s — skipping %s", src, name)
            continue
        # MP4 container for H.264 — broader player support; the lossy
        # path is already irrecoverable from float pixels so MKV's
        # `data-preserves-everything` strength doesn't apply here. FFV1
        # raw video stays MKV (MP4 doesn't standardize FFV1).
        to_encode.append((name, src, dst_dir / f"{name}.mp4"))

    written: dict[str, Path] = {}
    if not to_encode:
        return written

    with ThreadPoolExecutor(max_workers=len(to_encode)) as pool:
        futures = [
            pool.submit(_encode_one_stream, name, src, out, fps, crf)
            for name, src, out in to_encode
        ]
        for fut in as_completed(futures):
            name, out = fut.result()
            written[name] = out
    return written
