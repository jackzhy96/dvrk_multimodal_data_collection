"""Encode raw PNGs from `raw_dir` into MKV/FFV1 — **bit-exact** round-trip.

Per the packer invertibility contract, this is **mandatory** in every
release (see `tasks/M2-packing.md`). The unpack stage reconstructs
`image/{left,right,side*}/<frame>.png` from these MKVs.

The three streams (stereo_left / stereo_right / side) are independent —
each is its own ffmpeg subprocess. We dispatch them via a small
ThreadPoolExecutor so all three encoders run concurrently. ffmpeg
releases the GIL while running (it's a subprocess), so threads
suffice; no need for ProcessPoolExecutor here.
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dvrk_data_processing.surgsync.encode.codec import encode_mkv_ffv1
from dvrk_data_processing.surgsync.encode.video_processed import _iter_pngs_as_bgr


log = logging.getLogger(__name__)


def _encode_one_stream(name: str, src: Path, out: Path, fps: float) -> tuple[str, Path]:
    """Encode one camera stream. Runs in its own thread."""
    log.info("encoding raw %s → %s (FFV1, bit-exact)", name, out)
    encode_mkv_ffv1(_iter_pngs_as_bgr(src), out, fps=fps, pix_fmt="bgr24")
    return name, out


def write_raw_videos(
    raw_dir: Path,
    dst_dir: Path,
    *,
    fps: float,
    side_dir_name: str,
) -> dict[str, Path]:
    """Encode raw stereo + side cameras into FFV1, in parallel.

    `side_dir_name` is `"side"` for online clips and `"side1"` for
    offline (see `specs/raw_data_spec.md`). When the side folder
    doesn't exist (some clips have stereo only) it's silently skipped.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    streams = {
        "stereo_left":  raw_dir / "image" / "left",
        "stereo_right": raw_dir / "image" / "right",
        "side":         raw_dir / "image" / side_dir_name,
    }
    # Filter out missing streams before submitting work.
    to_encode: list[tuple[str, Path, Path]] = []
    for name, src in streams.items():
        if not src.exists() or not any(src.iterdir()):
            if name == "side":
                log.info("no side camera at %s — skipping (clip is stereo-only)", src)
                continue
            log.warning("raw video src missing: %s — skipping %s", src, name)
            continue
        to_encode.append((name, src, dst_dir / f"{name}.mkv"))

    written: dict[str, Path] = {}
    if not to_encode:
        return written

    # Three streams max in this module → cap workers at 3. Each
    # ffmpeg subprocess saturates ~1 CPU core for FFV1; running all
    # three concurrently is the sweet spot on a 4+ core box.
    with ThreadPoolExecutor(max_workers=len(to_encode)) as pool:
        futures = [
            pool.submit(_encode_one_stream, name, src, out, fps)
            for name, src, out in to_encode
        ]
        for fut in as_completed(futures):
            name, out = fut.result()  # propagate any encode failure
            written[name] = out
    return written
