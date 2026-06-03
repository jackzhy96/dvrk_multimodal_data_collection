"""Option-C preview writer — H.264 8-bit colorized preprocess videos.

Stub for the MVP. Source PNGs (disparity_image, optical_flow image,
heatmap viz) already exist under `processed_dir/<stage>/...`; a later
pass can stitch them through `encode_h264_crf` to produce smaller
preview-quality videos.
"""
from __future__ import annotations
import logging
from pathlib import Path


log = logging.getLogger(__name__)


def write_preview(processed_dir: Path, dst_dir: Path, *, fps: float) -> None:
    """Stub — Option C preview writer (not implemented for MVP)."""
    log.info("preview encoder is a stub for the MVP — skipping %s", dst_dir)
