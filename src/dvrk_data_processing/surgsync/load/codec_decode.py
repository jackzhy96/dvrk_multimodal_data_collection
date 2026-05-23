"""ffprobe + ffmpeg-decode helpers for the reader / decomposer."""
from __future__ import annotations
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from dvrk_data_processing.surgsync.encode.codec import decode_video_frames


log = logging.getLogger(__name__)


def _ffprobe_bin() -> str:
    bin_path = shutil.which("ffprobe")
    if bin_path is None:
        raise RuntimeError("ffprobe not found on PATH.")
    return bin_path


def probe_video(path: Path, *, count_frames: bool = False) -> dict:
    """Return `{width, height, pix_fmt, codec_name, n_frames}`.

    `count_frames=True` makes ffprobe walk every frame (slow on big
    FFV1 files); default reads container headers only.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    fields = "stream=width,height,pix_fmt,codec_name,nb_frames"
    args = ["-v", "error", "-select_streams", "v:0"]
    if count_frames:
        args += ["-count_frames"]
        fields = "stream=width,height,pix_fmt,codec_name,nb_read_frames"
    args += ["-show_entries", fields, "-of", "json", str(path)]
    out = subprocess.run([_ffprobe_bin(), *args], check=True, capture_output=True)
    info = json.loads(out.stdout.decode())
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    s = streams[0]
    n_frames_raw = s.get("nb_read_frames") or s.get("nb_frames") or 0
    return {
        "width":      int(s["width"]),
        "height":     int(s["height"]),
        "pix_fmt":    s.get("pix_fmt", "bgr24"),
        "n_frames":   int(n_frames_raw) if str(n_frames_raw).isdigit() else 0,
        "codec_name": s.get("codec_name", ""),
    }


def _decode_pix_fmt(probe: dict) -> str:
    """Map an ffprobe pix_fmt to the format we ask ffmpeg to decode to.

    H.264 → bgr24 (packed RGB for PNG writers). FFV1 is decoded back
    to its native pix_fmt so the bytes round-trip exactly; bgr0/bgra
    are normalized to bgr24 (the alpha byte carries no information).
    """
    pf = probe["pix_fmt"]
    codec = probe["codec_name"]
    if codec == "h264":
        return "bgr24"
    if codec == "ffv1":
        if pf in {"bgr24", "gray8", "gray16le", "gbrp16le"}:
            return pf
        if pf == "gray":
            return "gray8"
        if pf in {"bgr0", "bgra", "rgb0", "rgba"}:
            return "bgr24"
        log.warning("unrecognized FFV1 pix_fmt %s — decoding as bgr24", pf)
        return "bgr24"
    return "bgr24"


def iter_frames(path: Path, *, probe: Optional[dict] = None) -> Iterator[np.ndarray]:
    """Yield every frame as a numpy array."""
    pr = probe or probe_video(path)
    pix = _decode_pix_fmt(pr)
    yield from decode_video_frames(path, pix_fmt=pix, width=pr["width"], height=pr["height"])


def decode_all_frames(path: Path) -> Tuple[list[np.ndarray], dict]:
    """Decode every frame into a list. Convenience for short clips + tests."""
    pr = probe_video(path)
    return list(iter_frames(path, probe=pr)), pr
