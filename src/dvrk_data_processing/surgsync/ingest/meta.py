"""Thin wrapper around `serde/meta_io.py` so the ingest layer has a
uniform shape (each module = one raw concept)."""
from __future__ import annotations
from pathlib import Path

from dvrk_data_processing.surgsync.serde.meta_io import ClipMeta, load_clip_meta


def load_meta(meta_path: Path) -> ClipMeta:
    """Load `<raw_dir>/meta_data.json` into a ClipMeta."""
    return load_clip_meta(meta_path)
