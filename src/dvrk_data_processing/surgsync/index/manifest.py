"""Build `meta/manifest.json` — SHA256 of every file under the dataset
root except `manifest.json` itself.

Last step of an index build — the manifest covers everything else
that was just written.
"""
from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dvrk_data_processing.surgsync.schema import (
    SCHEMA_VERSION, Manifest, ManifestFile,
)


log = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _skip_path(path: Path, dataset_root: Path) -> bool:
    """Skip internal-state files that aren't part of the published
    artifact: the manifest itself, the JSONL build log, and any
    staging dirs / sentinel markers that survived a crash."""
    rel = path.relative_to(dataset_root)
    parts = rel.parts
    if parts[0] in (".staging", ".logs", ".tmp"):
        return True
    if parts == ("meta", "manifest.json"):
        return True
    # Per-episode running / failed / complete sentinels are runtime
    # state, not shipped content. The manifest captures what's in the
    # release; the sentinels describe pack lifecycle.
    if path.name in (".surgsync_running.json",
                     ".surgsync_failed.json",
                     ".surgsync_complete.json"):
        return True
    return False


def build_manifest(dataset_root: Path, *, data_version: str = "dev") -> dict:
    """Walk every file under `dataset_root`, hash it, write
    `meta/manifest.json`. Returns a small summary dict."""
    dataset_root = Path(dataset_root)

    files: dict[str, ManifestFile] = {}
    total_bytes = 0
    for p in sorted(dataset_root.rglob("*")):
        if not p.is_file() or _skip_path(p, dataset_root):
            continue
        rel = p.relative_to(dataset_root).as_posix()
        size = p.stat().st_size
        sha = _sha256_file(p)
        files[rel] = ManifestFile(sha256=sha, size_bytes=size)
        total_bytes += size

    m = Manifest(
        schema_version=SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        data_version=data_version,
        algorithm="sha256",
        files=files,
        total_files=len(files),
        total_size_bytes=total_bytes,
    )
    out = dataset_root / "meta" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(m.model_dump_json(indent=2))
    log.info("build_manifest: %d files, %.2f MB → %s",
             len(files), total_bytes / 1024 / 1024, out)
    return {"n_files": len(files), "total_bytes": total_bytes}
