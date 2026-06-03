"""Write `<episode>/time_sync_stat.json` via the TimeSyncStat pydantic model.

Per-topic latency detail — median / mean / std / max in ms, the
master-frame index of the max |delta|, and the count of present
stamps for each synced modality. Atomic temp + rename so a kill
mid-write leaves no partial file.

The data is computed in `align.aligner.align_clip` and attached to
`AlignedClip.per_topic_latency` (a plain dict-of-dicts). This module
validates that dict through `TimeSyncStat`/`PerTopicLatency` pydantic
models so any drift between the aligner output shape and the on-disk
schema fails fast at construction time, not at read time.
"""
from __future__ import annotations
import json
import logging
import os
import tempfile
from pathlib import Path

from dvrk_data_processing.surgsync.align.aligner import AlignedClip
from dvrk_data_processing.surgsync.schema import (
    SCHEMA_VERSION,
    TimeSyncStat,
    PerTopicLatency,
)


log = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(path.parent), suffix=".tmp", delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_time_sync_stat(
    *,
    aligned: AlignedClip,
    episode_id: str,
    dst_path: Path,
) -> TimeSyncStat:
    """Build and atomically write time_sync_stat.json.

    Returns the validated TimeSyncStat for the caller's structured log.
    """
    per_topic = {
        name: PerTopicLatency(**entry)
        for name, entry in aligned.per_topic_latency.items()
    }
    doc = TimeSyncStat(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        per_topic=per_topic,
    )
    _atomic_write_json(dst_path, doc.model_dump())
    return doc
