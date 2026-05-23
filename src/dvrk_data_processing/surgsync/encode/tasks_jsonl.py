"""Write `<dataset_root>/meta/tasks.jsonl` by projecting
`workflow_description.json` into the per-task vocab schema.

One JSONL row per entry in `_task_routing`. Each row is a strict
`TaskVocab` projection — task name, routed phase id, the phase's
description, and verbatim copies of the phase's `step` and `gesture`
dicts. Consumers can read the JSONL alone; no join back to the
workflow JSON is needed.

This module replaces the earlier "copy `config/surgsync/tasks.jsonl`
verbatim" path. That hand-authored file drifted from the workflow
JSON (e.g. shipped G1–G18 suturing gestures for every task,
including dissection clips that actually use a different vocab).
Auto-generation eliminates that class of drift: the parquet cell
text (from `verbalize_*`), the per-topic stats, and the tasks.jsonl
vocab all come from the same source.
"""
from __future__ import annotations
import json
import logging
import os
import tempfile
from pathlib import Path

from dvrk_data_processing.surgsync.schema import TaskVocab
from dvrk_data_processing.surgsync.serde.workflow_text import task_vocab_rows


log = logging.getLogger(__name__)


def _atomic_write_text(dst: Path, text: str) -> None:
    """Write text via temp+rename so a crash mid-write can never leave
    a half-written tasks.jsonl behind."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(dst.parent),
        prefix=dst.name + ".", suffix=".tmp", delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_name = tmp.name
    os.replace(tmp_name, str(dst))


def write_tasks_jsonl(dst: Path) -> int:
    """Generate and write `meta/tasks.jsonl`. Returns the row count.

    Each row goes through `TaskVocab` pydantic validation, so a typo
    in the workflow JSON's task-routing names or a phase that has no
    description trips here before the file lands on disk.
    """
    rows = task_vocab_rows()
    lines: list[str] = []
    for row in rows:
        # Validate then serialize. `model_dump_json()` preserves key
        # order from the model definition.
        tv = TaskVocab.model_validate(row)
        lines.append(tv.model_dump_json())
    payload = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write_text(dst, payload)
    log.info("wrote tasks.jsonl with %d task row(s) → %s", len(lines), dst)
    return len(lines)
