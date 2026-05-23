"""JSONL structured logger for export runs.

Mirrors `scripts/run_all_stages.py`'s pattern — one JSON object per
line, line-buffered so partial logs survive a hard kill. Stored under
`<dataset_root>/.logs/<run_id>.jsonl`.
"""
from __future__ import annotations
import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, log_path: Path):
        self.path = log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def log(self, **fields: Any) -> None:
        fields.setdefault("ts", time.time())
        self._f.write(json.dumps(fields, default=str) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def mint_run_id() -> str:
    """Short human-readable run id, same shape as run_all_stages."""
    return f"{socket.gethostname()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
