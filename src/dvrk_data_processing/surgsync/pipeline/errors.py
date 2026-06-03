"""Structured error classes for the export pipeline.

- `RecoverableExportError`: bad input for one clip; logged + skipped, the
  rest of the sweep continues.
- `FatalExportError`: configuration / dependency error; aborts the sweep.
"""
from __future__ import annotations


class SurgSyncError(Exception):
    """Root exception for SurgSync pipeline errors."""


class RecoverableExportError(SurgSyncError):
    """One clip failed; sweep continues. Use for missing preprocessing outputs,
    corrupted source data, validation failures, etc."""


class FatalExportError(SurgSyncError):
    """Sweep cannot continue. Use for missing ffmpeg, disk full,
    config conflicts."""


class MissingPreprocessingOutputError(RecoverableExportError):
    """Required preprocessing output is absent for this clip — operator
    should run `scripts/run_all_stages.py` first."""
