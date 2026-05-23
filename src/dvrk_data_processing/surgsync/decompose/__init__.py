"""Inverse of `surgsync build` — write a packed dataset back to the raw +
preprocess on-disk tree."""
from dvrk_data_processing.surgsync.decompose.orchestrator import (
    decompose,
    DecomposeReport,
    DecomposedClipReport,
)

__all__ = ["decompose", "DecomposeReport", "DecomposedClipReport"]
