"""Synchronization pipeline.

This subpackage is the **only** place in the codebase that does
timestamp matching. All other modules consume the resulting
`AlignedClip` and never touch `time_syn/` directly.
"""
from dvrk_data_processing.surgsync.align.master_clock import build_master_timeline
from dvrk_data_processing.surgsync.align.matcher import match_modality
from dvrk_data_processing.surgsync.align.policy import TolerancePolicy
from dvrk_data_processing.surgsync.align.contiguity import detect_contiguity
from dvrk_data_processing.surgsync.align.aligner import align_clip, AlignedClip

__all__ = [
    "build_master_timeline",
    "match_modality",
    "TolerancePolicy",
    "detect_contiguity",
    "align_clip",
    "AlignedClip",
]
