"""Three-layer validators (`code_design.md` § 6.4).

- raw_clip: structure check before ingest
- episode:  per-episode schema + decodability check, ideally before atomic rename
- dataset:  end-of-build consistency checks (I-1 .. I-7)

All validators return a list of `ValidationIssue`s; `severity ==
"ERROR"` is fatal. WARNINGs are surfaced but don't fail the run.
"""
from dvrk_data_processing.surgsync.validate.types import ValidationIssue
from dvrk_data_processing.surgsync.validate.raw_clip import validate_raw_clip
from dvrk_data_processing.surgsync.validate.episode import validate_episode
from dvrk_data_processing.surgsync.validate.dataset import validate_dataset

__all__ = [
    "ValidationIssue",
    "validate_raw_clip",
    "validate_episode",
    "validate_dataset",
]
