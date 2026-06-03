"""SurgSync — packer + reader / decomposer for the dVRK
multimodal dataset.

Public surface:

  Packer:
    - `surgsync build ...` CLI subcommand (no Python-level export).

  Reader:
    - `surgsync.open_dataset(path)`  → `Dataset` lazy reader.
    - `surgsync.open_episode(path)`  → `Episode` lazy reader.

  Inverse / decomposer:
    - `surgsync.decompose(dataset_root, out_root, ...)`
      → write the pre-pack raw + preprocess tree back to disk.
    - `surgsync unpack ...` CLI subcommand wraps the same function.
"""
from dvrk_data_processing.surgsync.decompose import (
    decompose, DecomposeReport, DecomposedClipReport,
)
from dvrk_data_processing.surgsync.load import (
    Dataset, Episode, VideoView, open_dataset, open_episode,
)
from dvrk_data_processing.surgsync.schema import SCHEMA_VERSION


__all__ = [
    "SCHEMA_VERSION",
    "open_dataset", "open_episode",
    "Dataset", "Episode", "VideoView",
    "decompose", "DecomposeReport", "DecomposedClipReport",
]
