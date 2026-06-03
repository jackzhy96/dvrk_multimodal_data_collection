"""Static schemas — Arrow and pydantic — for every contract SurgSync writes.

No I/O lives in this subpackage. Everything here is plain in-memory typed
definitions that the encoder modules consume.

`SCHEMA_VERSION` is the dataset-wide schema version, written into both
`meta/dataset.json` and every `episode_meta.json`. Bumped on a breaking change,
incremented in `MAJOR.MINOR.PATCH` semver style.
"""
from dvrk_data_processing.surgsync.schema.timestamp import build_timestamp_schema
from dvrk_data_processing.surgsync.schema.ecm import build_ecm_schema
from dvrk_data_processing.surgsync.schema.psm import build_psm_schema
from dvrk_data_processing.surgsync.schema.annotation import build_annotation_schema
from dvrk_data_processing.surgsync.schema.episodes import build_episodes_schema
from dvrk_data_processing.surgsync.schema.index import build_index_schema
from dvrk_data_processing.surgsync.schema.stats import build_stats_schema
from dvrk_data_processing.surgsync.schema.episode_meta import (
    EpisodeMeta, SyncStats, PipelineVersions, Tool,
)
from dvrk_data_processing.surgsync.schema.time_sync_stat import (
    TimeSyncStat, PerTopicLatency,
)
from dvrk_data_processing.surgsync.schema.dataset_meta import DatasetMeta
from dvrk_data_processing.surgsync.schema.tasks import TaskVocab
from dvrk_data_processing.surgsync.schema.manifest import Manifest, ManifestFile

# Dataset-wide schema version. Bumped on a schema-breaking change.
# Per spec § 3.1 / § 3.2: every dataset.json and every episode_meta.json carry
# this exact string. Reset to **1.0.0** after the v1.0 release —
# previous in-development bumps were folded into this canonical
# starting point.
SCHEMA_VERSION = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "build_timestamp_schema",
    "build_ecm_schema",
    "build_psm_schema",
    "build_annotation_schema",
    "build_episodes_schema",
    "build_index_schema",
    "build_stats_schema",
    "EpisodeMeta",
    "SyncStats",
    "PipelineVersions",
    "Tool",
    "TimeSyncStat",
    "PerTopicLatency",
    "DatasetMeta",
    "TaskVocab",
    "Manifest",
    "ManifestFile",
]
