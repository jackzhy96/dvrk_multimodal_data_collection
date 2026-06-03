"""Dataset-wide indexes built after every per-clip pack completes.

Four indexes:
- `episodes.parquet/task=*/part-*.parquet` — one row per episode
- `index.parquet/task=*/part-*.parquet`   — frame-level cross-episode view
- `stats.parquet`                          — per-column min/max/mean/std
- `manifest.json`                          — SHA256 of every file in the dataset

All builders are full-rebuild in v1 (incremental update is post-v1).
Truth lives in per-episode `episode_meta.json` and `frames.parquet`;
indexes are caches.
"""
from dvrk_data_processing.surgsync.index.episodes_index import build_episodes_index
from dvrk_data_processing.surgsync.index.frames_index import build_frames_index
from dvrk_data_processing.surgsync.index.stats import build_stats
from dvrk_data_processing.surgsync.index.manifest import build_manifest

__all__ = [
    "build_episodes_index",
    "build_frames_index",
    "build_stats",
    "build_manifest",
]
