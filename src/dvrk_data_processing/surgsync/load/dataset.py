"""Dataset-level reader."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from dvrk_data_processing.surgsync.load.episode import Episode, open_episode


log = logging.getLogger(__name__)


# Sentinel the packer stamps when finalization succeeds.
_COMPLETE_SENTINEL = ".surgsync_complete.json"


@dataclass(frozen=True)
class EpisodeRef:
    """Lightweight reference to a discovered episode."""
    dataset_name: str
    task:         str
    clip_index:   str
    path:         Path

    @property
    def key(self) -> str:
        return f"{self.dataset_name}/{self.task}/{self.clip_index}"


class Dataset:
    """Read-only handle on a packed SurgSync release."""

    def __init__(self, root: Path):
        self.root: Path = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")

        meta_path = self.root / "meta" / "dataset.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"meta/dataset.json not found under {self.root}")
        with open(meta_path) as f:
            self._meta: dict = json.load(f)

        self._tasks_jsonl_path = self.root / "meta" / "tasks.jsonl"
        self._task_vocab: Optional[list[dict]] = None
        self._episodes: list[EpisodeRef] = self._discover_episodes()

    # ---- metadata ----------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def schema_version(self) -> str:
        return str(self._meta.get("schema_version", "unknown"))

    @property
    def task_vocab(self) -> list[dict]:
        """Rows from `meta/tasks.jsonl`. Empty list if the file is missing."""
        if self._task_vocab is None:
            if not self._tasks_jsonl_path.is_file():
                self._task_vocab = []
            else:
                rows: list[dict] = []
                with open(self._tasks_jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                self._task_vocab = rows
        return self._task_vocab

    # ---- discovery + filtering --------------------------------------------

    def _discover_episodes(self) -> list[EpisodeRef]:
        """Walk `<root>/<dataset>/episodes/<task>/<clip>/`. Only return
        clips that have the packer's completion sentinel."""
        results: list[EpisodeRef] = []
        for ds_dir in sorted(self.root.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name in {"meta", ".logs"}:
                continue
            episodes_dir = ds_dir / "episodes"
            if not episodes_dir.is_dir():
                continue
            for task_dir in sorted(episodes_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                for clip_dir in sorted(task_dir.iterdir()):
                    if not clip_dir.is_dir():
                        continue
                    if not (clip_dir / _COMPLETE_SENTINEL).is_file():
                        log.debug("skipping incomplete episode dir: %s", clip_dir)
                        continue
                    results.append(EpisodeRef(
                        dataset_name=ds_dir.name,
                        task=task_dir.name,
                        clip_index=clip_dir.name,
                        path=clip_dir,
                    ))
        return results

    @property
    def episodes(self) -> list[EpisodeRef]:
        return list(self._episodes)

    @property
    def tasks(self) -> list[str]:
        return sorted({ep.task for ep in self._episodes})

    @property
    def dataset_names(self) -> list[str]:
        return sorted({ep.dataset_name for ep in self._episodes})

    def filter(
        self,
        *,
        dataset_name: Optional[str] = None,
        task:         Optional[str] = None,
        clip_index:   Optional[str] = None,
        episode_ids:  Optional[set[str]] = None,
        predicate:    Optional[Callable[[EpisodeRef], bool]] = None,
    ) -> list[EpisodeRef]:
        """Return matching episode refs. All filters AND together."""
        out: list[EpisodeRef] = []
        for ep in self._episodes:
            if dataset_name and ep.dataset_name != dataset_name:
                continue
            if task and ep.task != task:
                continue
            if clip_index and ep.clip_index != clip_index:
                continue
            if predicate is not None and not predicate(ep):
                continue
            if episode_ids is not None:
                ep_open = open_episode(ep.path)
                if ep_open.episode_id not in episode_ids:
                    continue
            out.append(ep)
        return out

    def __getitem__(self, key: str) -> Episode:
        """Open an episode by its `<dataset>/<task>/<clip>` key."""
        for ep in self._episodes:
            if ep.key == key:
                return open_episode(ep.path)
        raise KeyError(f"no episode with key {key!r}")

    def __iter__(self) -> Iterator[Episode]:
        for ep in self._episodes:
            yield open_episode(ep.path)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return (
            f"Dataset(root={str(self.root)!r}, "
            f"episodes={len(self._episodes)}, tasks={self.tasks})"
        )


def open_dataset(root: Path) -> Dataset:
    """Open a packed dataset root."""
    return Dataset(Path(root))
