"""Lazy reader for one packed SurgSync episode."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.load.codec_decode import iter_frames, probe_video


log = logging.getLogger(__name__)


RAW_VIDEO_STREAMS:        tuple[str, ...] = ("stereo_left", "stereo_right", "side")
PROCESSED_VIDEO_STREAMS:  tuple[str, ...] = ("stereo_left", "stereo_right")
PREPROCESS_DENSE_STREAMS: tuple[str, ...] = (
    "depth", "flow_left", "flow_right",
    "heatmap_PSM1_left", "heatmap_PSM1_right",
    "heatmap_PSM2_left", "heatmap_PSM2_right",
)


@dataclass(frozen=True)
class VideoView:
    """Handle for one video file. `iter_frames()` spawns a fresh ffmpeg per call."""
    path: Path
    name: str

    def probe(self) -> dict:
        return probe_video(self.path)

    def iter_frames(self) -> Iterator[np.ndarray]:
        return iter_frames(self.path)


@dataclass
class CalibrationBundle:
    """Files under `<episode>/calibration/`."""
    root:                Path
    camera_index_json:   Optional[Path]
    left_yaml:           Optional[Path]
    right_yaml:          Optional[Path]
    stereo_calib_json:   Optional[Path]
    rectify_params_json: Optional[Path]
    hand_eye_dir:        Optional[Path]


class Episode:
    """One packed clip. Construction reads only `episode_meta.json`;
    parquets and videos load lazily on first access."""

    def __init__(self, path: Path):
        self.path: Path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"episode directory does not exist: {self.path}")

        meta_path = self.path / "episode_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"episode_meta.json missing at {meta_path}")
        with open(meta_path) as f:
            self._meta: dict = json.load(f)

        self._timestamp_table = None
        self._ecm_table = None
        self._psm1_table = None
        self._psm2_table = None
        self._annotation_table = None
        self._modalities: Optional[dict] = None
        self._time_sync_stat: Optional[dict] = None

    # ---- metadata ----------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def episode_id(self) -> str:
        return str(self._meta["episode_id"])

    @property
    def task(self) -> str:
        return str(self._meta["task"])

    @property
    def length(self) -> int:
        return int(self._meta["length_frames"])

    @property
    def master_t0_ns(self) -> int:
        """Absolute ns timestamp of frame 0 (master clock)."""
        return int(self._meta["master_t0_ns"])

    @property
    def recorder_variant(self) -> str:
        return str(self._meta["recorder_variant"])

    @property
    def source_clip(self) -> str:
        return str(self._meta["source_clip"])

    @property
    def source_clip_parts(self) -> tuple[str, str]:
        """`(dataset_name, clip_index)` parsed from `source_clip`."""
        raw = self.source_clip.strip().strip("/")
        parts = [p for p in raw.split("/") if p]
        if parts and parts[0] == "data":
            parts = parts[1:]
        if len(parts) != 2:
            raise ValueError(
                f"source_clip {self.source_clip!r} does not parse to "
                f"(dataset, clip_index); got {parts}"
            )
        return parts[0], parts[1]

    @property
    def modalities(self) -> dict:
        if self._modalities is None:
            p = self.path / "modalities.json"
            self._modalities = json.loads(p.read_text()) if p.is_file() else {}
        return self._modalities

    @property
    def time_sync_stat(self) -> dict:
        if self._time_sync_stat is None:
            p = self.path / "time_sync_stat.json"
            self._time_sync_stat = json.loads(p.read_text()) if p.is_file() else {}
        return self._time_sync_stat

    # ---- parquet tables ----------------------------------------------------

    @property
    def timestamps(self):
        if self._timestamp_table is None:
            self._timestamp_table = pq.read_table(self.path / "timestamp.parquet")
        return self._timestamp_table

    @property
    def ecm(self):
        if self._ecm_table is None:
            self._ecm_table = pq.read_table(self.path / "ECM.parquet")
        return self._ecm_table

    @property
    def psm1(self):
        if self._psm1_table is None:
            self._psm1_table = pq.read_table(self.path / "PSM1.parquet")
        return self._psm1_table

    @property
    def psm2(self):
        if self._psm2_table is None:
            self._psm2_table = pq.read_table(self.path / "PSM2.parquet")
        return self._psm2_table

    @property
    def annotation(self):
        if self._annotation_table is None:
            self._annotation_table = pq.read_table(self.path / "annotation.parquet")
        return self._annotation_table

    def arm(self, name: str):
        if name == "ECM":  return self.ecm
        if name == "PSM1": return self.psm1
        if name == "PSM2": return self.psm2
        raise KeyError(f"unknown arm: {name!r}")

    # ---- videos ------------------------------------------------------------

    def video(self, name: str) -> Optional[VideoView]:
        if name not in PROCESSED_VIDEO_STREAMS:
            log.warning("video(%r): not a canonical processed stream", name)
        p = self.path / "video" / f"{name}.mp4"
        return VideoView(path=p, name=name) if p.is_file() else None

    def video_raw(self, name: str) -> Optional[VideoView]:
        if name not in RAW_VIDEO_STREAMS:
            log.warning("video_raw(%r): not a canonical raw stream", name)
        p = self.path / "video_raw" / f"{name}.mkv"
        return VideoView(path=p, name=name) if p.is_file() else None

    def preprocess(self, name: str) -> Optional[VideoView]:
        if name not in PREPROCESS_DENSE_STREAMS:
            log.warning("preprocess(%r): not a canonical preprocess stream", name)
        p = self.path / "preprocess" / f"{name}.mkv"
        return VideoView(path=p, name=name) if p.is_file() else None

    def available_videos(self) -> dict[str, list[str]]:
        return {
            "video":      [n for n in PROCESSED_VIDEO_STREAMS
                           if (self.path / "video" / f"{n}.mp4").is_file()],
            "video_raw":  [n for n in RAW_VIDEO_STREAMS
                           if (self.path / "video_raw" / f"{n}.mkv").is_file()],
            "preprocess": [n for n in PREPROCESS_DENSE_STREAMS
                           if (self.path / "preprocess" / f"{n}.mkv").is_file()],
        }

    # ---- calibration -------------------------------------------------------

    @property
    def calibration(self) -> CalibrationBundle:
        root = self.path / "calibration"
        if not root.is_dir():
            raise FileNotFoundError(f"calibration/ missing under {self.path}")

        def _opt(rel: str) -> Optional[Path]:
            p = root / rel
            return p if p.exists() else None

        hand_eye = root / "hand_eye"
        return CalibrationBundle(
            root=root,
            camera_index_json=_opt("camera.json"),
            left_yaml=_opt("left.yaml"),
            right_yaml=_opt("right.yaml"),
            stereo_calib_json=_opt("stereo_calib_params.json"),
            rectify_params_json=_opt("rectify_params.json"),
            hand_eye_dir=hand_eye if hand_eye.is_dir() else None,
        )

    # ---- lifecycle ---------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        """Drop cached parquet tables."""
        self._timestamp_table = None
        self._ecm_table = None
        self._psm1_table = None
        self._psm2_table = None
        self._annotation_table = None

    def __enter__(self) -> "Episode":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Episode({self.episode_id!r}, task={self.task!r}, frames={self.length})"


def open_episode(path: Path) -> Episode:
    """Open a packed episode directory."""
    return Episode(Path(path))
