"""Top-level decompose orchestrator."""
from __future__ import annotations
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from dvrk_data_processing.surgsync.decompose.preprocess import write_preprocess_domain
from dvrk_data_processing.surgsync.decompose.raw import write_raw_domain
from dvrk_data_processing.surgsync.load.dataset import EpisodeRef, open_dataset
from dvrk_data_processing.surgsync.load.episode import Episode, open_episode


log = logging.getLogger(__name__)


_UNPACKED_SENTINEL = ".surgsync_unpacked.json"


@dataclass
class DecomposedClipReport:
    """Per-clip summary."""
    dataset_name:    str
    clip_index:      str
    task:            str
    episode_id:      str
    out_dir:         str
    n_frames:        int
    raw_counts:      dict
    preprocess_counts: dict
    fidelity:        dict[str, str] = field(default_factory=dict)
    elapsed_s:       float = 0.0
    ok:              bool = True
    skipped:         bool = False
    error:           Optional[str] = None


@dataclass
class DecomposeReport:
    """Top-level summary of a decompose run."""
    dataset_root:       str
    out_root:           str
    started_at_utc:     str
    finished_at_utc:    str
    n_episodes_seen:    int
    n_episodes_ok:      int
    n_episodes_fail:    int
    n_episodes_skipped: int = 0
    clips:              list[DecomposedClipReport] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return asdict(self)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2))


# Fidelity tags emitted with each clip report. Same for every clip.
_FIDELITY_DEFAULTS: dict[str, str] = {
    "image":                "bit_exact",
    "kinematic":            "float_equivalent",
    "annotation":           "text_form",
    "time_syn":             "reconstructed",
    "calibration":          "bit_exact",
    "rectify_resize_image": "lossless_within_tol",
    "depth_estimation":     "bit_exact",
    "optical_flow":         "bit_exact",
    "kinematic_reproject":  "bit_exact",
}


def _decompose_one(
    *,
    episode_dir: str,
    out_clip_dir: str,
    streams: tuple[str, ...],
    workers: int,
    force: bool,
) -> dict:
    """Decompose one episode. Returns a flat dict for the parent.

    Skip-or-overwrite logic:
      * sentinel present + force=False → skipped=True (no work).
      * non-empty dir + no sentinel + force=False → refuse with error.
      * force=True → wipe and re-write.
    """
    from time import time as _now
    started = _now()
    ep = open_episode(Path(episode_dir))

    out = Path(out_clip_dir)
    sentinel = out / _UNPACKED_SENTINEL

    if sentinel.is_file() and not force:
        try:
            prior = json.loads(sentinel.read_text())
        except Exception:
            prior = {}
        return {
            "ok": True, "error": None, "skipped": True,
            "n_frames": ep.length, "task": ep.task, "episode_id": ep.episode_id,
            "raw_counts": prior.get("raw_counts", {}),
            "preprocess_counts": prior.get("preprocess_counts", {}),
            "elapsed_s": _now() - started,
        }

    if out.exists() and any(out.iterdir()):
        if not force:
            return {
                "ok": False,
                "error": (f"output dir already populated (no unpack sentinel): {out}; "
                          "pass force=True to overwrite"),
                "skipped": False, "n_frames": ep.length, "task": ep.task,
                "episode_id": ep.episode_id, "raw_counts": {},
                "preprocess_counts": {}, "elapsed_s": _now() - started,
            }
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    raw_counts: dict = {}
    prep_counts: dict = {}
    error: Optional[str] = None
    try:
        if "raw" in streams:
            raw_counts = write_raw_domain(ep, out, workers=workers)
        if "preprocess" in streams:
            prep_counts = write_preprocess_domain(ep, out, workers=workers)
    except Exception as e:
        log.exception("decompose failed for %s", episode_dir)
        error = f"{type(e).__name__}: {e}"
    finally:
        ep.close()

    if error is None:
        payload = {
            "episode_id": ep.episode_id,
            "task": ep.task,
            "n_frames": ep.length,
            "streams": list(streams),
            "raw_counts": raw_counts,
            "preprocess_counts": prep_counts,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(sentinel)

    return {
        "ok": error is None, "error": error, "skipped": False,
        "n_frames": ep.length, "task": ep.task, "episode_id": ep.episode_id,
        "raw_counts": raw_counts, "preprocess_counts": prep_counts,
        "elapsed_s": _now() - started,
    }


def decompose(
    dataset_root: Path,
    out_root: Path,
    *,
    episode_ids:   Optional[Iterable[str]] = None,
    clips:         Optional[Iterable[str]] = None,
    tasks:         Optional[Iterable[str]] = None,
    dataset_names: Optional[Iterable[str]] = None,
    streams:       Sequence[str] = ("raw", "preprocess"),
    force:         bool = False,
    parallelism:   int = 1,
    workers_per_clip: int = 4,
) -> DecomposeReport:
    """Decompose every selected episode under `dataset_root` into `out_root`.

    Filters AND together. `clips` matches `<dataset>/<clip_index>`;
    `episode_ids` matches `episode_meta.json:episode_id`.

    `parallelism > 1` packs clips concurrently via ProcessPoolExecutor.
    Always writes `<out_root>/decompose_report.json`.
    """
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ds = open_dataset(dataset_root)

    eps_filter  = set(episode_ids) if episode_ids else None
    clip_filter = set(clips) if clips else None
    task_filter = set(tasks) if tasks else None
    ds_filter   = set(dataset_names) if dataset_names else None

    candidates: list[EpisodeRef] = []
    for ep_ref in ds.episodes:
        if task_filter and ep_ref.task not in task_filter:
            continue
        if ds_filter and ep_ref.dataset_name not in ds_filter:
            continue
        if clip_filter and f"{ep_ref.dataset_name}/{ep_ref.clip_index}" not in clip_filter:
            continue
        if eps_filter is not None:
            ep_open = open_episode(ep_ref.path)
            try:
                if ep_open.episode_id not in eps_filter:
                    continue
            finally:
                ep_open.close()
        candidates.append(ep_ref)

    log.info("decompose: %d/%d episodes selected (out_root=%s, streams=%s, "
             "force=%s, parallelism=%d)",
             len(candidates), len(ds.episodes), out_root, streams, force, parallelism)

    # Same `(dataset, clip_index)` under two tasks would silently overwrite.
    seen: dict[tuple[str, str], EpisodeRef] = {}
    collisions: list[tuple[EpisodeRef, EpisodeRef]] = []
    for ep_ref in candidates:
        key = (ep_ref.dataset_name, ep_ref.clip_index)
        if key in seen:
            collisions.append((seen[key], ep_ref))
        else:
            seen[key] = ep_ref
    if collisions:
        details = "; ".join(
            f"{a.dataset_name}/{a.clip_index}: tasks={a.task!r} vs {b.task!r}"
            for a, b in collisions
        )
        raise ValueError(
            f"decompose aborted — {len(collisions)} clip(s) appear under multiple "
            f"task folders and would overwrite each other in the unpacked tree: "
            f"{details}. Use --task to disambiguate."
        )

    streams_t = tuple(streams)
    started_utc = datetime.now(timezone.utc).isoformat()
    reports: list[DecomposedClipReport] = []

    def _build_report(ep_ref: EpisodeRef, payload: dict) -> DecomposedClipReport:
        return DecomposedClipReport(
            dataset_name=ep_ref.dataset_name,
            clip_index=ep_ref.clip_index,
            task=payload["task"],
            episode_id=payload["episode_id"],
            out_dir=str(out_root / ep_ref.dataset_name / ep_ref.clip_index),
            n_frames=payload["n_frames"],
            raw_counts=payload["raw_counts"],
            preprocess_counts=payload["preprocess_counts"],
            fidelity=dict(_FIDELITY_DEFAULTS),
            elapsed_s=payload["elapsed_s"],
            ok=payload["ok"],
            skipped=bool(payload.get("skipped", False)),
            error=payload.get("error"),
        )

    def _label(payload: dict) -> str:
        if payload.get("skipped"):
            return "SKIPPED (already unpacked)"
        return "OK" if payload.get("ok") else "FAILED"

    if parallelism > 1:
        with ProcessPoolExecutor(max_workers=parallelism) as pool:
            future_to_ref = {}
            for ep_ref in candidates:
                out_clip = out_root / ep_ref.dataset_name / ep_ref.clip_index
                fut = pool.submit(
                    _decompose_one,
                    episode_dir=str(ep_ref.path),
                    out_clip_dir=str(out_clip),
                    streams=streams_t,
                    workers=workers_per_clip,
                    force=force,
                )
                future_to_ref[fut] = ep_ref
            for fut in as_completed(future_to_ref):
                ep_ref = future_to_ref[fut]
                payload = fut.result()
                reports.append(_build_report(ep_ref, payload))
                log.info("decompose: %s/%s/%s %s in %.1fs",
                         ep_ref.dataset_name, ep_ref.task, ep_ref.clip_index,
                         _label(payload), payload["elapsed_s"])
    else:
        for ep_ref in candidates:
            out_clip = out_root / ep_ref.dataset_name / ep_ref.clip_index
            payload = _decompose_one(
                episode_dir=str(ep_ref.path),
                out_clip_dir=str(out_clip),
                streams=streams_t,
                workers=workers_per_clip,
                force=force,
            )
            reports.append(_build_report(ep_ref, payload))
            log.info("decompose: %s/%s/%s %s in %.1fs",
                     ep_ref.dataset_name, ep_ref.task, ep_ref.clip_index,
                     _label(payload), payload["elapsed_s"])

    finished_utc = datetime.now(timezone.utc).isoformat()
    report = DecomposeReport(
        dataset_root=str(dataset_root),
        out_root=str(out_root),
        started_at_utc=started_utc,
        finished_at_utc=finished_utc,
        n_episodes_seen=len(candidates),
        n_episodes_ok=sum(1 for r in reports if r.ok),
        n_episodes_fail=sum(1 for r in reports if not r.ok),
        n_episodes_skipped=sum(1 for r in reports if r.skipped),
        clips=sorted(reports, key=lambda r: (r.dataset_name, r.clip_index)),
    )
    report.write(out_root / "decompose_report.json")
    log.info("decompose summary: ok=%d fail=%d skipped=%d (of %d selected)",
             report.n_episodes_ok, report.n_episodes_fail,
             report.n_episodes_skipped, report.n_episodes_seen)
    return report
