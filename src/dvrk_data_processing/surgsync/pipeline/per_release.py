"""Per-release orchestrator.

`build_release(cfg)` is the Hydra entry point for `surgsync build`. It:
1. Discovers clips via `ingest.clip.discover_clips` (filtered by the
   `clips` selector in the config).
2. Maps each clip to a task via `cfg.tasks`.
3. Invokes `convert_clip` per clip, logging per-clip outcome.
4. Writes `meta/dataset.json` describing the release.

Indexing (`meta/episodes.parquet`, `index.parquet`, etc.) and
manifest/validation are deferred from the MVP.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig, OmegaConf

from dvrk_data_processing.surgsync.ingest.clip import (
    RawClip, discover_clip, discover_clips,
)
from dvrk_data_processing.surgsync.pipeline.errors import (
    FatalExportError, MissingPreprocessingOutputError, RecoverableExportError,
)
from dvrk_data_processing.surgsync.pipeline.jlogger import JsonlLogger, mint_run_id
from dvrk_data_processing.surgsync.pipeline.per_clip import (
    ConvertedEpisode, convert_clip,
)
from dvrk_data_processing.surgsync.schema import SCHEMA_VERSION, DatasetMeta


log = logging.getLogger(__name__)


def _pack_one_worker(clip: RawClip, task: str, convert_kwargs: dict) -> ConvertedEpisode:
    """Worker entrypoint for ProcessPoolExecutor.

    Lives at module level so it can be pickled by the spawn/fork-based
    multiprocessing implementation. We accept `convert_kwargs` as a
    plain dict (containing nested dicts for align/encode), and rebuild
    DictConfig views here so downstream code (which uses attribute
    access like `align_cfg.tol_ms_kinematic_online`) keeps working.
    """
    kwargs = dict(convert_kwargs)
    kwargs["align_cfg"]  = OmegaConf.create(kwargs["align_cfg"])
    kwargs["encode_cfg"] = OmegaConf.create(kwargs["encode_cfg"])
    return convert_clip(clip, task=task, **kwargs)


def _resolve_clips(cfg: DictConfig) -> list[RawClip]:
    """Pick which clips to pack based on `cfg.clips`."""
    data_dir = Path(cfg.path_config.data_dir)
    source = cfg.clips.source

    if source == "all":
        return discover_clips(data_dir, datasets=list(cfg.path_config.datasets))

    if source == "dataset":
        dataset = cfg.clips.dataset_name or "online_data"
        clips: list[RawClip] = []
        ds_root = data_dir / dataset
        if not ds_root.exists():
            raise FatalExportError(f"clips.dataset_name={dataset!r}: {ds_root} not found")
        for sub in sorted(p for p in ds_root.iterdir() if p.is_dir()):
            try:
                clips.append(discover_clip(data_dir, dataset, sub.name))
            except Exception as e:
                log.warning("skipping %s: %s", sub, e)
        return clips

    if source == "list":
        # Each entry is "<dataset>/<clip_idx>".
        out: list[RawClip] = []
        for entry in cfg.clips.list:
            ds, ci = entry.split("/", 1)
            out.append(discover_clip(data_dir, ds, ci))
        return out

    raise FatalExportError(f"Unknown clips.source: {source!r}")


def _resolve_task(clip: RawClip, cfg: DictConfig) -> str:
    """Look up the task label for one clip via `cfg.tasks`.

    Resolution order:
    1. `cfg.tasks.overrides[<dataset>/<clip_idx>]` — explicit operator
       choice, always wins.
    2. If `cfg.tasks.default_task == "auto"` (or the empty string):
       infer the task by reading the clip's `annotation/phase/*.json`
       and taking the dominant phase. The phase → task map comes from
       `workflow_description.json:_task_routing`.
    3. Otherwise: use `cfg.tasks.default_task` literally.

    Auto-inference is the recommended default — it puts each clip in
    the correct task folder without operator bookkeeping, and the
    workflow-vocab validator only passes when the folder matches.
    """
    from dvrk_data_processing.surgsync.ingest.clip import infer_task_from_phase

    key = f"{clip.dataset_name}/{clip.clip_index}"
    overrides = cfg.tasks.overrides or {}
    if key in overrides:
        return str(overrides[key])

    default = str(cfg.tasks.default_task)
    if default in ("auto", ""):
        inferred = infer_task_from_phase(clip.raw_dir)
        if inferred is None:
            raise FatalExportError(
                f"[{clip.source_clip_str}] tasks.default_task='auto' but the clip's "
                "annotation/phase/ is missing, empty, or maps to an unrouted phase. "
                "Set an explicit task via `tasks.overrides`, e.g. "
                f"`tasks.overrides.\"{key}\"=single_interrupted_stitch`."
            )
        log.info("[%s] auto-detected task: %s", clip.source_clip_str, inferred)
        return inferred
    return default


def _discover_existing_tasks(dataset_root: Path) -> set[str]:
    """Walk the on-disk episode tree and return every task folder name
    that already has at least one completed episode.

    Used to keep `dataset.json.tasks` in sync with reality across
    incremental builds — packing one clip shouldn't shrink the list
    of advertised tasks. Only completed episodes (sentinel-stamped)
    count; partial / failed dirs are ignored.
    """
    found: set[str] = set()
    if not dataset_root.is_dir():
        return found
    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        episodes_root = dataset_dir / "episodes"
        if not episodes_root.is_dir():
            continue
        for task_dir in episodes_root.iterdir():
            if not task_dir.is_dir():
                continue
            # Require at least one completed episode under this task.
            for ep in task_dir.iterdir():
                if ep.is_dir() and (ep / ".surgsync_complete.json").is_file():
                    found.add(task_dir.name)
                    break
    return found


def _write_dataset_meta(
    dataset_root: Path, *, tasks: list[str], release_option: str,
    data_version: str,
) -> None:
    # Union this run's tasks with any tasks already represented by
    # completed episodes on disk — incremental builds must not drop
    # tasks they didn't touch.
    all_tasks = set(tasks) | _discover_existing_tasks(dataset_root)
    dm = DatasetMeta(
        schema_version=SCHEMA_VERSION,
        data_version=data_version,
        release_option=release_option,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=sorted(all_tasks),
    )
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "dataset.json").write_text(dm.model_dump_json(indent=2))


def build_release(cfg: DictConfig) -> dict:
    """Top-level build orchestrator. Returns a summary dict."""
    dataset_root = Path(cfg.path_config.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    run_id = mint_run_id()
    log_dir = Path(cfg.log_dir) if cfg.log_dir else dataset_root / ".logs"
    log_path = log_dir / f"{run_id}.jsonl"
    jlogger = JsonlLogger(log_path)
    log.info("surgsync build run_id=%s log=%s", run_id, log_path)
    jlogger.log(event="start", run_id=run_id, dataset_root=str(dataset_root))

    try:
        clips = _resolve_clips(cfg)
    except Exception as e:
        jlogger.log(event="fatal", error=str(e))
        jlogger.close()
        raise

    log.info("discovered %d clip(s) to pack", len(clips))
    jlogger.log(event="clips_discovered", count=len(clips),
                clips=[c.source_clip_str for c in clips])

    results: list[ConvertedEpisode] = []
    tasks_emitted: list[str] = []
    n_ok = n_skip = n_fail = 0

    # Pre-resolve every clip's task so the worker dispatch loop has no
    # cfg-dependent logic — workers only need primitives.
    clip_tasks = [(clip, _resolve_task(clip, cfg)) for clip in clips]
    tasks_emitted = [t for _, t in clip_tasks]

    parallelism = max(1, int(getattr(cfg, "parallelism", 1)))

    # Shared convert_clip kwargs (all primitives or simple containers
    # so they pickle cleanly for ProcessPoolExecutor workers).
    convert_kwargs = dict(
        dataset_root=dataset_root,
        align_cfg=OmegaConf.to_container(cfg.align, resolve=True),
        encode_cfg=OmegaConf.to_container(cfg.encode, resolve=True),
        fps=float(cfg.fps),
        include_video_processed=bool(cfg.include_video_processed),
        include_video_raw=bool(cfg.include_video_raw),
        include_preprocess=bool(cfg.include_preprocess),
        include_preview=bool(cfg.include_preview),
        force=bool(cfg.force),
        clean_staging=bool(cfg.clean_staging),
    )

    def _handle_outcome(clip, task, exc, out):
        """Update counters + structured log for a single clip outcome."""
        nonlocal n_ok, n_skip, n_fail
        if isinstance(exc, MissingPreprocessingOutputError):
            log.warning("[%s] %s", clip.source_clip_str, exc)
            jlogger.log(event="clip_skipped", clip=clip.source_clip_str,
                        reason="missing_m1", error=str(exc))
            n_skip += 1
            return
        if isinstance(exc, RecoverableExportError):
            log.warning("[%s] recoverable error: %s", clip.source_clip_str, exc)
            jlogger.log(event="clip_failed", clip=clip.source_clip_str, error=str(exc))
            n_fail += 1
            return
        if exc is not None:
            log.error("[%s] unhandled exception: %s", clip.source_clip_str, exc)
            jlogger.log(event="clip_failed", clip=clip.source_clip_str,
                        error=str(exc), unhandled=True)
            n_fail += 1
            return
        results.append(out)
        jlogger.log(event="clip_end", clip=clip.source_clip_str,
                    episode_id=out.episode_id, length_frames=out.length_frames,
                    duration_s=out.duration_s, final_path=str(out.final_path))
        n_ok += 1

    if parallelism == 1:
        # Serial path — keeps the stack trace simple when debugging a
        # single problem clip. Also the test default. Reconstitute the
        # DictConfig views once (same wrap the worker does in parallel).
        serial_kwargs = dict(convert_kwargs)
        serial_kwargs["align_cfg"]  = OmegaConf.create(serial_kwargs["align_cfg"])
        serial_kwargs["encode_cfg"] = OmegaConf.create(serial_kwargs["encode_cfg"])
        for clip, task in clip_tasks:
            jlogger.log(event="clip_start", clip=clip.source_clip_str, task=task)
            try:
                out = convert_clip(clip, task=task, **serial_kwargs)
            except (MissingPreprocessingOutputError, RecoverableExportError) as exc:
                _handle_outcome(clip, task, exc, None)
                continue
            except Exception as exc:
                log.exception("[%s] unhandled exception", clip.source_clip_str)
                _handle_outcome(clip, task, exc, None)
                continue
            _handle_outcome(clip, task, None, out)
    else:
        # Process-pool fan-out. We use processes (not threads) so each
        # worker's ffmpeg subprocesses get their own scheduling slot
        # without any GIL contention on the Python side. With ~3
        # ffmpeg processes per clip (stereo_left/right/side encoded
        # concurrently via stream-level threads inside the worker), a
        # parallelism=N clip pool produces ~3N concurrent encoders;
        # pick N so total ≤ available cores.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        log.info("clip-level parallelism: %d worker(s)", parallelism)
        with ProcessPoolExecutor(max_workers=parallelism) as pool:
            future_to_ct = {}
            for clip, task in clip_tasks:
                jlogger.log(event="clip_start", clip=clip.source_clip_str, task=task)
                fut = pool.submit(_pack_one_worker, clip, task, convert_kwargs)
                future_to_ct[fut] = (clip, task)
            for fut in as_completed(future_to_ct):
                clip, task = future_to_ct[fut]
                try:
                    out = fut.result()
                except (MissingPreprocessingOutputError, RecoverableExportError) as exc:
                    _handle_outcome(clip, task, exc, None)
                except Exception as exc:
                    log.exception("[%s] unhandled exception in worker",
                                  clip.source_clip_str)
                    _handle_outcome(clip, task, exc, None)
                else:
                    _handle_outcome(clip, task, None, out)

    # Dataset-level metadata
    _write_dataset_meta(
        dataset_root,
        tasks=tasks_emitted or [str(cfg.tasks.default_task)],
        release_option=str(cfg.release_option),
        data_version=str(cfg.data_version),
    )

    # Task vocab — auto-generated from workflow_description.json. One
    # row per `_task_routing` entry, each carrying the phase's exact
    # step + gesture vocab. The earlier "copy
    # config/surgsync/tasks.jsonl verbatim" path was retired: a
    # hand-authored vocab inevitably drifts from the workflow JSON,
    # and the parquet `verbalize_*` text already sources from the
    # JSON, so a divergent tasks.jsonl was a footgun for downstream
    # joins.
    from dvrk_data_processing.surgsync.encode.tasks_jsonl import write_tasks_jsonl
    tasks_jsonl_dst = dataset_root / "meta" / "tasks.jsonl"
    try:
        n_tasks = write_tasks_jsonl(tasks_jsonl_dst)
        jlogger.log(event="tasks_jsonl_generated",
                    dst=str(tasks_jsonl_dst), n_rows=n_tasks)
    except Exception as e:
        log.warning("tasks.jsonl generation failed: %s — meta/tasks.jsonl will be absent "
                    "(validator I-6 will WARN, not ERROR)", e)
        jlogger.log(event="tasks_jsonl_failed", error=str(e))

    # NOTE: the consolidated workflow_description.json is treated as a
    # hand-curated input under `workflow_description/`. The packer does
    # not generate or modify it — `serde/workflow_text.py` just reads it
    # for verbalize_* lookups. To update the vocab, edit the JSON
    # directly; no rebuild of MKVs needed.

    # Stages 4–6: build the dataset-wide indexes + manifest after every
    # per-clip pack finishes. Each step is idempotent (full rebuild).
    from dvrk_data_processing.surgsync.index import (
        build_episodes_index, build_frames_index, build_stats, build_manifest,
    )
    try:
        ep_stats = build_episodes_index(dataset_root)
        fr_stats = build_frames_index(dataset_root)
        st_stats = build_stats(dataset_root)
        # Aggregate per-episode modalities.json into meta/modalities.json
        # before the manifest pass so it gets hashed alongside everything else.
        from dvrk_data_processing.surgsync.encode.modalities import (
            write_aggregate_modalities,
        )
        mod_path = write_aggregate_modalities(dataset_root)
        log.info("aggregate modalities → %s", mod_path)
        man_stats = build_manifest(dataset_root, data_version=str(cfg.data_version))
        jlogger.log(event="indexes_built",
                    episodes=ep_stats, frames=fr_stats,
                    stats=st_stats, manifest=man_stats,
                    modalities_aggregate=str(mod_path))
    except Exception as e:
        log.exception("index build failed: %s", e)
        jlogger.log(event="indexes_failed", error=str(e))

    summary = {
        "run_id": run_id,
        "dataset_root": str(dataset_root),
        "n_clips": len(clips),
        "n_ok": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "log_path": str(log_path),
        "results": [
            {"episode_id": r.episode_id, "task": r.task, "path": str(r.final_path),
             "length_frames": r.length_frames}
            for r in results
        ],
    }
    log.info("build done: ok=%d skipped=%d failed=%d  → %s",
             n_ok, n_skip, n_fail, dataset_root)
    jlogger.log(event="finish", **{k: v for k, v in summary.items() if k != "results"})
    jlogger.close()

    return summary
