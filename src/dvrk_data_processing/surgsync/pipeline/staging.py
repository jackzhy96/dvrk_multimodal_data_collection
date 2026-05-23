"""Final-location build with a 3-marker completion protocol.

The earlier design wrote each episode under `<dataset_root>/.staging/<id>/`
then `os.replace`'d it to the final location. That `os.replace` is fast
on ext4 / xfs / btrfs (an inode swap) but degenerates to a multi-GB
physical copy on exFAT and FUSE filesystems — observed at ~25 MB/s on
the SSD's exFAT volume, dominating wall time on every clip.

The new strategy writes outputs **directly into the final dataset
directory** and signals state via three sentinel files at well-known
paths, each flipped into place via atomic single-file rename (fast on
every filesystem):

1. `.surgsync_running.json` — written first, at the moment we open
   the episode dir for writing. Contains run metadata (pid, host,
   start time, episode_id). A scanner that sees this file but no
   complete file knows the pack is either in-flight or crashed.

2. `.surgsync_complete.json` — written last, **only** after all
   encoders + validation succeed. Carries a small completion manifest
   (start + end timestamps, episode_id, length_frames, duration_s).
   This is the canonical "this episode is shippable" signal.
   The running marker is removed once complete is in place.

3. `.surgsync_failed.json` — written if the encoder raises. Carries
   error type + message + start time, so an operator scanning the
   release for failures has a single grep target.

Scanners (validator, index builders, unpack reader) treat
`.surgsync_complete.json` as the **only** source of truth. Anything
else (running, failed, missing) → ignore / report.
"""
from __future__ import annotations
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


log = logging.getLogger(__name__)


# Hidden so they don't clutter file managers; small JSON so the writes
# are essentially free even on slow filesystems.
RUNNING_MARKER  = ".surgsync_running.json"
COMPLETE_MARKER = ".surgsync_complete.json"
FAILED_MARKER   = ".surgsync_failed.json"

# Public alias kept for any callers that imported the old constant.
COMPLETION_SENTINEL = COMPLETE_MARKER


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(dst: Path, payload: dict) -> None:
    """Write JSON via temp-file + os.replace.

    Atomic on every filesystem we care about (including exFAT) — the
    visibility transition "absent → present" is observably instant,
    so a scanner can never read a half-written JSON.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(dst.parent),
        prefix=dst.name + ".", suffix=".tmp", delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())  # durable before rename
        tmp_name = tmp.name
    os.replace(tmp_name, str(dst))


def _remove_silently(path: Path) -> None:
    """Best-effort unlink; ignore FileNotFoundError."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def episode_final_dir(
    dataset_root: Path,
    dataset_name: str,
    task: str,
    clip_index: str,
) -> Path:
    """Resolve the on-disk location for one episode.

    Layout: `<dataset_root>/<dataset_name>/episodes/<task>/<clip_index>/`

    `dataset_name` is the raw top-level folder under `data/` —
    e.g. `offline_data`, `online_data`, or any future name like
    `synthetic_data`. Putting it at the top partitions the release so
    consumers can grab one dataset without touching the others.
    """
    return dataset_root / dataset_name / "episodes" / task / clip_index


def is_episode_complete(episode_dir: Path) -> bool:
    """Did this episode finish a successful pack?

    True iff `.surgsync_complete.json` exists in the directory. The
    running / failed markers don't count — only complete is shippable.
    """
    return (episode_dir / COMPLETE_MARKER).is_file()


def load_completion_manifest(episode_dir: Path) -> Optional[dict]:
    """Return the parsed `.surgsync_complete.json` payload or None.

    Useful for index builders that want to surface duration_s /
    length_frames without re-opening the full episode_meta.json.
    """
    sentinel = episode_dir / COMPLETE_MARKER
    if not sentinel.is_file():
        return None
    try:
        return json.loads(sentinel.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_running_marker(episode_dir: Path, *, episode_id: str) -> None:
    """Stamp `.surgsync_running.json` with run metadata."""
    _atomic_write_json(
        episode_dir / RUNNING_MARKER,
        {
            "schema_version": "1.0.0",
            "kind": "running",
            "episode_id": episode_id,
            "started_at_utc": _utc_iso_now(),
            "pid": os.getpid(),
            "host": platform.node(),
        },
    )


def _write_failed_marker(episode_dir: Path, *, episode_id: str, exc: BaseException) -> None:
    """Stamp `.surgsync_failed.json` with error info; remove running."""
    try:
        running = json.loads((episode_dir / RUNNING_MARKER).read_text())
        started_at = running.get("started_at_utc")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        started_at = None
    _atomic_write_json(
        episode_dir / FAILED_MARKER,
        {
            "schema_version": "1.0.0",
            "kind": "failed",
            "episode_id": episode_id,
            "started_at_utc": started_at,
            "failed_at_utc": _utc_iso_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    )
    _remove_silently(episode_dir / RUNNING_MARKER)


@contextmanager
def episode_staging(
    dataset_root: Path,
    dataset_name: str,
    task: str,
    clip_index: str,
    *,
    episode_id: str,
    clean_existing: bool = False,
) -> Iterator[Path]:
    """Open a final-location episode dir for writing.

    On entry:
    - If the dir exists and is COMPLETE → refuse (caller should have
      short-circuited or wiped with force=True).
    - If the dir exists with running/failed markers (or any stray
      files) → refuse unless `clean_existing=True`, then wipe and
      restart.
    - Make the dir; stamp `.surgsync_running.json`.

    On clean exit: caller is expected to call `finalize_episode` to
    write the complete manifest. The context manager itself only
    handles the failure path.

    On exception: write `.surgsync_failed.json`, leave partial files
    in place for inspection, re-raise.
    """
    final = episode_final_dir(dataset_root, dataset_name, task, clip_index)
    label = f"{dataset_name}/{task}/{clip_index}"

    if final.exists():
        if is_episode_complete(final):
            raise FileExistsError(
                f"final dir exists and is COMPLETE: {final}. "
                "Pass force=true to re-pack."
            )
        if clean_existing:
            log.info("episode %s — wiping incomplete leftover at %s", label, final)
            shutil.rmtree(final)
        else:
            raise FileExistsError(
                f"Incomplete episode dir already exists: {final} "
                "(no .surgsync_complete.json). "
                "Pass clean_staging=true to wipe and retry."
            )

    final.parent.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=False, exist_ok=False)
    _write_running_marker(final, episode_id=episode_id)
    log.info("packing episode %s into %s", label, final)
    try:
        yield final
    except BaseException as exc:
        log.warning(
            "episode %s encountered an error; partial dir preserved at %s "
            "(.surgsync_failed.json written)",
            label, final,
        )
        try:
            _write_failed_marker(final, episode_id=episode_id, exc=exc)
        except Exception:
            log.exception("could not write failure marker at %s", final)
        raise


def finalize_episode(
    episode_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    task: str,
    clip_index: str,
    *,
    episode_id: str,
    length_frames: int,
    duration_s: float,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Stamp the completion manifest and clear the running marker.

    Writes `.surgsync_complete.json` via atomic temp+rename, then
    best-effort removes `.surgsync_running.json`. The complete file is
    the only fact consumers rely on; the running marker is just a
    fast in-flight signal for operators / scanners.

    The `extra` dict, if supplied, is merged into the manifest under
    the same top level — useful for stamping codec versions, source
    clip path, etc., without bloating this function's signature.
    """
    final = episode_final_dir(dataset_root, dataset_name, task, clip_index)
    if not final.is_dir():
        raise FileNotFoundError(f"cannot finalize a missing episode dir: {final}")
    if episode_dir != final:
        raise ValueError(
            f"finalize_episode called with {episode_dir} but final is {final}"
        )

    # Recover the original start time from the running marker (if
    # still around) so the complete manifest carries the full
    # start→complete span.
    try:
        running = json.loads((final / RUNNING_MARKER).read_text())
        started_at = running.get("started_at_utc")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        started_at = None

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "complete",
        "episode_id": episode_id,
        "dataset_name": dataset_name,
        "task": task,
        "clip_index": clip_index,
        "length_frames": int(length_frames),
        "duration_s": float(duration_s),
        "started_at_utc": started_at,
        "completed_at_utc": _utc_iso_now(),
    }
    if extra:
        # Preserve canonical keys above; extras can shadow at the
        # caller's discretion via `extra`.
        payload.update(extra)

    _atomic_write_json(final / COMPLETE_MARKER, payload)
    _remove_silently(final / RUNNING_MARKER)
    # Clear any prior failed marker too — a successful re-pack
    # supersedes the recorded failure.
    _remove_silently(final / FAILED_MARKER)

    log.info("finalized episode %s/%s/%s at %s", dataset_name, task, clip_index, final)
    return final
