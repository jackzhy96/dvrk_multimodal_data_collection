#!/usr/bin/env python3
"""
SurgSync release downloader — pull a published release from the Hugging
Face Hub (default ``jackzhy/surgsync``) onto local disk.

Sibling of ``upload_to_hf.py``; mirrors its staged-gate design. The
gates exist so a user can do cheap (~seconds) smoke checks before
committing to a ~6.5 h, ~670 GB bulk pull:

    smoke    — pull a single tiny file (meta/dataset.json) into the
               target directory. Confirms auth + repo_id + network. ~3 s.
    meta     — pull the entire meta/ subtree (~33 MB). Useful when you
               just want to inspect the dataset (episode counts, task
               vocab, per-column stats) without touching the videos.
    task     — pull one task partition from one of the two splits
               (online_data / offline_data). Specify with
               ``--task <name> --partition <online_data|offline_data>``.
               Meta is bundled so the result is a self-consistent slice.
    episode  — pull one specific episode (path-relative-to-release-root).
               Auto-detects whether it lives under online_data or
               offline_data if you give only the bare ``<task>/<idx>``.
               Verifies the round-trip by checking ``video_raw/
               stereo_left.mkv`` is a real video (>5 MB), not a stray
               LFS pointer.
    full     — the real bulk pull. Resumable via the huggingface_hub
               internal cache; safe to Ctrl-C and re-invoke.
    verify   — re-hash every file in the local target against
               ``meta/manifest.json`` and report any mismatches. Use
               after ``--gate full`` to confirm integrity.

Typical sequence:

    python download_from_hf.py --gate smoke
    python download_from_hf.py --gate meta
    python download_from_hf.py --gate episode --episode online_data/episodes/peg_transfer/96

    # if you only want one task:
    python download_from_hf.py --gate task --task peg_transfer --partition online_data

    # full pull (long-running — run under tmux):
    tmux new -s surgsync-download
    python download_from_hf.py --gate full --max-workers 8
    # Ctrl-B D to detach.
    python download_from_hf.py --gate verify       # after full returns

Prerequisites (one-time):

    pip install --upgrade "huggingface_hub[hf_transfer]>=0.26" "hf-transfer>=0.1.6"
    hf auth login         # write token NOT required for downloads; a read
                          # token is enough. If the repo is public, no auth
                          # at all is needed — the script still works.

This script never mutates the HF repo; it only reads. Re-running any
gate is safe — ``snapshot_download`` is idempotent and resumable.

Configuration:

All volatile runtime settings (target path, repo id, max workers,
fallback partition order, always-include files) live in
``config/surgsync/download_hf.yaml`` — edit values there, not in this
file. CLI flags (e.g. ``--target-dir``, ``--max-workers``) override the
YAML values when supplied; otherwise the YAML defaults apply. Override
the config-file path itself with ``--config <path>``.

TARGET_DIR — where the downloaded copy lands on disk:

The YAML default is intentionally NOT the same path the upload script
reads from. We do NOT want to clobber the canonical packed release on
the SSD by writing the HF-mirrored copy on top of it; that would
defeat the point of an independent verify pass and risks data loss
if HF returns incomplete files. Override with ``--target-dir <path>``.
"""

# Lazy type annotations so ``str | None`` works on Python 3.9 (the
# target env). Same rationale as ``upload_to_hf.py``.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml  # PyYAML — for loading config/surgsync/download_hf.yaml

# Force hf-transfer (Rust accelerator). Same rationale as the uploader —
# the bottleneck on a 670 GB pull is the network plus the Python overhead
# of the chunked HTTP loop. ``setdefault`` lets the user override via env
# if they need pure-Python behavior for debugging.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Import after the env var so ``huggingface_hub`` picks it up at import.
from huggingface_hub import HfApi, snapshot_download  # noqa: E402


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
#
# All volatile knobs (target path, repo id, max workers, fallback
# partitions, always-include files) live in
# `config/surgsync/download_hf.yaml`. This module reads the YAML at
# startup and binds the values into module-level globals that the gate
# functions consume. CLI flags override the YAML values when supplied.
#
# Module-global pattern (instead of passing a config object into every
# gate function): keeps the per-gate code diff-minimal — only the
# binding mechanism changes.

# Default config path — sits next to the repo's other surgsync configs.
# Resolved relative to the toolkit repo root: tests/download_from_hf.py
# → parents[1] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "surgsync" / "download_hf.yaml"

# Module globals — populated by load_config() in main(). Initialised to
# safe sentinel values so import-time inspection (e.g. running --help)
# doesn't trip on a missing config file.
REPO_ID: str = "jackzhy/surgsync"
REPO_TYPE: str = "dataset"
TARGET_DIR: Path = Path("/media/jackzhy/Extreme SSD/surgsync_release_dl")
PARTITIONS: tuple = ("online_data", "offline_data")
ALWAYS_INCLUDE: list = []
DEFAULT_MAX_WORKERS: int = 8


def load_config(path: Path) -> dict:
    """Read the YAML config and return a plain dict.

    Mirrors the uploader's loader. Fails loudly if missing/malformed
    — the script should never silently fall through to baked-in
    defaults that may not match the user's environment.
    """
    if not path.is_file():
        sys.exit(
            f"[config] config file not found: {path}\n"
            f"        Either create it (copy from "
            f"{DEFAULT_CONFIG_PATH}) or pass --config <path>."
        )
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        sys.exit(f"[config] malformed YAML in {path}: {e}")
    if not isinstance(cfg, dict):
        sys.exit(f"[config] top-level of {path} must be a mapping, got {type(cfg).__name__}")
    return cfg


def apply_config(cfg: dict) -> None:
    """Bind YAML values into module globals.

    Required keys: repo_id, repo_type, target_dir, max_workers,
    partitions, always_include. Missing keys raise loudly.
    """
    global REPO_ID, REPO_TYPE, TARGET_DIR
    global PARTITIONS, ALWAYS_INCLUDE, DEFAULT_MAX_WORKERS

    missing = [k for k in (
        "repo_id", "repo_type", "target_dir",
        "max_workers", "partitions", "always_include",
    ) if k not in cfg]
    if missing:
        sys.exit(f"[config] required keys missing in YAML: {missing}")

    REPO_ID              = str(cfg["repo_id"])
    REPO_TYPE            = str(cfg["repo_type"])
    TARGET_DIR           = Path(str(cfg["target_dir"])).resolve()
    DEFAULT_MAX_WORKERS  = int(cfg["max_workers"])
    PARTITIONS           = tuple(cfg["partitions"])
    ALWAYS_INCLUDE       = list(cfg["always_include"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def check_auth_optional(api: HfApi) -> Optional[str]:
    """Resolve current user if logged in; return None if anonymous.

    Unlike the uploader, downloads from a public repo work without
    auth, so we don't abort. For a private repo (jackzhy/surgsync is
    currently private), the underlying snapshot_download call will
    surface a clear 401/403 if the token is missing or read-only.
    """
    try:
        me = api.whoami()
        name = me.get("name") or me.get("fullname") or "?"
        print(f"[auth] logged in as: {name}")
        return name
    except Exception:
        print("[auth] not logged in — proceeding anonymously "
              "(works for public repos; private repos will 401)")
        return None


def _normalize_episode_rel(episode: str) -> str:
    """Accept either a full release-relative path or a short
    ``<task>/<idx>`` form, return the canonical full path.

    The bare form is what most users will type from memory; the full
    form (``online_data/episodes/peg_transfer/96``) is what the rest
    of the codebase uses internally.

    Resolution order for short form:
      1. online_data/episodes/<task>/<idx>
      2. offline_data/episodes/<task>/<idx>
      3. abort
    Resolution checks against ``meta/episodes.parquet`` would be more
    accurate but would force a meta pull just to validate input; we
    instead use a path-existence probe AFTER a meta-only snapshot.
    """
    # If the user already passed a full path with a partition prefix,
    # accept it as-is.
    if episode.startswith(("online_data/", "offline_data/")):
        return episode.rstrip("/")
    # Strip any leading ``episodes/`` they might have typed defensively.
    if episode.startswith("episodes/"):
        episode = episode[len("episodes/"):]
    # Bare form — return the online_data candidate; the caller is
    # responsible for trying offline_data if online doesn't exist.
    return f"online_data/episodes/{episode.strip('/')}"


def _allow_patterns_for_subtree(prefix: str) -> list[str]:
    """Build glob patterns that match every file under ``prefix``.

    huggingface_hub's pattern matcher behaves like git's pathspec —
    ``foo/*`` matches one level, ``foo/**`` matches deeper. We always
    include both to cover Hive-partitioned dirs (``foo/task=*/part-*``)
    and arbitrary nesting under the episode directory tree.
    """
    prefix = prefix.rstrip("/")
    return [f"{prefix}/*", f"{prefix}/**"]


def _print_local_summary(target: Path, patterns: Iterable[str]) -> None:
    """Walk the just-downloaded tree under ``target`` and print a
    one-line summary (n files, total size). Cheap and reassuring.

    Limited to files matching the gate's patterns so we don't double-
    count older content from a prior gate that may also live under
    ``target``.
    """
    n_files = 0
    n_bytes = 0
    pats = list(patterns)
    for p in target.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(target).as_posix()
        if pats and not any(_fnmatch_simple(rel, pat) for pat in pats):
            continue
        n_files += 1
        n_bytes += p.stat().st_size
    print(f"[done] {n_files} files, {n_bytes / 1e9:.2f} GB under {target}")


def _fnmatch_simple(rel: str, pat: str) -> bool:
    """fnmatch with a `**` extension that crosses path separators.

    Standard fnmatch has `*` cross `/`; the huggingface_hub matcher
    distinguishes `*` (one segment) from `**` (any depth). We don't
    need the strict semantics here — only ``_print_local_summary``
    calls this for an approximate count.
    """
    import fnmatch as _fn
    # `**` always matches; collapse it to `*` for fnmatch's purposes.
    return _fn.fnmatch(rel, pat.replace("**", "*"))


# ---------------------------------------------------------------------------
# Gate — smoke
# ---------------------------------------------------------------------------

def gate_smoke() -> None:
    """Pull meta/dataset.json and print its release header.

    Mirrors the uploader's smoke gate: a single small file, end to
    end, just enough to confirm the wiring works. Cheaper than the
    uploader's smoke because nothing is being mutated server-side —
    we expect this to take ~3 s.
    """
    print(f"[smoke] target: {TARGET_DIR}")
    snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        allow_patterns=["meta/dataset.json"],
        local_dir=str(TARGET_DIR),
    )
    p = TARGET_DIR / "meta" / "dataset.json"
    if not p.exists():
        sys.exit(f"[smoke] FAIL — expected {p} after snapshot_download")
    # Parse + print a few key fields so the user can confirm they're
    # looking at the version they expect.
    try:
        info = json.loads(p.read_text())
        print(
            f"[smoke] OK — {info.get('name', '?')} "
            f"data_version={info.get('data_version', '?')} "
            f"schema_version={info.get('schema_version', '?')} "
            f"release_option={info.get('release_option', '?')} "
            f"tasks={len(info.get('tasks', []))}"
        )
    except Exception as e:
        # Don't escalate parse errors here; the file is on disk and
        # readable, which is what the smoke test actually tests.
        print(f"[smoke] OK — file landed (header parse failed: {e})")


# ---------------------------------------------------------------------------
# Gate — meta only
# ---------------------------------------------------------------------------

def gate_meta() -> None:
    """Pull the entire meta/ subtree (~33 MB).

    Useful when you only need the indexes (`episodes.parquet`,
    `index.parquet`, `stats.parquet`, `tasks.jsonl`) to plan a
    selective download or to inspect the dataset without committing
    to the bulk pull. Bundles ``.gitattributes`` so subsequent task /
    episode pulls behave the same as on a fresh clone.
    """
    print(f"[meta] target: {TARGET_DIR}")
    patterns = ["meta/*", "meta/**"] + ALWAYS_INCLUDE
    snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=str(TARGET_DIR),
    )
    if not (TARGET_DIR / "meta" / "manifest.json").exists():
        sys.exit("[meta] FAIL — meta/manifest.json missing after pull")
    _print_local_summary(TARGET_DIR, patterns)
    print("[meta] OK — meta/ is ready. Suggested next step:")
    print(f"       python -c \"import pyarrow.parquet as pq; "
          f"print(pq.read_metadata(r'{TARGET_DIR}/meta/stats.parquet'))\"")


# ---------------------------------------------------------------------------
# Gate — one task partition
# ---------------------------------------------------------------------------

def gate_task(task: str, partition: Optional[str]) -> None:
    """Pull one task partition from one split.

    With ``--partition`` unset, we try online_data first and fall back
    to offline_data. ``cold_cut_dissection_skin_peel`` only exists in
    offline_data in v1.0, so the fallback is load-bearing for that
    task.

    We ALWAYS bundle meta/ so the resulting subtree on disk is a
    valid SurgSync release that the toolkit reader can open without
    a separate meta pull. (``ss.open_dataset(target)`` needs
    ``meta/dataset.json`` + ``meta/episodes.parquet`` to work.)
    """
    candidates = [partition] if partition else list(PARTITIONS)
    # Probe each candidate by listing the repo tree once and checking
    # whether the prefix has any files. We do this with HfApi rather
    # than risking an empty snapshot_download that returns 0 bytes
    # and looks like a success.
    api = HfApi()
    chosen = None
    for cand in candidates:
        prefix = f"{cand}/episodes/{task}"
        try:
            tree = list(api.list_repo_tree(
                repo_id=REPO_ID, repo_type=REPO_TYPE,
                path_in_repo=prefix, recursive=False,
            ))
        except Exception as e:
            # 404 / EntryNotFound means the prefix doesn't exist in
            # this partition — try the next one.
            msg = str(e)
            if "404" in msg or "not found" in msg.lower() or "EntryNotFound" in msg:
                tree = []
            else:
                # Anything else is a real failure (auth, network, ...).
                raise
        if tree:
            chosen = cand
            print(f"[task] resolved {task!r} under partition {cand!r}")
            break

    if chosen is None:
        sys.exit(
            f"[task] FAIL — task {task!r} not found under any of {candidates}. "
            f"Available tasks (from meta/tasks.jsonl after `--gate meta`): see "
            f"`{TARGET_DIR}/meta/tasks.jsonl`."
        )

    prefix = f"{chosen}/episodes/{task}"
    patterns = _allow_patterns_for_subtree(prefix) + ["meta/*", "meta/**"] + ALWAYS_INCLUDE
    print(f"[task] target: {TARGET_DIR}")
    snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=str(TARGET_DIR),
    )
    _print_local_summary(TARGET_DIR, patterns)
    print(f"[task] OK — task {task!r} pulled from {chosen!r}.")


# ---------------------------------------------------------------------------
# Gate — one episode
# ---------------------------------------------------------------------------

def gate_episode(episode: str) -> None:
    """Pull one specific episode and verify the MKV is a real video.

    Accepts either a full path (``online_data/episodes/peg_transfer/96``)
    or a bare ``<task>/<idx>`` form (``peg_transfer/96``). For the
    bare form we try online_data first, then offline_data.

    The post-download size check is the same shape as the uploader's
    canary gate: if ``video_raw/stereo_left.mkv`` lands at <5 MB it's
    a Git LFS pointer file, not the real video. Catches a class of
    silent-failure modes where the HF repo has ``.gitattributes`` but
    the consumer side doesn't have git-lfs configured.
    """
    api = HfApi()
    ep_rel = _normalize_episode_rel(episode)

    # If the user passed a bare path and online_data doesn't have it,
    # retry against offline_data.
    def _exists_in_repo(p: str) -> bool:
        try:
            list(api.list_repo_tree(
                repo_id=REPO_ID, repo_type=REPO_TYPE,
                path_in_repo=p, recursive=False,
            ))
            return True
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower() or "EntryNotFound" in str(e):
                return False
            raise

    if not _exists_in_repo(ep_rel) and ep_rel.startswith("online_data/"):
        alt = ep_rel.replace("online_data/", "offline_data/", 1)
        if _exists_in_repo(alt):
            print(f"[episode] online_data has no {ep_rel}; using {alt}")
            ep_rel = alt
        else:
            sys.exit(f"[episode] FAIL — neither {ep_rel} nor {alt} exists in repo")
    elif not _exists_in_repo(ep_rel):
        sys.exit(f"[episode] FAIL — {ep_rel} does not exist in repo")

    patterns = _allow_patterns_for_subtree(ep_rel) + ALWAYS_INCLUDE
    print(f"[episode] pulling {ep_rel} into {TARGET_DIR}")
    snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=str(TARGET_DIR),
    )

    # Validate: stereo_left.mkv must be a real video, not an LFS pointer
    # file. Same threshold as the uploader's canary check.
    mkv = TARGET_DIR / ep_rel / "video_raw" / "stereo_left.mkv"
    if not mkv.exists():
        sys.exit(
            f"[episode] FAIL — {mkv} missing after pull. allow_patterns may "
            "have been off-by-one. Inspect the target dir manually."
        )
    size_mb = mkv.stat().st_size / (1024 * 1024)
    if size_mb < 5:
        sys.exit(
            f"[episode] FAIL — stereo_left.mkv is only {size_mb:.1f} MB. "
            "That's a Git LFS pointer file, not a real video. Either the "
            "consumer-side git-lfs is missing (install with `apt install "
            "git-lfs` or `brew install git-lfs`) or the source repo doesn't "
            "have LFS routing live — re-check `.gitattributes` on HF."
        )
    print(f"[episode] OK — {ep_rel} ready. stereo_left.mkv = {size_mb:.1f} MB")
    _print_local_summary(TARGET_DIR, patterns)


# ---------------------------------------------------------------------------
# Gate — full pull
# ---------------------------------------------------------------------------

def gate_full(max_workers: int) -> None:
    """Pull the entire release. Resumable, idempotent on re-runs.

    snapshot_download with no allow_patterns / ignore_patterns walks
    the repo tree exhaustively. It uses the huggingface_hub local
    cache for interrupted-download resume: if you Ctrl-C and re-run,
    completed files are skipped and the in-flight one is restarted
    from the partial file in ``~/.cache/huggingface/``.

    ``max_workers`` controls download parallelism inside snapshot_
    download (number of concurrent file fetches). 8 is a reasonable
    default for a ~282 Mbps link; bump to 16 if you have headroom.
    """
    print(f"[full] repo       : {REPO_ID}")
    print(f"[full] target     : {TARGET_DIR}")
    print(f"[full] max_workers: {max_workers}")
    print(f"[full] hf_transfer: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')!r}")
    snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        local_dir=str(TARGET_DIR),
        max_workers=max_workers,
    )
    print("[full] snapshot_download returned cleanly.")
    print(f"[full] verify integrity with: python {Path(__file__).name} --gate verify --target-dir {TARGET_DIR}")


# ---------------------------------------------------------------------------
# Gate — verify
# ---------------------------------------------------------------------------

def gate_verify() -> None:
    """Re-hash every local file against ``meta/manifest.json``.

    This is the post-pull integrity check. ``manifest.json`` carries
    SHA-256 + size for every file in the release; we walk it, hash
    the local copy of each entry, and report any mismatch.

    Caveats — the manifest does NOT cover:
      - ``meta/manifest.json`` itself (chicken-and-egg)
      - ``README.md`` / ``CHANGELOG.md`` (stamped *after* the manifest)
      - ``.logs/*.jsonl`` (operational logs)
      - per-episode ``.surgsync_complete.json`` sentinels
    These are skipped by the manifest builder by design and are not
    a sign of corruption when absent.

    For very large releases (this one is 670 GB), hashing is I/O
    bound — expect ~3-4 hours on an external SSD. If you only want a
    spot-check, pass ``--sample N`` (currently always-on for now;
    not yet wired through the CLI). TODO: add ``--sample`` flag.
    """
    manifest_path = TARGET_DIR / "meta" / "manifest.json"
    if not manifest_path.exists():
        sys.exit(
            f"[verify] FAIL — {manifest_path} missing. Run `--gate meta` "
            "first, or this is a partial pull."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    files: dict = manifest.get("files", {})
    if not files:
        sys.exit("[verify] FAIL — manifest carries no `files` block")

    print(f"[verify] manifest entries: {len(files)}")
    total_bytes = sum(int(e.get("size_bytes", 0)) for e in files.values())
    print(f"[verify] total expected bytes: {total_bytes / 1e9:.2f} GB")

    n_ok = 0
    n_missing = 0
    n_size_mismatch = 0
    n_hash_mismatch = 0
    bytes_hashed = 0
    # Report progress every ~5 GB so a long verify doesn't go silent.
    PROGRESS_EVERY_BYTES = 5_000_000_000

    for rel_path, entry in files.items():
        local = TARGET_DIR / rel_path
        expected_sha = entry.get("sha256")
        expected_size = int(entry.get("size_bytes", -1))

        if not local.exists():
            n_missing += 1
            print(f"[verify] MISSING  {rel_path}")
            continue

        actual_size = local.stat().st_size
        if expected_size >= 0 and actual_size != expected_size:
            n_size_mismatch += 1
            print(
                f"[verify] SIZE     {rel_path}  "
                f"expected={expected_size} actual={actual_size}"
            )
            # Don't bother hashing a size-mismatched file.
            continue

        # SHA-256 in 1 MiB chunks. hashlib's `file_digest` would be
        # cleaner on 3.11+ but we target 3.9; manual streaming is fine.
        h = hashlib.sha256()
        with open(local, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        actual_sha = h.hexdigest()

        if expected_sha and actual_sha != expected_sha:
            n_hash_mismatch += 1
            print(
                f"[verify] HASH     {rel_path}  "
                f"expected={expected_sha[:12]}… actual={actual_sha[:12]}…"
            )
        else:
            n_ok += 1

        bytes_hashed += actual_size
        # Periodic heartbeat.
        if bytes_hashed and bytes_hashed % PROGRESS_EVERY_BYTES < (1024 * 1024):
            pct = 100.0 * bytes_hashed / total_bytes if total_bytes else 0
            print(f"[verify] progress: {bytes_hashed/1e9:.1f} / {total_bytes/1e9:.1f} GB ({pct:.1f}%)")

    print()
    print(f"[verify] OK             : {n_ok}")
    print(f"[verify] missing        : {n_missing}")
    print(f"[verify] size_mismatch  : {n_size_mismatch}")
    print(f"[verify] hash_mismatch  : {n_hash_mismatch}")
    if n_missing or n_size_mismatch or n_hash_mismatch:
        sys.exit(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # ``global`` must be declared BEFORE any use of the name in the
    # function body — Python rejects "used prior to global declaration".
    # We mutate TARGET_DIR below (after CLI parsing) AND read it in the
    # argparse `default=` for --target-dir. Declare it up front.
    global TARGET_DIR

    # Stage 1 — pre-parse just enough to find --config. Same pattern as
    # the uploader: load the YAML first so the full parser's defaults
    # can read from the just-bound module globals.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _remaining = pre.parse_known_args()

    cfg = load_config(Path(pre_args.config))
    apply_config(cfg)

    # Stage 2 — full parser. Defaults read from the just-bound globals.
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help=(
            f"path to the runtime YAML config. Default: {DEFAULT_CONFIG_PATH} "
            "(relative to repo root). Override to swap environments."
        ),
    )
    p.add_argument(
        "--gate",
        required=True,
        choices=["smoke", "meta", "task", "episode", "full", "verify"],
        help=(
            "which gate to run. Typical order: smoke → meta → "
            "(task | episode | full) → verify."
        ),
    )
    p.add_argument(
        "--target-dir", type=str, default=str(TARGET_DIR),
        help=(
            "where the downloaded copy lands on disk. Overrides YAML "
            "`target_dir`. Default points at an SSD location distinct "
            "from the upload script's release root so a corrupted "
            "download cannot shadow the canonical pack."
        ),
    )
    p.add_argument(
        "--task", type=str, default=None,
        help="task name for --gate task (e.g. peg_transfer)",
    )
    p.add_argument(
        "--partition", type=str, default=None, choices=list(PARTITIONS),
        help=(
            "for --gate task: which split to look under. Default tries "
            "online_data first, then offline_data (order from YAML "
            "`partitions`)."
        ),
    )
    p.add_argument(
        "--episode", type=str, default=None,
        help=(
            "episode path for --gate episode. Accepts either a full path "
            "(online_data/episodes/peg_transfer/96) or a bare form "
            "(peg_transfer/96 — online_data assumed)."
        ),
    )
    p.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"parallel downloaders for --gate full (config default: {DEFAULT_MAX_WORKERS}; 4-16 reasonable)",
    )
    args = p.parse_args()

    # Apply CLI override into the module global. TARGET_DIR is the only
    # global we mutate at this stage; PARTITIONS / ALWAYS_INCLUDE / REPO_*
    # come exclusively from YAML. (The ``global`` statement was hoisted
    # to the top of main() so the read above — ``default=str(TARGET_DIR)``
    # — sees the same name as this write.)
    TARGET_DIR = Path(args.target_dir).resolve()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Auth is optional — public repos work anonymously; private repos
    # will surface a clear 401/403 from snapshot_download.
    api = HfApi()
    check_auth_optional(api)

    if args.gate == "smoke":
        gate_smoke()
    elif args.gate == "meta":
        gate_meta()
    elif args.gate == "task":
        if not args.task:
            p.error("--gate task requires --task <name>")
        gate_task(args.task, args.partition)
    elif args.gate == "episode":
        if not args.episode:
            p.error("--gate episode requires --episode <path-or-task/idx>")
        gate_episode(args.episode)
    elif args.gate == "full":
        gate_full(args.max_workers)
    elif args.gate == "verify":
        gate_verify()


if __name__ == "__main__":
    main()
