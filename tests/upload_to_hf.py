#!/usr/bin/env python3
"""
SurgSync release uploader — push the packed release sitting in this
directory to the Hugging Face Hub as ``jackzhy/surgsync``.

The full upload is staged through a series of *gates* so we de-risk
the ~6.5 h bulk push with cheap (~10 min total) smoke tests first:

    smoke   — push one tiny known-good file (meta/dataset.json) to
              confirm auth + repo_id + token scope. Failure here means
              the wiring is wrong; the bulk push would 401/404 too.
    card    — push a local markdown file as README.md on the HF repo
              (the "dataset card" — what visitors see on the dataset's
              HF page). Pick which card via --card <name> (selects from
              the YAML's `cards` dict) or --card-path <path> (ad-hoc).
              IMPORTANT: --gate full does NOT push README.md (it's in
              base_ignore because the release tree's README.md is the
              operator-facing one from `surgsync release`, not the
              public dataset card). Push the card explicitly via this
              gate after / instead of --gate full.
    update  — push a hand-picked set of files (or directories, walked
              recursively) in a single atomic commit, overwriting
              existing remote files. Designed for follow-up edits
              after --gate full has already shipped the bulk (e.g.
              pushing four edited episode_meta.json files + the
              regenerated meta/episodes.parquet + meta/manifest.json
              after a case_type fix). Pass `--path <p>` repeatedly,
              or pre-populate the YAML's `update_paths` list. Skips
              the base_ignore filter — so it can push README,
              CHANGELOG, or anything else --gate full deliberately
              excludes. Supports --dry-run.
    update-all — auto-detected counterpart to `update`. Downloads the
              remote meta/manifest.json, compares it to the local
              one, and pushes every file whose SHA-256 differs (or
              that's missing remotely). Fast (~2.5 MB download, no
              local re-hashing — trusts the local manifest's SHAs).
              Refuses to run if the local manifest is stale or the
              remote manifest doesn't exist (latter case: use --gate
              full for the first push). Cannot detect changes to
              non-manifest files (README, CHANGELOG, .gitattributes,
              sentinels) — push those via --gate card / update / lfs.
    lfs     — upload a .gitattributes pointing *.mkv / *.mp4 / *.parquet
              at Git LFS. MUST run BEFORE the first binary, otherwise
              MKVs land as raw git blobs (bad).
    canary  — push one auto-discovered small episode (smallest clip under
              peg_transfer/) end to end, then snapshot_download it back
              and decode a frame. Catches: LFS misrouting, folder layout,
              hf-transfer perf. Episode indices are sparse in the packed
              release (peg_transfer starts at 91 in v1.0), so we
              discover the canary at runtime rather than hardcoding it.
    meta    — push only the meta/ subtree (33 MB). Useful when --gate
              full was run with --no-include-meta and you want to add
              the dataset's metadata + indexes afterwards.
    cleanup — delete the smoke (meta/dataset.json) and canary
              (peg_transfer/<idx>/) uploads from HF, preserving
              .gitattributes. Dry-run by default — pass --confirm to
              actually delete. Optional: --gate full would simply skip
              over those files (same content hash), but running cleanup
              first makes the bulk push look like one atomic commit set.
    full    — the real ~6.5 h job. Uses upload_large_folder which is
              resumable, multi-worker, idempotent on re-runs.
    squash  — collapse every commit on `main` into a single one via
              HfApi.super_squash_history. By the time the staged
              upload finishes, the HF repo will have many commits
              (one per gate, plus N from upload_large_folder's chunking).
              Squash makes the public history read as a single
              "Initial v1.0 release" commit. Dry-run by default —
              pass --confirm to actually rewrite history. ⚠ Destructive:
              prior commit SHAs become unreachable. Run AFTER --gate full
              has returned cleanly and AFTER --gate cleanup (if used).

Typical sequence:

    python upload_to_hf.py --gate smoke
    python upload_to_hf.py --gate lfs
    python upload_to_hf.py --gate canary
    # if all three are green:
    tmux new -s surgsync-upload
    python upload_to_hf.py --gate full --num-workers 8
    # Ctrl-B D to detach; reattach later with: tmux attach -t surgsync-upload
    # After full returns cleanly (next morning):
    python upload_to_hf.py --gate squash               # dry-run — shows current history
    python upload_to_hf.py --gate squash --confirm     # actually collapse into 1 commit

Prerequisites (one-time):

    pip install --upgrade "huggingface_hub[hf_transfer]>=0.26" "hf-transfer>=0.1.6"
    hf auth login         # paste a *write*-scoped token
    hf auth whoami        # must print: jackzhy

Configuration:

All volatile runtime settings (release root path, repo id, num
workers, canary task, ignore patterns, squash commit message) live in
``config/surgsync/upload_hf.yaml`` — edit values there, not in this
file. CLI flags (e.g. ``--release-root``, ``--num-workers``, ``--canary``)
override the YAML values when supplied; otherwise the YAML defaults
apply. Override the config-file path itself with ``--config <path>``.

This script used to live INSIDE the release tree and self-locate via
``Path(__file__).resolve().parent``. It now lives in ``tests/`` of the
toolkit repo (relocated 2026-06-02), so the YAML's ``release_root``
key replaces the old self-location. The script never mutates the
on-disk release tree — re-running any gate is safe.
"""

# Make all type annotations lazy strings (PEP 563), so PEP 604 union
# syntax like `str | None` works on Python 3.9 (the target env). Without
# this, `pick_canary_rel(override: str | None = None)` raises
# `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`
# at import time on <3.10.
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from fnmatch import fnmatch
from pathlib import Path

import yaml  # PyYAML — for loading config/surgsync/upload_hf.yaml

# Force hf-transfer (Rust accelerator). The upload is bandwidth-bound;
# the pure-Python backend tops out at ~35% of the link, vs ~85% for the
# Rust path. setdefault so the user can still override via env if they
# need to debug pure-Python behavior.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Import after the env var so huggingface_hub picks it up on import.
from huggingface_hub import HfApi, snapshot_download  # noqa: E402


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
#
# All volatile knobs (paths, workers, repo id, canary task, ignore
# patterns) live in `config/surgsync/upload_hf.yaml`. This module reads
# the YAML at startup and binds the values into module-level globals
# that the gate functions consume. CLI flags override the YAML values
# when supplied.
#
# Module-global pattern (instead of passing a config object into every
# gate function): the gate functions were originally written to read
# self-located constants. Keeping them as globals after the move keeps
# the per-gate code diff-minimal — only the binding mechanism changes.
#
# GITATTRIBUTES stays as a constant in this script (it's not config —
# it's a fixed HF Hub template + custom additions that should not vary
# per environment).

# Default config path — sits next to the repo's other surgsync configs.
# Resolved relative to the toolkit repo root: tests/upload_to_hf.py →
# parents[1] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "surgsync" / "upload_hf.yaml"

# Module globals — populated by load_config() in main(). Initialised to
# safe sentinel values so import-time inspection (e.g. running --help)
# doesn't trip on a missing config file.
REPO_ID: str = "jackzhy/surgsync"
REPO_TYPE: str = "dataset"
RELEASE_ROOT: Path = Path("<release_root>")
CANARY_TASK: str = "online_data/episodes/peg_transfer"
CANARY_DOWNLOAD_DIR: str = "tmp/canary_download"
BASE_IGNORE: list = []
DEFAULT_NUM_WORKERS: int = 8
DEFAULT_INCLUDE_META: bool = True
DEFAULT_SQUASH_MESSAGE: str = ""
DEFAULT_CANARY_OVERRIDE: "str | None" = None
# --gate card: dict of named card files + which one is the default.
# Keys are short names (e.g. "dataset_card"), values are paths relative
# to release_root or absolute. The user picks via --card <name>; the
# default fires when no --card / --card-path is given.
DEFAULT_CARD_NAME: str = "dataset_card"
CARDS: dict = {}
# --gate update: bundle a hand-picked set of files into a single
# atomic commit on HF. The YAML carries the default file list and
# commit message; both are overridable from the CLI.
DEFAULT_UPDATE_MESSAGE: str = "Apply targeted file updates"
DEFAULT_UPDATE_PATHS: list = []
# --gate update-all: auto-detected counterpart. Default commit message.
DEFAULT_UPDATE_ALL_MESSAGE: str = "Apply targeted file updates (auto-detected)"


def load_config(path: Path) -> dict:
    """Read the YAML config and return a plain dict.

    Fails loudly if the file is missing or malformed — these scripts
    should never silently fall through to baked-in defaults that may
    not match the user's environment.
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

    Validates required keys are present; missing keys raise rather than
    silently degrade. The CLI override pass in main() can subsequently
    overwrite any of these.
    """
    global REPO_ID, REPO_TYPE, RELEASE_ROOT
    global CANARY_TASK, CANARY_DOWNLOAD_DIR, BASE_IGNORE
    global DEFAULT_NUM_WORKERS, DEFAULT_INCLUDE_META
    global DEFAULT_SQUASH_MESSAGE, DEFAULT_CANARY_OVERRIDE
    global DEFAULT_CARD_NAME, CARDS
    global DEFAULT_UPDATE_MESSAGE, DEFAULT_UPDATE_PATHS
    global DEFAULT_UPDATE_ALL_MESSAGE

    missing = [k for k in (
        "repo_id", "repo_type", "release_root",
        "canary_task", "canary_download_dir", "base_ignore",
        "num_workers", "include_meta", "squash_message",
        "card_default", "cards",
        "update_message", "update_paths",
        "update_all_message",
    ) if k not in cfg]
    if missing:
        sys.exit(f"[config] required keys missing in YAML: {missing}")

    REPO_ID                  = str(cfg["repo_id"])
    REPO_TYPE                = str(cfg["repo_type"])
    RELEASE_ROOT             = Path(str(cfg["release_root"])).resolve()
    CANARY_TASK              = str(cfg["canary_task"])
    CANARY_DOWNLOAD_DIR      = str(cfg["canary_download_dir"])
    BASE_IGNORE              = list(cfg["base_ignore"])
    DEFAULT_NUM_WORKERS      = int(cfg["num_workers"])
    DEFAULT_INCLUDE_META     = bool(cfg["include_meta"])
    DEFAULT_SQUASH_MESSAGE   = str(cfg["squash_message"]).strip()
    DEFAULT_CANARY_OVERRIDE  = cfg.get("canary_override")  # may be None
    DEFAULT_CARD_NAME        = str(cfg["card_default"])
    # cards is a mapping str -> str; coerce defensively in case YAML
    # parsed the keys as ints (e.g. if someone names a card "1").
    CARDS                    = {str(k): str(v) for k, v in dict(cfg["cards"]).items()}
    if DEFAULT_CARD_NAME not in CARDS:
        sys.exit(
            f"[config] card_default={DEFAULT_CARD_NAME!r} not present in cards "
            f"dict (available: {sorted(CARDS)})"
        )
    DEFAULT_UPDATE_MESSAGE   = str(cfg["update_message"]).strip()
    DEFAULT_UPDATE_PATHS     = [str(p) for p in (cfg["update_paths"] or [])]
    DEFAULT_UPDATE_ALL_MESSAGE = str(cfg["update_all_message"]).strip()

# What --gate lfs uploads as .gitattributes. Without this, large MKVs land
# as raw git blobs and the repo becomes unclonable. Order matters for
# readability; HF respects standard gitattributes syntax.
#
# This is the standard Hugging Face Hub .gitattributes template (the same
# one HF auto-generates when you create a dataset repo in the web UI),
# extended with `*.mkv` which is what surgsync stores raw stereo/side
# video in. `*.mp4` and `*.parquet` are already covered by the HF defaults
# so we don't need to repeat them.
GITATTRIBUTES = """\
*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.avro filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.lz4 filter=lfs diff=lfs merge=lfs -text
*.mds filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
# Audio files - uncompressed
*.pcm filter=lfs diff=lfs merge=lfs -text
*.sam filter=lfs diff=lfs merge=lfs -text
*.raw filter=lfs diff=lfs merge=lfs -text
# Audio files - compressed
*.aac filter=lfs diff=lfs merge=lfs -text
*.flac filter=lfs diff=lfs merge=lfs -text
*.mp3 filter=lfs diff=lfs merge=lfs -text
*.ogg filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
# Image files - uncompressed
*.bmp filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.tiff filter=lfs diff=lfs merge=lfs -text
# Image files - compressed
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.webp filter=lfs diff=lfs merge=lfs -text
# Video files - compressed
*.mkv filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.webm filter=lfs diff=lfs merge=lfs -text
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def check_auth(api: HfApi) -> str:
    """Resolve current user; fail loudly if not logged in.

    Returns the resolved username so callers can sanity-check against the
    expected REPO_ID owner.
    """
    try:
        me = api.whoami()
    except Exception as e:
        sys.exit(
            f"[auth] FAILED to resolve user. Run `hf auth login` with a "
            f"write-scoped token first. Underlying error: {e}"
        )
    name = me.get("name") or me.get("fullname") or "?"
    print(f"[auth] logged in as: {name}")
    # Soft warning if the namespace doesn't match the repo owner — caller
    # may legitimately be an org collaborator, so we don't abort.
    expected_owner = REPO_ID.split("/", 1)[0]
    if name != expected_owner:
        print(
            f"[auth] WARN: logged-in user `{name}` differs from repo owner "
            f"`{expected_owner}` — confirm you have write access."
        )
    return name


def pick_canary_rel(override: str | None = None) -> str:
    """Pick which episode to use as the --gate canary.

    Preference order:
      1. CLI override (--canary <rel>), if it points at a real episode dir.
      2. Smallest clip under CANARY_TASK (peg_transfer) — fastest to push,
         most representative of a "small clip" smoke test.
      3. Smallest clip under any task in online_data/episodes/.
      4. Abort.

    "Smallest" is measured by video_raw/stereo_left.mkv size; the canary
    must have a stereo_left.mkv for the LFS-routing check to work.
    """
    if override:
        # Normalize abs paths to relative — `--canary /media/.../foo`
        # would otherwise survive `RELEASE_ROOT / override` (Path
        # semantics: dividing by an absolute path REPLACES the LHS) and
        # then land at the wrong path_in_repo on HF. Convert to relative
        # form so the rest of the function is path-form-agnostic.
        p = Path(override)
        if p.is_absolute():
            try:
                override = str(p.resolve().relative_to(RELEASE_ROOT.resolve()))
            except ValueError:
                sys.exit(
                    f"[canary] --canary {override!r} is an absolute path "
                    f"outside the release root {RELEASE_ROOT}"
                )
        cand = RELEASE_ROOT / override
        if not (cand / "video_raw" / "stereo_left.mkv").exists():
            sys.exit(
                f"[canary] --canary {override!r} does not look like a packed "
                f"episode (no video_raw/stereo_left.mkv at {cand})"
            )
        return override

    def _candidates(base: Path):
        """Yield (mkv_size, episode_rel_path) for every packable episode."""
        if not base.is_dir():
            return
        for ep_dir in sorted(base.iterdir()):
            if not ep_dir.is_dir():
                continue
            mkv = ep_dir / "video_raw" / "stereo_left.mkv"
            if not mkv.exists():
                continue
            yield mkv.stat().st_size, ep_dir.relative_to(RELEASE_ROOT).as_posix()

    # Try preferred task first.
    preferred = sorted(_candidates(RELEASE_ROOT / CANARY_TASK))
    if preferred:
        size, rel = preferred[0]
        print(
            f"[canary] auto-picked {rel} (stereo_left.mkv = {size/1e6:.1f} MB, "
            f"smallest under {CANARY_TASK})"
        )
        return rel

    # Fall back to ANY task.
    print(
        f"[canary] {CANARY_TASK} has no episodes; falling back to smallest "
        "clip across all online_data tasks"
    )
    all_tasks = RELEASE_ROOT / "online_data" / "episodes"
    if not all_tasks.is_dir():
        sys.exit(f"[canary] no online_data/episodes/ — release looks broken: {all_tasks}")
    all_candidates = []
    for task_dir in sorted(all_tasks.iterdir()):
        all_candidates.extend(_candidates(task_dir))
    if not all_candidates:
        sys.exit("[canary] no episodes with stereo_left.mkv found anywhere")
    size, rel = sorted(all_candidates)[0]
    print(f"[canary] auto-picked {rel} (stereo_left.mkv = {size/1e6:.1f} MB)")
    return rel


def ensure_repo(api: HfApi) -> None:
    """Idempotently confirm the (already-created) repo exists.

    `exist_ok=True` makes this a no-op when the repo is already created
    — which it is, since we created it manually on HF first. Including it
    means the script also works on a fresh slate without manual setup.
    """
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------------
# Gate 2 — smoke test
# ---------------------------------------------------------------------------

def gate_smoke(api: HfApi) -> None:
    """Push one tiny file to confirm auth + repo + scope are wired right."""
    src = RELEASE_ROOT / "meta" / "dataset.json"
    if not src.exists():
        sys.exit(f"[smoke] expected file missing: {src}")
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo="meta/dataset.json",
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        commit_message="Smoke test: meta/dataset.json",
    )
    print(
        "[smoke] OK — uploaded meta/dataset.json.\n"
        f"        Verify in browser: https://huggingface.co/datasets/{REPO_ID}/tree/main/meta"
    )


# ---------------------------------------------------------------------------
# Gate 3 — LFS routing
# ---------------------------------------------------------------------------

def gate_lfs(api: HfApi) -> None:
    """Push .gitattributes so binary types are routed through LFS."""
    # Write to a tempfile rather than poisoning the release tree.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".gitattributes", delete=False
    ) as f:
        f.write(GITATTRIBUTES)
        tmp_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=".gitattributes",
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            commit_message="Track binary file types (video / parquet / archives / images / etc.) via LFS",
        )
    finally:
        # Clean up the tempfile regardless of upload outcome.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    print("[lfs] OK — uploaded .gitattributes (LFS routing live).")


# ---------------------------------------------------------------------------
# Gate 4 — canary episode
# ---------------------------------------------------------------------------

def gate_canary(api: HfApi, canary_rel: str) -> None:
    """Push one small episode + round-trip-decode a frame to validate LFS.

    This is the gate that catches the most expensive failure mode: a
    misconfigured .gitattributes that stores MKVs as raw git blobs. We
    detect it by re-downloading the file and asserting its size is
    realistic — an LFS pointer file is ~130 bytes, a real MKV is 100+ MB.

    `canary_rel` is the auto-picked (or --canary-overridden) episode
    path, relative to RELEASE_ROOT, e.g. "online_data/episodes/peg_transfer/96".
    """
    src = RELEASE_ROOT / canary_rel
    if not src.exists():
        sys.exit(f"[canary] expected episode missing: {src}")

    print(f"[canary] uploading {canary_rel} ...")
    # IMPORTANT: pass BASE_IGNORE here. `api.upload_folder` does NOT inherit
    # the script's exclusion list; without this, sentinels like
    # `.surgsync_complete.json` and any of the other BASE_IGNORE patterns
    # rooted under the episode dir get uploaded along with the real data.
    # Note: ignore_patterns in upload_folder match against paths *relative
    # to folder_path*, not the repo root — so the top-level
    # ".surgsync_*.json" pattern correctly catches the sentinel sitting at
    # `<episode>/.surgsync_complete.json`.
    api.upload_folder(
        folder_path=str(src),
        path_in_repo=canary_rel,
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        commit_message=f"Canary: {canary_rel}",
        ignore_patterns=BASE_IGNORE,
    )
    # Pull the episode back down into tmp/canary_download/ for verification.
    # Explicit local_dir keeps the round-trip on the same SSD as the source,
    # off the home-dir HF cache, and confined to a single directory the
    # user can `rm -rf` between runs. The source tree is never touched —
    # snapshot_download creates a separate subdirectory.
    download_root = RELEASE_ROOT / CANARY_DOWNLOAD_DIR
    download_root.mkdir(parents=True, exist_ok=True)
    print(f"[canary] verifying round-trip — downloading into {download_root} ...")

    root = snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        allow_patterns=[
            f"{canary_rel}/*",
            f"{canary_rel}/**",
            ".gitattributes",
        ],
        local_dir=str(download_root),
    )
    # `root` will equal `str(download_root)` when local_dir is set. We
    # rebuild the episode path from that base so the rest is uniform.
    ep_dir = Path(root) / canary_rel
    mkv = ep_dir / "video_raw" / "stereo_left.mkv"
    if not mkv.exists():
        sys.exit(
            f"[canary] FAIL — {mkv} missing from snapshot.\n"
            f"        Expected it under {download_root} after snapshot_download. "
            "Most likely cause: allow_patterns glob didn't match the episode "
            "files. Inspect the download root manually."
        )
    # stat() follows symlinks — modern huggingface_hub may materialize files
    # as symlinks back into the global cache. Either form reports the true
    # byte size, which is what we care about for LFS-pointer detection.
    size_mb = mkv.stat().st_size / (1024 * 1024)
    if size_mb < 5:
        sys.exit(
            f"[canary] FAIL — stereo_left.mkv is only {size_mb:.1f} MB. "
            "That's a Git LFS pointer file, not the real video. "
            "The .gitattributes from `--gate lfs` either didn't land or "
            "didn't run BEFORE this canary upload. Fix: delete the canary "
            "episode from HF (web UI), re-run `--gate lfs`, then `rm -rf "
            f"{download_root}` and re-run this gate."
        )
    print(f"[canary] size OK — stereo_left.mkv = {size_mb:.1f} MB")

    # Deeper validation: open with the surgsync reader and decode a frame.
    # Optional because the env may not have dvrk_data_processing importable
    # everywhere (e.g. a clean conda env without `pip install -e .`).
    try:
        import dvrk_data_processing.surgsync as ss
    except ImportError:
        print(
            "[canary] (skipped reader check — dvrk_data_processing not "
            "importable. Run `pip install -e .` in the toolkit repo to enable.)"
        )
        return

    ep = ss.open_episode(str(ep_dir))
    try:
        # IMPORTANT: do NOT do `frame = next(view.iter_frames())` here.
        # surgsync's decode_video_frames spawns ffmpeg with stdout=PIPE
        # and only cleans up via `proc.wait()` in a finally block. If we
        # take one frame and stop, ffmpeg keeps producing frames, fills
        # the 64 KB PIPE buffer, and blocks on write() — meanwhile our
        # generator's finally block calls proc.wait() and blocks on
        # ffmpeg, classic deadlock. The script then never exits.
        #
        # Workaround: consume the *entire* generator so ffmpeg hits EOF
        # cleanly. We snapshot the first frame for validation and count
        # the rest; memory footprint stays small (we don't accumulate
        # frames, we just iterate past them).
        #
        # Real fix lives in src/dvrk_data_processing/surgsync/encode/codec.py
        # — its `finally:` should `proc.kill()` before `proc.wait()` so
        # abandoned generators don't hang the caller.
        view = ep.video_raw("stereo_left")
        first_frame = None
        n_frames = 0
        for frame in view.iter_frames():
            if first_frame is None:
                first_frame = frame
            n_frames += 1
        assert first_frame is not None, "video_raw produced zero frames"
        assert first_frame.ndim == 3 and first_frame.shape[2] == 3, first_frame.shape
        print(
            f"[canary] reader OK — decoded {n_frames} frames, "
            f"first frame {first_frame.shape} {first_frame.dtype} "
            f"(episode_id={ep.episode_id}, length={ep.length})"
        )
    finally:
        ep.close()

    # Tell the user where to inspect the round-trip. They can `rm -rf`
    # this between runs to force a fresh download; otherwise it accumulates.
    print(f"[canary] round-trip files at: {download_root}/{canary_rel}")


# ---------------------------------------------------------------------------
# Gate 5 — meta only
# ---------------------------------------------------------------------------

def gate_meta(api: HfApi) -> None:
    """Push only the meta/ subtree (~33 MB).

    Useful when a previous --gate full was run with --no-include-meta
    (e.g. meta/ was still being regenerated) and you want to add the
    dataset.json / episodes.parquet / index.parquet / stats.parquet now.
    """
    src = RELEASE_ROOT / "meta"
    if not src.exists():
        sys.exit(f"[meta] missing: {src}")
    # Defensive: meta/ doesn't currently contain any BASE_IGNORE matches,
    # but mirror the canary's handling so the two upload_folder call sites
    # stay symmetrical and future-proof against meta/ growing surprise
    # sentinels.
    api.upload_folder(
        folder_path=str(src),
        path_in_repo="meta",
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        commit_message="Add meta/ (dataset.json + Hive-partitioned indexes + stats)",
        ignore_patterns=BASE_IGNORE,
    )
    print("[meta] OK — meta/ uploaded.")


# ---------------------------------------------------------------------------
# Gate 5b — cleanup smoke + canary uploads
# ---------------------------------------------------------------------------

def gate_cleanup(api: HfApi, canary_rel: str, confirm: bool) -> None:
    """Remove the smoke (meta/dataset.json) and canary (one episode) uploads.

    Why ever do this:
      - The smoke and canary files would also be re-uploaded by --gate
        full as part of the regular release. upload_large_folder dedupes
        by content hash, so they wouldn't actually re-transfer — but
        they'd already exist as separate commits in the HF git log,
        scattered across earlier timestamps. Running cleanup first makes
        the bulk push read as one atomic commit set in the HF UI.

    What we DON'T touch:
      - .gitattributes — load-bearing for LFS routing in --gate full.
        Removing it before the bulk push would store every MKV as a raw
        git blob (bad).
      - meta/ as a whole — only `meta/dataset.json` since that's all the
        smoke test put there.

    Safety:
      - Defaults to dry-run. Prints what *would* be deleted.
      - Pass --confirm to actually delete. Each deletion creates a real
        commit in the HF git history; nothing is truly destroyed (you
        can rewind via the revisions tab) but it's still a one-way trip
        for casual users.
    """
    # Folder deletion in huggingface_hub uses delete_folder; file
    # deletion uses delete_file. We treat both uniformly with a tag.
    targets = [
        ("file",   "meta/dataset.json",
         "smoke-test upload (Gate 2)"),
        ("folder", canary_rel,
         "canary upload (Gate 4)"),
    ]

    print("[cleanup] would delete from HF:")
    for kind, path, note in targets:
        print(f"  - {kind:6s} {path}   ({note})")
    print("[cleanup] LEAVE in place: .gitattributes  (needed for LFS in --gate full)")

    if not confirm:
        print("[cleanup] DRY-RUN — re-run with `--confirm` to actually delete.")
        return

    # Real deletion. Each call is a separate commit. We swallow 404 /
    # EntryNotFoundError so re-running cleanup after partial success is
    # safe (no point hard-failing because the smoke file already went
    # away last time).
    for kind, path, _note in targets:
        try:
            if kind == "file":
                api.delete_file(
                    path_in_repo=path,
                    repo_id=REPO_ID, repo_type=REPO_TYPE,
                    commit_message=f"Cleanup: drop {path} (pre-bulk-upload tidy)",
                )
            else:
                api.delete_folder(
                    path_in_repo=path,
                    repo_id=REPO_ID, repo_type=REPO_TYPE,
                    commit_message=f"Cleanup: drop {path}/ (pre-bulk-upload tidy)",
                )
            print(f"[cleanup] deleted {kind}: {path}")
        except Exception as e:
            # 404 is fine — already gone. Anything else is worth showing.
            msg = str(e)
            if "404" in msg or "not found" in msg.lower() or "EntryNotFound" in msg:
                print(f"[cleanup] (already absent) {kind}: {path}")
            else:
                print(f"[cleanup] WARN — could not delete {kind} {path}: {e}")


# ---------------------------------------------------------------------------
# Gate 5c — dataset card (README.md on HF)
# ---------------------------------------------------------------------------

def _resolve_card_path(card_name: str | None, card_path: str | None) -> Path:
    """Resolve which markdown file to push as README.md.

    Precedence:
      1. --card-path <p>     : ad-hoc explicit path; absolute, or relative
                                to release_root.
      2. --card <name>       : look up in CARDS dict.
      3. CARDS[DEFAULT_CARD_NAME] (the YAML's `card_default`).

    Relative paths resolve against RELEASE_ROOT so users can pass the
    same kind of in-tree path the YAML uses.
    """
    if card_path:
        p = Path(card_path)
        return p if p.is_absolute() else (RELEASE_ROOT / p).resolve()

    name = card_name or DEFAULT_CARD_NAME
    if name not in CARDS:
        sys.exit(
            f"[card] unknown --card {name!r}. Available: {sorted(CARDS)}. "
            "Add an entry to upload_hf.yaml:cards, or pass --card-path."
        )
    p = Path(CARDS[name])
    return p if p.is_absolute() else (RELEASE_ROOT / p).resolve()


def gate_card(api: HfApi, card_name: str | None, card_path: str | None) -> None:
    """Push a local markdown file as README.md on the HF dataset repo.

    The HF dataset card IS the repo's README.md — Hugging Face renders
    it as the dataset page and parses the YAML frontmatter for badges,
    tags, license, and the dataset-viewer's `configs:` wiring.

    --gate full deliberately does NOT push README.md (it's in
    base_ignore), so this gate is the only path to update the card.

    The body of the upload commit message names the source file so the
    HF git log makes it easy to tell which iteration of the card
    landed in any given commit.
    """
    src = _resolve_card_path(card_name, card_path)
    if not src.is_file():
        sys.exit(
            f"[card] FAIL — {src} does not exist or is not a file. "
            "Pick a different --card or fix the path in upload_hf.yaml:cards."
        )
    # Show what we're about to push so the operator can confirm visually
    # before the API call goes out — the card includes YAML frontmatter
    # that materially affects how HF renders the page, so an accidental
    # push of the operator-facing release_readme over the dataset_card
    # would be visible immediately.
    size_kb = src.stat().st_size / 1024
    print(f"[card] uploading: {src}  ({size_kb:.1f} KB)")
    print(f"[card] target   : {REPO_ID}:README.md  (replaces existing if any)")

    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo="README.md",
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        commit_message="update README.md and dataset card",
    )
    print(
        f"[card] OK — pushed.\n"
        f"        Inspect at: https://huggingface.co/datasets/{REPO_ID}\n"
        f"        Edit history: https://huggingface.co/datasets/{REPO_ID}/commits/main"
    )


# ---------------------------------------------------------------------------
# Gate 5d — targeted file update
# ---------------------------------------------------------------------------

def _expand_update_paths(paths: list[str]) -> list[tuple[Path, str]]:
    """Expand a list of file-or-directory paths into a flat list of
    (local-file-path, path-in-repo) tuples.

    Rules:
      - Absolute paths are taken as-is on the local side.
      - Relative paths resolve against RELEASE_ROOT.
      - Directories are walked recursively (sorted for deterministic
        commit order).
      - path-in-repo is derived as the path relative to RELEASE_ROOT.
        If the local path is outside RELEASE_ROOT, the gate aborts —
        we don't guess a destination for ad-hoc external files; pass
        them via --gate card or --path-pair (not yet implemented).

    Hidden files (`.surgsync_*.json` sentinels, dotfiles in general)
    and `__pycache__` directories are NOT filtered — the user is
    explicitly listing what to push, and being prescriptive about
    filtering could surprise them. If you want a sentinel pushed,
    you can; the default `base_ignore` from --gate full doesn't apply.
    """
    expanded: list[tuple[Path, str]] = []
    for raw in paths:
        local = Path(raw)
        if not local.is_absolute():
            local = (RELEASE_ROOT / raw).resolve()
        else:
            local = local.resolve()

        if not local.exists():
            sys.exit(
                f"[update] path does not exist: {local}\n"
                f"        (originally: {raw!r}; resolved against "
                f"RELEASE_ROOT={RELEASE_ROOT})"
            )

        # Validate that the path lives under RELEASE_ROOT so we can
        # derive its path-in-repo unambiguously.
        try:
            rel = local.relative_to(RELEASE_ROOT)
        except ValueError:
            sys.exit(
                f"[update] path is outside RELEASE_ROOT: {local}\n"
                f"        RELEASE_ROOT={RELEASE_ROOT}\n"
                f"        Update gate derives path-in-repo from the relative path; "
                f"external files don't have a natural destination."
            )

        if local.is_file():
            expanded.append((local, rel.as_posix()))
        elif local.is_dir():
            # Walk the directory in sorted order for a stable commit
            # ordering (the HF git log is easier to read when files
            # land in a deterministic sequence).
            for f in sorted(local.rglob("*")):
                if f.is_file():
                    rel_inside = f.relative_to(RELEASE_ROOT).as_posix()
                    expanded.append((f, rel_inside))
        else:
            sys.exit(f"[update] not a file or directory: {local}")

    if not expanded:
        sys.exit("[update] FAIL — no files matched. Pass --path or populate "
                 "update_paths in upload_hf.yaml.")
    return expanded


def gate_update(
    api: HfApi,
    paths: list[str],
    message: str,
    dry_run: bool,
) -> None:
    """Push a hand-picked set of files in a single atomic commit.

    Overwrites whatever's at the same path on HF. Designed for
    follow-up edits after --gate full has already shipped the bulk
    of the release — much faster than re-running full just to update
    a handful of files, and bundles everything into one commit
    instead of one-per-file.

    Use cases that motivated this gate:
      - patched data-correction files (e.g. four episode_meta.json +
        regenerated meta/episodes.parquet + meta/episodes.jsonl +
        meta/manifest.json after a case_type fix);
      - pushing files normally blocked by base_ignore (README, etc.)
        that the more-specialized gates don't cover;
      - one-shot fixups where you know exactly which files changed.

    NOTE on dedup: unlike --gate full's upload_large_folder (which
    skips files whose local SHA matches the remote SHA), this gate
    transfers every listed file unconditionally. HF's LFS-blob layer
    will still dedup unchanged binaries server-side, but you pay
    the round-trip bandwidth. For very large changesets, prefer
    --gate full; for ≤ a few dozen files, this gate is fine.
    """
    expanded = _expand_update_paths(paths)
    total_bytes = sum(p.stat().st_size for p, _ in expanded)

    print(f"[update] target repo   : {REPO_ID}")
    print(f"[update] commit message: {message!r}")
    print(f"[update] file count    : {len(expanded)}")
    print(f"[update] total bytes   : {total_bytes:,} ({total_bytes / 1e6:.2f} MB)")
    print(f"[update] files (local → path-in-repo):")
    # Cap the printed list at 50 entries so a giant directory walk
    # doesn't drown the terminal; the user can still see the count.
    PRINT_CAP = 50
    for local, in_repo in expanded[:PRINT_CAP]:
        size = local.stat().st_size
        print(f"  + {in_repo:<60s}  ({size:>12,} B)  ← {local}")
    if len(expanded) > PRINT_CAP:
        print(f"  ... and {len(expanded) - PRINT_CAP} more")

    if dry_run:
        print("[update] DRY-RUN — no files were pushed. Re-run without --dry-run.")
        return

    # Build the commit. CommitOperationAdd carries the file body + the
    # destination path; HfApi.create_commit bundles every op into one
    # atomic commit on the branch.
    from huggingface_hub import CommitOperationAdd
    ops = [
        CommitOperationAdd(path_in_repo=in_repo, path_or_fileobj=str(local))
        for local, in_repo in expanded
    ]
    api.create_commit(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        operations=ops,
        commit_message=message,
    )
    print(
        f"[update] OK — pushed {len(expanded)} files in one commit.\n"
        f"        Inspect at: https://huggingface.co/datasets/{REPO_ID}\n"
        f"        Edit history: https://huggingface.co/datasets/{REPO_ID}/commits/main"
    )


# ---------------------------------------------------------------------------
# Gate 5e — auto-detected update (manifest delta)
# ---------------------------------------------------------------------------

def gate_update_all(api: HfApi, message: str, dry_run: bool) -> None:
    """Push every file whose local SHA-256 differs from the remote copy.

    Compares the **local** ``meta/manifest.json`` (which the packer +
    ``surgsync index`` keep up to date with SHA-256 + size for every
    shipped file) against the **remote** manifest on HF. Any entry
    whose hash differs, or that is missing remotely, gets pushed.
    The local manifest itself is always included in the commit (its
    own SHA changes whenever any entry's SHA changes), so the public
    release stays self-consistent after the push.

    Why this is fast:
      - ``meta/manifest.json`` is ~2.5 MB. Downloading the remote
        copy is one HTTP request.
      - No re-hashing of local data — we trust the local manifest's
        SHAs, which is what ``build_manifest`` just computed.
      - Diff is a dict comparison; takes milliseconds.

    When to use it vs the other gates:
      - ``--gate update``    — you know exactly which files to push;
                               pass them with ``--path``.
      - ``--gate update-all`` (this one) — let the gate figure it out.
      - ``--gate full``      — first push, or you want to re-walk
                               everything; honors base_ignore.

    Refuses to run when:
      - The local manifest is missing (run ``surgsync index`` first).
      - The remote manifest doesn't exist (likely a first push —
        use ``--gate full``).

    Files NOT in the manifest (by manifest-builder design):
      ``README.md``, ``CHANGELOG.md``, ``.gitattributes``,
      ``.surgsync_*.json`` sentinels, ``.logs/*.jsonl``, the manifest
      itself. The gate cannot detect changes to those — push them
      via ``--gate card``, ``--gate lfs``, or ``--gate update --path
      <p>`` as appropriate.
    """
    # 1. Load local manifest.
    local_manifest_path = RELEASE_ROOT / "meta" / "manifest.json"
    if not local_manifest_path.is_file():
        sys.exit(
            f"[update-all] local manifest missing: {local_manifest_path}\n"
            f"        Run `surgsync index <release_root>` first so the "
            f"manifest's SHAs match what's on disk."
        )
    with open(local_manifest_path) as f:
        local_manifest = json.load(f)
    local_files = local_manifest.get("files", {})
    if not local_files:
        sys.exit("[update-all] local manifest has no `files` block")

    # 2. Download remote manifest into a temp location. We never write
    # it into the release tree — the local manifest is authoritative.
    from huggingface_hub import hf_hub_download
    try:
        remote_manifest_path = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename="meta/manifest.json",
        )
    except Exception as e:
        msg = str(e)
        if "404" in msg or "EntryNotFound" in msg or "not found" in msg.lower():
            sys.exit(
                "[update-all] remote meta/manifest.json not found on HF.\n"
                "        This is likely a first push to the repo — use "
                "`--gate full` first, then come back to --gate update-all "
                "for follow-up edits."
            )
        raise
    with open(remote_manifest_path) as f:
        remote_manifest = json.load(f)
    remote_files = remote_manifest.get("files", {})

    print(f"[update-all] local  manifest: {len(local_files)} files, "
          f"{int(local_manifest.get('total_size_bytes', 0)) / 1e9:.2f} GB")
    print(f"[update-all] remote manifest: {len(remote_files)} files, "
          f"{int(remote_manifest.get('total_size_bytes', 0)) / 1e9:.2f} GB")

    # 3. Compute the delta. For every entry in the local manifest:
    #   - if path not in remote → new file, push it
    #   - if remote sha256 differs → changed file, push it
    #   - else → unchanged, skip
    # We do NOT walk the remote-only set (files that exist on HF but
    # not in the local manifest) — those would be deletions, which is
    # outside the "update" semantics; the gate is one-way push only.
    to_push: list[tuple[Path, str]] = []
    new_count = 0
    changed_count = 0
    for rel_path, entry in local_files.items():
        local_sha = entry.get("sha256")
        remote_entry = remote_files.get(rel_path)
        if remote_entry is None:
            new_count += 1
        elif remote_entry.get("sha256") != local_sha:
            changed_count += 1
        else:
            continue  # unchanged

        local = (RELEASE_ROOT / rel_path).resolve()
        if not local.is_file():
            # Manifest references a file that's not on disk. The
            # local manifest is supposed to be in sync with the
            # working tree — if it isn't, the user probably has a
            # half-finished edit. Warn and skip rather than abort the
            # entire push.
            print(f"[update-all] WARN: manifest references missing file: {rel_path}")
            continue
        to_push.append((local, rel_path))

    # 4a. Defense in depth: filter the delta through base_ignore. The
    # local manifest *should* already exclude operational paths like
    # `.cache/`, `.logs/`, `tmp/`, etc., but a regression in
    # `build_manifest` can quietly pull them in (real incident:
    # `.cache/huggingface/upload/...` lock + metadata files from
    # upload_large_folder's resumable-upload state got included on
    # 2026-06-03). We refuse to push anything matching base_ignore
    # regardless of what the manifest says — those paths have no
    # business landing on HF.
    filtered_out: list[str] = []
    kept: list[tuple[Path, str]] = []
    for local, in_repo in to_push:
        if any(fnmatch(in_repo, pat) for pat in BASE_IGNORE):
            filtered_out.append(in_repo)
        else:
            kept.append((local, in_repo))
    if filtered_out:
        print(
            f"[update-all] WARN: {len(filtered_out)} manifest entries "
            f"excluded by base_ignore. The local manifest is likely "
            f"polluted — clean these out of `.cache/`, `.logs/`, etc., "
            f"and re-run `surgsync index` before pushing for clean state."
        )
        EX_CAP = 8
        for p in filtered_out[:EX_CAP]:
            print(f"  - {p}")
        if len(filtered_out) > EX_CAP:
            print(f"  ... and {len(filtered_out) - EX_CAP} more")
    to_push = kept

    # 4b. Always re-push the manifest itself. Its content changes
    # whenever any underlying file's SHA changes — even if the diff
    # found 0 file deltas (e.g. the user only touched README.md
    # which isn't manifest-tracked), the local manifest's
    # `generated_at_utc` will differ from the remote.
    manifest_rel = "meta/manifest.json"
    if not any(in_repo == manifest_rel for _, in_repo in to_push):
        to_push.append((local_manifest_path, manifest_rel))

    # If the only file to push is the manifest itself AND the manifest
    # body is byte-identical to remote, there's nothing to do.
    if len(to_push) == 1 and to_push[0][1] == manifest_rel:
        import hashlib
        with open(local_manifest_path, "rb") as f:
            local_mn_sha = hashlib.sha256(f.read()).hexdigest()
        with open(remote_manifest_path, "rb") as f:
            remote_mn_sha = hashlib.sha256(f.read()).hexdigest()
        if local_mn_sha == remote_mn_sha:
            print("[update-all] no file deltas and manifest is byte-identical "
                  "to remote. Nothing to push.")
            return

    total_bytes = sum(p.stat().st_size for p, _ in to_push)

    print(f"[update-all] new files     : {new_count}")
    print(f"[update-all] changed files : {changed_count}")
    print(f"[update-all] manifest itself: {'+' if to_push[-1][1] == manifest_rel else '-'}")
    print(f"[update-all] total to push : {len(to_push)} files, "
          f"{total_bytes:,} bytes ({total_bytes / 1e6:.2f} MB)")
    print(f"[update-all] target repo   : {REPO_ID}")
    print(f"[update-all] commit message: {message!r}")

    # Cap the printed list — a giant delta would drown the terminal.
    PRINT_CAP = 30
    print(f"[update-all] delta files (first {PRINT_CAP}):")
    for local, in_repo in to_push[:PRINT_CAP]:
        size = local.stat().st_size
        marker = "NEW" if remote_files.get(in_repo) is None else "MOD"
        print(f"  [{marker}] {in_repo:<70s}  ({size:>12,} B)")
    if len(to_push) > PRINT_CAP:
        print(f"  ... and {len(to_push) - PRINT_CAP} more")

    if dry_run:
        print("[update-all] DRY-RUN — no files pushed. Re-run without --dry-run.")
        return

    # 5. Push as a single atomic commit.
    from huggingface_hub import CommitOperationAdd
    ops = [
        CommitOperationAdd(path_in_repo=in_repo, path_or_fileobj=str(local))
        for local, in_repo in to_push
    ]
    api.create_commit(
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        operations=ops,
        commit_message=message,
    )
    print(
        f"[update-all] OK — pushed {len(to_push)} files in one commit.\n"
        f"        Inspect at: https://huggingface.co/datasets/{REPO_ID}\n"
        f"        Edit history: https://huggingface.co/datasets/{REPO_ID}/commits/main"
    )


# ---------------------------------------------------------------------------
# Gate 6 — full bulk push
# ---------------------------------------------------------------------------

def gate_full(api: HfApi, include_meta: bool, num_workers: int) -> None:
    """The real upload. Resumable; safe to Ctrl-C and re-run.

    upload_large_folder behavior:
      - chunks files into many commits (HF has a single-commit size cap)
      - parallel workers within each chunk
      - persists progress in <folder>/.cache/huggingface/upload-large-folder/
        so a re-run picks up where it left off
      - dedupes by content hash on the HF side; already-pushed files skip
    """
    ignore = list(BASE_IGNORE)
    if not include_meta:
        # Two patterns because meta/ contains both flat files (dataset.json)
        # and nested Hive-partitioned dirs (episodes.parquet/task=*/part-*).
        ignore += ["meta/*", "meta/**"]
        print("[full] meta/ EXCLUDED (--no-include-meta)")
    else:
        print("[full] meta/ INCLUDED")

    print(f"[full] release root: {RELEASE_ROOT}")
    print(f"[full] target repo : {REPO_ID}")
    print(f"[full] workers     : {num_workers}")
    print(f"[full] ignore      : {ignore}")
    print(f"[full] hf_transfer : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')!r}")

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(RELEASE_ROOT),
        ignore_patterns=ignore,
        num_workers=num_workers,
        # print_report/print_report_every: emit a status table every N
        # seconds so you can grep progress in long-running tmux logs.
        print_report=True,
        print_report_every=60,
    )
    print("[full] upload_large_folder returned cleanly.")


# ---------------------------------------------------------------------------
# Gate 7 — squash history
# ---------------------------------------------------------------------------

def gate_squash(api: HfApi, confirm: bool, message: str) -> None:
    """Collapse every commit on the default branch into a single one.

    Why:
      By the time the staged upload finishes, the HF dataset has many
      commits — one per gate (smoke, lfs, canary, cleanup, meta) plus
      N from upload_large_folder's internal chunking (HF caps commit
      size, so a 670 GB push lands as ~dozens of chunked commits).
      For a public release, that history is operational noise; the
      dataset is more legible if every consumer sees one well-named
      commit instead of "Cleanup: drop ..." / "Add files: chunk 17/42".

    What it does:
      HfApi.super_squash_history rewrites the branch ref to point at a
      single new commit whose tree equals the current branch tip. The
      net file set on HF is unchanged; only the commit DAG flattens.

    Safety:
      - Dry-run by default. Prints current commit count + the top of
        the log so you can see what's about to disappear.
      - Pass --confirm to actually squash. Prior commit SHAs become
        unreachable after this; if you had bookmarked specific
        revisions, save them BEFORE squashing.
      - Run AFTER --gate full has returned cleanly. Squashing mid-upload
        risks dropping in-flight commits.
    """
    # Show current history so the user can sanity-check.
    try:
        # list_repo_commits returns newest-first; we just want a count + top 5.
        commits = api.list_repo_commits(repo_id=REPO_ID, repo_type=REPO_TYPE)
        n_total = len(commits)
        print(f"[squash] current commit count on `main`: {n_total}")
        print("[squash] most recent commits (top of log):")
        for c in commits[:5]:
            # Each commit has commit_id, title, created_at, message
            title = (c.title or "").splitlines()[0] if c.title else "(no title)"
            print(f"  {c.commit_id[:10]}  {title}")
        if n_total > 5:
            print(f"  ... and {n_total - 5} older commit(s)")
    except Exception as e:
        # Don't abort — `list_repo_commits` is informational. The squash
        # itself doesn't depend on it.
        print(f"[squash] (could not enumerate commits: {e})")

    print(f"[squash] proposed commit message: {message!r}")

    if not confirm:
        print("[squash] DRY-RUN — re-run with `--confirm` to actually squash.")
        return

    # Real squash. Single API call; HF handles the rewrite server-side.
    api.super_squash_history(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=message,
    )
    print(f"[squash] OK — branch collapsed to 1 commit: {message!r}")

    # Verify by re-querying.
    try:
        new_commits = api.list_repo_commits(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"[squash] post-squash commit count: {len(new_commits)}")
    except Exception as e:
        print(f"[squash] (could not re-verify: {e})")


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------

def dry_run(include_meta: bool) -> None:
    """Approximate what --gate full would upload, without uploading anything.

    Uses fnmatch as a stand-in for huggingface_hub's gitignore-style
    matcher; the byte total is therefore approximate. Good enough to
    catch "oh, I forgot to exclude that giant temp dir" before kickoff.
    """
    ignore = list(BASE_IGNORE)
    if not include_meta:
        ignore += ["meta/*", "meta/**"]

    n_files = 0
    n_bytes = 0
    for p in RELEASE_ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(RELEASE_ROOT).as_posix()
        if any(fnmatch(rel, pat) for pat in ignore):
            continue
        n_files += 1
        n_bytes += p.stat().st_size

    print(f"[dry-run] include_meta = {include_meta}")
    print(f"[dry-run] ignore_patterns = {ignore}")
    print(
        f"[dry-run] would upload approximately {n_files} files, "
        f"{n_bytes / 1e9:.1f} GB"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # ``global`` must be declared BEFORE any use of the name in the
    # function body — Python rejects "used prior to global declaration".
    # We mutate RELEASE_ROOT below (after CLI parsing) AND read it in
    # the argparse `default=` for --release-root. Declare it up front.
    global RELEASE_ROOT

    # Stage 1 — pre-parse just enough to find --config. We do this in a
    # throwaway parser (add_help=False so it doesn't intercept --help)
    # because the full parser's argument defaults are sourced from the
    # YAML; we therefore need to load the YAML before constructing it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _remaining = pre.parse_known_args()

    # Stage 2 — load YAML and bind module globals. Any CLI flag the user
    # supplies in stage 3 below overrides these values; flags omitted
    # fall back to the YAML.
    cfg = load_config(Path(pre_args.config))
    apply_config(cfg)

    # Stage 3 — full parser. Defaults read from the just-bound globals.
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
        required=False,
        choices=["smoke", "lfs", "canary", "cleanup", "meta", "card", "update", "update-all", "full", "squash"],
        help=(
            "which gate to run. Typical order: smoke → lfs → canary → "
            "(cleanup) → full → card → (squash). Use `update` for "
            "follow-up edits — push a hand-picked set of files in one "
            "atomic commit, overwriting remote copies. Use `cleanup` to "
            "remove the smoke + canary uploads from HF before the bulk "
            "push; pair with --confirm to actually delete. Use `meta` "
            "only if a prior --gate full was run with --no-include-meta. "
            "Use `card` to push a local markdown file as README.md (the "
            "HF dataset card) — see --card / --card-path. Use `squash` "
            "AFTER --gate full to collapse the operational commit log "
            "into a single 'Initial release' commit; pair with --confirm. "
            "Omit to use --dry-run."
        ),
    )
    p.add_argument(
        "--include-meta", dest="include_meta",
        action="store_true", default=DEFAULT_INCLUDE_META,
        help=f"include meta/ in --gate full (config default: {DEFAULT_INCLUDE_META})",
    )
    p.add_argument(
        "--no-include-meta", dest="include_meta",
        action="store_false",
        help="exclude meta/ from --gate full (e.g. meta is still being regenerated)",
    )
    p.add_argument(
        "--canary", type=str, default=DEFAULT_CANARY_OVERRIDE,
        help=(
            "explicit episode path (relative to release root) to use as the "
            "canary, e.g. online_data/episodes/peg_transfer/96. "
            "Default: auto-pick the smallest clip under "
            f"{CANARY_TASK}/ (overridable via config `canary_override`)."
        ),
    )
    p.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help=f"parallel uploaders for --gate full (config default: {DEFAULT_NUM_WORKERS}; 4-16 reasonable)",
    )
    p.add_argument(
        "--confirm", action="store_true",
        help=(
            "required for --gate cleanup and --gate squash to actually "
            "mutate HF state. Without it, those gates run as dry-runs and "
            "only print what would happen."
        ),
    )
    p.add_argument(
        "--squash-message", type=str,
        default=DEFAULT_SQUASH_MESSAGE,
        help=(
            "commit message for the single post-squash commit "
            "(only used by --gate squash --confirm). Config default is in "
            "upload_hf.yaml:squash_message."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print what --gate full would upload, then exit. Does not touch HF.",
    )
    p.add_argument(
        "--release-root", type=str, default=str(RELEASE_ROOT),
        help=(
            "path to the packed dataset on disk. Overrides the YAML "
            "`release_root`. Required because this script no longer lives "
            "inside the release tree (it sits in tests/ of the toolkit repo)."
        ),
    )
    p.add_argument(
        "--card", type=str, default=None,
        choices=list(CARDS) if CARDS else None,
        help=(
            f"for --gate card: pick a named card from upload_hf.yaml:cards "
            f"(available: {sorted(CARDS)}). Default: "
            f"{DEFAULT_CARD_NAME!r} (the YAML's `card_default`). "
            "Overridden by --card-path if both are passed."
        ),
    )
    p.add_argument(
        "--card-path", type=str, default=None,
        help=(
            "for --gate card: explicit path to the markdown file to push "
            "as README.md. Absolute, or relative to release_root. Wins "
            "over --card when both are passed."
        ),
    )
    p.add_argument(
        "--path", type=str, action="append", default=None,
        help=(
            "for --gate update: one path (file or directory) to push, "
            "relative to release_root or absolute. Repeatable — pass "
            "--path multiple times to bundle several entries into one "
            "atomic commit. Directories are walked recursively. When "
            "omitted, the YAML's `update_paths` list is used."
        ),
    )
    p.add_argument(
        "--update-message", type=str, default=None,
        help=(
            "for --gate update / --gate update-all: commit message for "
            "the single atomic commit. Default comes from YAML "
            f"`update_message` ({DEFAULT_UPDATE_MESSAGE!r}) for `update` and "
            f"`update_all_message` ({DEFAULT_UPDATE_ALL_MESSAGE!r}) for "
            "`update-all`."
        ),
    )
    args = p.parse_args()

    # Apply CLI overrides into module globals. RELEASE_ROOT is the only
    # global we mutate here because every other YAML value is read once
    # at parse time (via the `default=` defaults above) and reaches the
    # gate functions via `args.*` from this point on. (The ``global``
    # statement was hoisted to the top of main() so the read above —
    # ``default=str(RELEASE_ROOT)`` — sees the same name as this write.)
    RELEASE_ROOT = Path(args.release_root).resolve()
    if not RELEASE_ROOT.is_dir():
        sys.exit(
            f"[init] --release-root {RELEASE_ROOT} does not exist or is not a "
            "directory. Point this at your packed dataset (the one containing "
            "meta/dataset.json, online_data/, offline_data/, etc.)."
        )

    if not args.gate and not args.dry_run:
        p.error("must pass --gate <choice> or --dry-run")

    api = HfApi()
    check_auth(api)
    ensure_repo(api)

    # --dry-run currently has two meanings depending on context:
    #   1. With no --gate (or --gate full): show what the bulk
    #      --gate full upload would push, then exit. Handled here.
    #   2. With --gate update: print the planned commit, don't push.
    #      Handled inside gate_update — pass through the flag.
    # This split keeps --dry-run behaving the same for the legacy
    # full-push preview while letting the update gate use the same
    # flag name naturally.
    # --dry-run is overloaded: with --gate update or --gate update-all
    # it means "print the planned commit, don't push" (handled inside
    # the gate function); otherwise it means "show what --gate full
    # would push, then exit" (the legacy behavior, handled here).
    if args.dry_run and args.gate not in ("update", "update-all"):
        dry_run(args.include_meta)
        return

    if args.gate == "smoke":
        gate_smoke(api)
    elif args.gate == "lfs":
        gate_lfs(api)
    elif args.gate == "canary":
        canary_rel = pick_canary_rel(args.canary)
        gate_canary(api, canary_rel)
    elif args.gate == "cleanup":
        # Same auto-discovery for canary path so cleanup deletes whatever
        # the canary actually uploaded (--canary <override> applies here too
        # if the user used it for --gate canary).
        canary_rel = pick_canary_rel(args.canary)
        gate_cleanup(api, canary_rel, args.confirm)
    elif args.gate == "meta":
        gate_meta(api)
    elif args.gate == "card":
        gate_card(api, args.card, args.card_path)
    elif args.gate == "update":
        # CLI --path overrides the YAML default; if neither is given,
        # fail loudly rather than uploading nothing.
        paths = args.path if args.path else list(DEFAULT_UPDATE_PATHS)
        if not paths:
            p.error(
                "--gate update needs at least one --path <p> "
                "(or populate `update_paths` in upload_hf.yaml)"
            )
        message = args.update_message or DEFAULT_UPDATE_MESSAGE
        gate_update(api, paths, message, args.dry_run)
    elif args.gate == "update-all":
        message = args.update_message or DEFAULT_UPDATE_ALL_MESSAGE
        gate_update_all(api, message, args.dry_run)
    elif args.gate == "full":
        gate_full(api, args.include_meta, args.num_workers)
    elif args.gate == "squash":
        gate_squash(api, args.confirm, args.squash_message)


if __name__ == "__main__":
    main()
