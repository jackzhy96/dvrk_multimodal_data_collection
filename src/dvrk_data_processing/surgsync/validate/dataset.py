"""Dataset-wide validator — checks I-1 through I-7.

I-1 every episode dir has a matching row in `meta/episodes.parquet`
I-2 every row in episodes.parquet has a matching dir
I-3 episodes.jsonl row count == episodes.parquet row count + same ids
I-4 every per-modality parquet (timestamp / ECM / PSM1 / PSM2 / annotation) has row count == episode.length_frames
I-5 every file under <dataset_root> (except manifest) is listed in manifest
I-6 every annotation value in annotation.parquet has a key in tasks.jsonl vocab
I-7 dataset.tasks superset of partition keys under episodes.parquet/task=*/

I-6 requires `meta/tasks.jsonl` — if absent we surface as WARNING.
"""
from __future__ import annotations
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import List

import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.validate.types import ValidationIssue


log = logging.getLogger(__name__)


def _iter_episode_dirs(dataset_root: Path):
    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir() or not (dataset_dir / "episodes").is_dir():
            continue
        for task_dir in (dataset_dir / "episodes").iterdir():
            if not task_dir.is_dir():
                continue
            for ep_dir in task_dir.iterdir():
                # Skip in-flight or crashed episodes — the sentinel
                # file is the marker for "fully written".
                if (ep_dir.is_dir()
                        and (ep_dir / "episode_meta.json").is_file()
                        and (ep_dir / ".surgsync_complete.json").is_file()):
                    yield task_dir.name, ep_dir


def validate_dataset(
    dataset_root: Path,
    *,
    full_manifest_check: bool = False,
    manifest_sample_pct: float = 0.01,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    dataset_root = Path(dataset_root)

    if not dataset_root.is_dir():
        return [ValidationIssue("ERROR", "ds_missing",
                                f"dataset root does not exist: {dataset_root}")]

    meta_dir = dataset_root / "meta"
    if not meta_dir.is_dir():
        return [ValidationIssue("ERROR", "ds_no_meta",
                                f"missing meta/ under {dataset_root} — run `surgsync index` first")]

    # Collect episodes from disk.
    disk_episodes: dict[str, tuple[str, Path, int]] = {}   # id -> (task, dir, length_frames)
    for task, ep_dir in _iter_episode_dirs(dataset_root):
        with open(ep_dir / "episode_meta.json") as f:
            em = json.load(f)
        disk_episodes[em["episode_id"]] = (task, ep_dir, em["length_frames"])

    # ---- I-1, I-2: parquet ↔ disk -----------------------------------------
    eps_parquet = meta_dir / "episodes.parquet"
    parquet_ids: set[str] = set()
    parquet_rows: dict[str, dict] = {}
    if eps_parquet.is_dir():
        # Read each part file directly via ParquetFile to bypass
        # pyarrow's auto Hive-partition inference (which mis-types the
        # `task` column as a dictionary vs the on-disk string).
        for part in eps_parquet.rglob("*.parquet"):
            tab = pq.ParquetFile(part).read()
            for row in tab.to_pylist():
                parquet_ids.add(row["episode_id"])
                parquet_rows[row["episode_id"]] = row
    else:
        issues.append(ValidationIssue("ERROR", "I-1",
                                      "meta/episodes.parquet/ not present"))
        return issues

    for eid in set(disk_episodes) - parquet_ids:
        issues.append(ValidationIssue("ERROR", "I-1",
                                      f"episode dir {eid} not in episodes.parquet"))
    for eid in parquet_ids - set(disk_episodes):
        issues.append(ValidationIssue("ERROR", "I-2",
                                      f"episodes.parquet row {eid} has no matching dir"))

    # ---- I-3: episodes.jsonl matches ---------------------------------------
    jsonl_path = meta_dir / "episodes.jsonl"
    if not jsonl_path.is_file():
        issues.append(ValidationIssue("ERROR", "I-3", "meta/episodes.jsonl missing"))
    else:
        jsonl_ids: set[str] = set()
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                jsonl_ids.add(json.loads(line)["episode_id"])
        if jsonl_ids != parquet_ids:
            issues.append(ValidationIssue(
                "ERROR", "I-3",
                f"episodes.jsonl ids != episodes.parquet ids "
                f"(jsonl-only={jsonl_ids - parquet_ids}, parquet-only={parquet_ids - jsonl_ids})",
            ))

    # ---- I-4: every per-modality parquet length == length_frames -----------
    PER_MODALITY = (
        "timestamp.parquet", "ECM.parquet",
        "PSM1.parquet", "PSM2.parquet", "annotation.parquet",
    )
    for eid, (task, ep_dir, declared) in disk_episodes.items():
        for name in PER_MODALITY:
            fp = ep_dir / name
            if not fp.is_file():
                issues.append(ValidationIssue("ERROR", "I-4",
                                              f"{eid}: {name} missing"))
                continue
            n = pq.read_metadata(fp).num_rows
            if n != declared:
                issues.append(ValidationIssue("ERROR", "I-4",
                                              f"{eid}: {name} has {n} rows; "
                                              f"episode_meta.json says {declared}"))

    # ---- I-5: manifest covers every file (sampled or full) -----------------
    manifest_path = meta_dir / "manifest.json"
    if not manifest_path.is_file():
        issues.append(ValidationIssue("WARNING", "I-5",
                                      "meta/manifest.json missing — run `surgsync index`"))
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)
        listed = manifest.get("files", {})
        # Membership check: every non-internal file under dataset_root
        # should be a key. We don't recompute every SHA unless asked.
        # Skip rules MUST match `index/manifest.py:_skip_path` — the
        # manifest's skip list is the authoritative definition of
        # "not shipped". Validator skips the same files so coverage
        # comparison is apples-to-apples.
        SENTINEL_NAMES = {
            ".surgsync_running.json",
            ".surgsync_failed.json",
            ".surgsync_complete.json",
        }
        disk_files = []
        for p in dataset_root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(dataset_root).as_posix()
            parts = p.relative_to(dataset_root).parts
            if parts and parts[0] in (".staging", ".logs", ".tmp"):
                continue
            if parts == ("meta", "manifest.json"):
                continue
            # Per-episode lifecycle sentinels are runtime state.
            if p.name in SENTINEL_NAMES:
                continue
            disk_files.append(rel)
        missing = set(disk_files) - set(listed.keys())
        extra = set(listed.keys()) - set(disk_files)
        for rel in sorted(missing):
            issues.append(ValidationIssue("ERROR", "I-5",
                                          f"file under dataset not in manifest: {rel}"))
        for rel in sorted(extra):
            issues.append(ValidationIssue("ERROR", "I-5",
                                          f"manifest lists file not on disk: {rel}"))

        # SHA check — sample (or full).
        sample_rels = list(set(disk_files) & set(listed.keys()))
        if not full_manifest_check:
            k = max(1, int(len(sample_rels) * manifest_sample_pct))
            sample_rels = random.sample(sample_rels, k=min(k, len(sample_rels)))
        for rel in sample_rels:
            h = hashlib.sha256()
            with open(dataset_root / rel, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != listed[rel]["sha256"]:
                issues.append(ValidationIssue("ERROR", "I-5",
                                              f"sha256 mismatch on {rel}"))

    # ---- I-6: annotation values cross-checked against the workflow text
    # descriptions in `workflow_description/`. The parquet now stores
    # verbalized text (not raw ids), so the check is: every observed
    # phase/step/gesture cell value must appear in the relevant
    # description table. Pass-through ids (cells where the id wasn't
    # in the table) trip the check, which is what we want — they
    # surface as "unknown id NN" rather than silently going through.
    from dvrk_data_processing.surgsync.serde.workflow_text import (
        PHASE_DESCRIPTIONS, STEP_DESCRIPTIONS, SUTURING_STEP_DESCRIPTIONS,
        SUTURING_GESTURE_DESCRIPTIONS, DISSECTION_GESTURE_DESCRIPTIONS,
        SUTURING_TASKS, DISSECTION_TASKS,
    )

    phase_texts = set(PHASE_DESCRIPTIONS.values())
    # The packer's verbalize_step routes by task to the right phase in
    # `workflow_description.json` and falls back to a full-phase scan
    # for callers that pass task=None. Either path resolves to a text
    # already present in some phase's step block, so the union of
    # STEP_DESCRIPTIONS (= union of every phase's step dict) and
    # SUTURING_STEP_DESCRIPTIONS (= phase 1's step block) covers every
    # legitimate cell value.
    step_texts        = set(STEP_DESCRIPTIONS.values()) | set(SUTURING_STEP_DESCRIPTIONS.values())
    sut_gesture_texts = set(SUTURING_GESTURE_DESCRIPTIONS.values())
    dis_gesture_texts = set(DISSECTION_GESTURE_DESCRIPTIONS.values())

    for task, ep_dir in _iter_episode_dirs(dataset_root):
        # Annotations now live in annotation.parquet (the per-modality
        # split). ParquetFile bypass avoids Hive-partition inference.
        tab = pq.ParquetFile(ep_dir / "annotation.parquet").read(
            columns=["phase", "step", "gesture.PSM1", "gesture.PSM2"]
        )
        # Build the (label → expected text set) table for this task.
        if task in SUTURING_TASKS:
            gesture_texts = sut_gesture_texts
        elif task in DISSECTION_TASKS:
            gesture_texts = dis_gesture_texts
        else:
            gesture_texts = None  # no vocab → skip gesture I-6 for this task
        checks = [
            ("phase",        phase_texts),
            ("step",         step_texts),
            ("gesture.PSM1", gesture_texts),
            ("gesture.PSM2", gesture_texts),
        ]
        for col, expected in checks:
            if expected is None:
                continue
            seen = {x for x in tab.column(col).to_pylist() if x is not None}
            unknown = seen - expected
            for u in sorted(unknown):
                # Truncate the displayed cell — descriptions are long.
                shown = (u[:60] + "...") if len(u) > 60 else u
                issues.append(ValidationIssue(
                    "ERROR", "I-6",
                    f"{task}/{ep_dir.name}: {col} contains text not from the "
                    f"workflow_description table: {shown!r}",
                ))

    # `meta/tasks.jsonl` is now reference-only (it carries the per-task
    # instruction strings for VLA training). Its absence is a WARNING.
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if not tasks_jsonl.is_file():
        issues.append(ValidationIssue("WARNING", "I-6",
                                      "meta/tasks.jsonl absent — per-task instruction "
                                      "strings unavailable. Required before public release."))

    # ---- I-7: dataset.json.tasks ⊇ observed task partitions ----------------
    ds_json = meta_dir / "dataset.json"
    if ds_json.is_file():
        ds = json.loads(ds_json.read_text())
        declared_tasks = set(ds.get("tasks", []))
        observed_tasks = {task for task, _ in _iter_episode_dirs(dataset_root)}
        extra = observed_tasks - declared_tasks
        for t in sorted(extra):
            issues.append(ValidationIssue(
                "ERROR", "I-7",
                f"task {t!r} present under episodes/ but not in dataset.json.tasks",
            ))

    return issues
