"""Release-time README + CHANGELOG generator (`surgsync release`)."""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------

# Tolerant — accepts M, M.m, or M.m.p. The packer build default is "1.0".
_SEMVER_RE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$")


def _parse_semver(s: str) -> tuple[int, int, int]:
    """Parse `M[.m[.p]]` → `(M, m, p)`. Missing components default to 0."""
    m = _SEMVER_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"data_version {s!r} is not parseable (expected M, M.m, or M.m.p)"
        )
    return int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)


def bump_version(current: str, kind: str) -> str:
    """`patch` / `minor` / `major`."""
    M, m, p = _parse_semver(current)
    if kind == "patch": return f"{M}.{m}.{p+1}"
    if kind == "minor": return f"{M}.{m+1}.0"
    if kind == "major": return f"{M+1}.0.0"
    raise ValueError(f"unknown bump kind: {kind!r}")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class _ReleaseInventory:
    meta:             dict
    episode_count:    int
    episodes_by_task: dict[str, int]
    dataset_names:    list[str]
    total_frames:     int


def _scan_inventory(dataset_root: Path) -> _ReleaseInventory:
    """Walk completed episodes and tally frames + tasks + datasets."""
    meta_path = dataset_root / "meta" / "dataset.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta/dataset.json missing under {dataset_root}")
    meta = json.loads(meta_path.read_text())

    episodes_by_task: dict[str, int] = {}
    dataset_names: set[str] = set()
    total_frames = 0
    ep_count = 0
    for ds_dir in sorted(dataset_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in {"meta", ".logs"}:
            continue
        ep_root = ds_dir / "episodes"
        if not ep_root.is_dir():
            continue
        for task_dir in sorted(ep_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for clip_dir in sorted(task_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                if not (clip_dir / ".surgsync_complete.json").is_file():
                    continue
                em_path = clip_dir / "episode_meta.json"
                if em_path.is_file():
                    try:
                        em = json.loads(em_path.read_text())
                        total_frames += int(em.get("length_frames", 0))
                    except Exception:
                        log.warning("could not read %s; ignoring", em_path)
                ep_count += 1
                episodes_by_task[task_dir.name] = episodes_by_task.get(task_dir.name, 0) + 1
                dataset_names.add(ds_dir.name)

    return _ReleaseInventory(
        meta=meta,
        episode_count=ep_count,
        episodes_by_task=dict(sorted(episodes_by_task.items())),
        dataset_names=sorted(dataset_names),
        total_frames=total_frames,
    )


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _render_readme(inv: _ReleaseInventory, dataset_root: Path) -> str:
    meta = inv.meta
    L: list[str] = []
    L.append(f"# {meta.get('name', 'SurgSync')} dataset — v{meta.get('data_version', '?')}")
    L.append("")
    L.append(f"- **schema_version**: `{meta.get('schema_version', '?')}`")
    L.append(f"- **data_version**:  `{meta.get('data_version', '?')}`")
    L.append(f"- **release_option**: `{meta.get('release_option', '?')}` "
             "(A=structure only, B=+preprocess, C=+preview)")
    L.append(f"- **created_at_utc**: `{meta.get('created_at_utc', '?')}`")
    L.append("")
    L.append(f"This release contains **{inv.episode_count} episode(s)** "
             f"totaling **{inv.total_frames:,} frame(s)** across "
             f"**{len(inv.dataset_names)} dataset partition(s)** "
             f"and **{len(inv.episodes_by_task)} task(s)**.")
    L.append("")
    if inv.dataset_names:
        L.append("## Datasets")
        for d in inv.dataset_names:
            L.append(f"- `{d}/`")
        L.append("")
    if inv.episodes_by_task:
        L.append("## Tasks")
        L.append("")
        L.append("| Task | Episodes |")
        L.append("|---|---:|")
        for task, n in inv.episodes_by_task.items():
            L.append(f"| `{task}` | {n} |")
        L.append("")

    mods = meta.get("modalities", {})
    if mods:
        L.append("## Modalities")
        for key in ("video", "preprocess", "state", "action", "annotation"):
            vals = mods.get(key, [])
            if vals:
                L.append(f"- **{key}**: {', '.join(f'`{v}`' for v in vals)}")
        L.append("")

    conv = meta.get("conventions", {})
    if conv:
        L.append("## Conventions")
        for k in ("master_clock", "alignment_policy", "quaternion_order",
                  "length_unit", "angle_unit", "image_size",
                  "frame_index_basis", "image_normalization"):
            if k in conv:
                L.append(f"- **{k}**: `{conv[k]}`")
        L.append("")

    L.append("## Quick start")
    L.append("")
    L.append("```python")
    L.append("import dvrk_data_processing.surgsync as surgsync")
    L.append("")
    L.append(f"ds = surgsync.open_dataset({str(dataset_root)!r})")
    L.append("for ep_ref in ds.episodes:")
    L.append("    ep = surgsync.open_episode(ep_ref.path)")
    L.append("    print(ep.episode_id, ep.task, ep.length)")
    L.append("    psm1_pos = ep.psm1.column('measured_js.position')[0]")
    L.append("    phase    = ep.annotation.column('phase')[0]  # verbalized text")
    L.append("    for frame in ep.video_raw('stereo_left').iter_frames():")
    L.append("        ...  # (H, W, 3) uint8 BGR")
    L.append("        break")
    L.append("    ep.close()")
    L.append("```")
    L.append("")
    L.append("To get the raw pre-pack layout back on disk:")
    L.append("")
    L.append("```bash")
    L.append("surgsync unpack <this dataset root> --out <output dir>")
    L.append("```")
    L.append("")
    L.append("## Known limitations")
    L.append("")
    L.append("- CUDA non-determinism: depth/flow MKVs depend on the GPU/driver "
             "used during preprocessing. Round-trip pixel-matches within encoder tolerance, "
             "not byte-exact across machines.")
    L.append("- ECM Cartesian setpoint is dropped by the packer schema "
             "(only `setpoint_js` is carried). PSMs are unaffected.")
    if not inv.episodes_by_task:
        L.append("- No episodes finalized yet in this dataset root.")
    L.append("")
    L.append("## Layout")
    L.append("")
    L.append("```")
    L.append("<dataset_root>/")
    L.append("├── meta/")
    L.append("│   ├── dataset.json  tasks.jsonl  episodes.parquet")
    L.append("│   ├── index.parquet  stats.parquet  manifest.json")
    L.append("├── <dataset_name>/episodes/<task>/<clip_idx>/")
    L.append("│   ├── episode_meta.json  timestamp.parquet")
    L.append("│   ├── ECM.parquet  PSM1.parquet  PSM2.parquet  annotation.parquet")
    L.append("│   ├── video/        (H.264 rectified — when preprocessing ran)")
    L.append("│   ├── video_raw/    (FFV1 bit-exact — always)")
    L.append("│   ├── preprocess/   (FFV1 viz from preprocessing outputs)")
    L.append("│   └── calibration/")
    L.append("└── README.md  CHANGELOG.md")
    L.append("```")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CHANGELOG
# ---------------------------------------------------------------------------

_CHANGELOG_HEADER = (
    "# Changelog\n"
    "\n"
    "All notable changes to this dataset release are documented here.\n"
    "Each entry is appended at the top by `surgsync release`.\n"
    "\n"
)


def _render_changelog_entry(version: str, notes: Optional[str]) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = notes.strip() if notes else (
        "TODO: describe what changed in this release."
    )
    return "\n".join([f"## v{version} — {when}", "", body, ""])


def _append_changelog(path: Path, entry: str) -> None:
    """Insert `entry` after the header. Creates the file if absent. Atomic."""
    if path.is_file():
        existing = path.read_text()
        tail = existing[len(_CHANGELOG_HEADER):] if existing.startswith(_CHANGELOG_HEADER) else existing
    else:
        tail = ""
    new_body = _CHANGELOG_HEADER + entry + tail
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_body)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def write_release_docs(
    dataset_root: Path,
    *,
    new_version: Optional[str] = None,
    changelog_notes: Optional[str] = None,
) -> dict:
    """Emit README.md + CHANGELOG.md into the dataset root.

    `new_version` updates `meta/dataset.json:data_version` in place
    (atomic). `changelog_notes` is the entry body; pass None to leave
    a TODO placeholder.
    """
    dataset_root = Path(dataset_root)
    meta_path = dataset_root / "meta" / "dataset.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"cannot write release docs: {meta_path} missing")

    if new_version is not None:
        meta = json.loads(meta_path.read_text())
        meta["data_version"] = new_version
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(meta_path)

    inv = _scan_inventory(dataset_root)
    (dataset_root / "README.md").write_text(_render_readme(inv, dataset_root))

    cl_entry = _render_changelog_entry(
        version=str(inv.meta.get("data_version", "1.0")),
        notes=changelog_notes,
    )
    _append_changelog(dataset_root / "CHANGELOG.md", cl_entry)

    return {
        "readme_path":    str(dataset_root / "README.md"),
        "changelog_path": str(dataset_root / "CHANGELOG.md"),
        "n_episodes":     inv.episode_count,
        "n_tasks":        len(inv.episodes_by_task),
        "total_frames":   inv.total_frames,
        "data_version":   inv.meta.get("data_version"),
    }


def run_release(
    dataset_root: Path,
    *,
    bump: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """Read current version, optionally bump it, write docs."""
    dataset_root = Path(dataset_root)
    meta_path = dataset_root / "meta" / "dataset.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"cannot run release: {meta_path} missing")
    current = str(json.loads(meta_path.read_text()).get("data_version", "1.0"))
    new_version = bump_version(current, bump) if bump else None
    summary = write_release_docs(
        dataset_root,
        new_version=new_version,
        changelog_notes=notes,
    )
    summary["bumped_from"] = current if new_version else None
    summary["bumped_to"]   = new_version
    return summary
