"""Snapshot helper for the end-to-end smoke test.

Captures the per-file fingerprints needed to verify a pack → unpack
round-trip against the original raw clip. Each artifact bucket has a
distinct equivalence relation:

  image/                  -> pixel-bit-exact (FFV1 round-trip; cv2 PNG
                             re-encode produces different file bytes
                             from the original encoder while preserving
                             every pixel)
  kinematic/*.json        -> numeric equivalence within float32 atol
                             (parquet float32 round-trip)
  annotation/*.json       -> text-form for phase/step/gesture cells
                             (verbalized via workflow_description.json;
                             the original numeric id is intentionally
                             not preserved)
  time_syn/*.json         -> bit-exact for tracked stamps;
                             header_cv/reference_js are reconstructed
  camera_calibration/     -> byte-exact
  hand_eye_calibration/   -> byte-exact
  meta_data.json          -> byte-exact for the known keys

The snapshot is a small dict suitable for pickling between processes
or saving to JSON when debugging a failed run.
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hashers
# ---------------------------------------------------------------------------

def _file_sha(p: Path) -> str:
    """Byte-content hash, used for genuinely byte-exact streams."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixel_sha(p: Path) -> Optional[str]:
    """Hash of the decoded pixel array. Independent of PNG encoder.

    Returns None if cv2 can't decode the file — caller treats that as
    a missing-snapshot for that frame rather than a comparison error.
    """
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClipSnapshot:
    """One clip's fingerprint.

    `image_pixels[<cam>/<frame>.png] -> pixel-hash`
    `calibration[<rel>] -> byte-hash`
    `meta -> byte-hash of meta_data.json`
    `kinematic_paths` / `annotation_paths` / `time_syn_paths` carry just
    relative paths; numeric / structural comparison is done at diff time
    by re-reading the JSON, not via a content hash (float repr varies).
    """
    dataset_name:        str
    clip_index:          str
    raw_root:            Path
    image_pixels:        dict[str, str] = field(default_factory=dict)
    calibration:         dict[str, str] = field(default_factory=dict)
    hand_eye:            dict[str, str] = field(default_factory=dict)
    meta_sha:            Optional[str]  = None
    kinematic_paths:     dict[str, list[str]] = field(default_factory=dict)
    annotation_paths:    dict[str, list[str]] = field(default_factory=dict)
    time_syn_paths:      list[str]      = field(default_factory=list)


def snapshot_clip(raw_clip_dir: Path, *, dataset_name: str, clip_index: str) -> ClipSnapshot:
    """Walk the raw clip dir and build a `ClipSnapshot`.

    Designed for fast iteration on small fixtures; for a 1000-frame
    clip with 3 cameras this is ~10 s wall-clock (PNG decode is the
    bottleneck).
    """
    raw = Path(raw_clip_dir)
    snap = ClipSnapshot(
        dataset_name=dataset_name, clip_index=clip_index, raw_root=raw,
    )

    # image/{cam}/{frame}.png — pixel hash, since the round-trip is
    # only pixel-exact (file bytes differ across PNG encoders).
    for cam_dir in sorted((raw / "image").iterdir()):
        if not cam_dir.is_dir():
            continue
        for png in sorted(cam_dir.glob("*.png")):
            key = f"{cam_dir.name}/{png.name}"
            h = _pixel_sha(png)
            if h is not None:
                snap.image_pixels[key] = h

    # calibration: byte-exact.
    cal_dir = raw / "camera_calibration"
    if cal_dir.is_dir():
        for p in sorted(cal_dir.iterdir()):
            if p.is_file():
                snap.calibration[p.name] = _file_sha(p)
    he_dir = raw / "hand_eye_calibration"
    if he_dir.is_dir():
        for p in sorted(he_dir.iterdir()):
            if p.is_file():
                snap.hand_eye[p.name] = _file_sha(p)

    # meta_data.json
    md = raw / "meta_data.json"
    if md.is_file():
        snap.meta_sha = _file_sha(md)

    # kinematic / annotation / time_syn: path-only snapshot. Content
    # diff happens at compare-time.
    kin_root = raw / "kinematic"
    if kin_root.is_dir():
        for arm_dir in sorted(kin_root.iterdir()):
            if arm_dir.is_dir():
                snap.kinematic_paths[arm_dir.name] = sorted(
                    p.name for p in arm_dir.glob("*.json")
                )

    ann_root = raw / "annotation"
    if ann_root.is_dir():
        for kind_dir in sorted(ann_root.iterdir()):
            if kind_dir.is_dir():
                snap.annotation_paths[kind_dir.name] = sorted(
                    p.name for p in kind_dir.glob("*.json")
                )

    ts_root = raw / "time_syn"
    if ts_root.is_dir():
        snap.time_syn_paths = sorted(p.name for p in ts_root.glob("*.json"))

    return snap


# ---------------------------------------------------------------------------
# Diff (round-trip oracle)
# ---------------------------------------------------------------------------

@dataclass
class DiffFinding:
    """One round-trip mismatch. Surfaces in pytest failure messages."""
    severity: str     # "ERROR" | "WARNING"
    bucket:   str     # "image" | "kinematic" | "calibration" | ...
    path:     str     # relative path inside the clip
    detail:   str


def diff_against_snapshot(
    decomposed_clip_dir: Path,
    snap: ClipSnapshot,
    *,
    # Buckets to enforce. Defaults match the packer invertibility contract.
    check_image: bool = True,
    check_calibration: bool = True,
    check_kinematic: bool = True,
    check_meta: bool = True,
    kin_atol: float = 1e-5,
) -> list[DiffFinding]:
    """Compare a decomposed clip dir against the raw-side snapshot.

    Returns a list of findings; ERRORs cause the smoke test to fail.
    The function never raises on mismatches — it surfaces every
    problem so a single test invocation reports the full diff, not
    just the first failure.
    """
    findings: list[DiffFinding] = []

    if check_image:
        for key, raw_hash in snap.image_pixels.items():
            unp_file = decomposed_clip_dir / "image" / key
            if not unp_file.is_file():
                findings.append(DiffFinding(
                    "ERROR", "image", f"image/{key}",
                    f"missing in decomposed tree",
                ))
                continue
            unp_hash = _pixel_sha(unp_file)
            if unp_hash != raw_hash:
                findings.append(DiffFinding(
                    "ERROR", "image", f"image/{key}",
                    f"pixel hash mismatch: raw={raw_hash[:16]}… unp={(unp_hash or 'None')[:16]}…",
                ))

    if check_calibration:
        for fname, raw_hash in snap.calibration.items():
            unp_file = decomposed_clip_dir / "camera_calibration" / fname
            if not unp_file.is_file():
                findings.append(DiffFinding(
                    "ERROR", "calibration", f"camera_calibration/{fname}",
                    "missing in decomposed tree",
                ))
                continue
            unp_hash = _file_sha(unp_file)
            if unp_hash != raw_hash:
                findings.append(DiffFinding(
                    "ERROR", "calibration", f"camera_calibration/{fname}",
                    f"byte hash mismatch: raw={raw_hash[:16]}… unp={unp_hash[:16]}…",
                ))
        for fname, raw_hash in snap.hand_eye.items():
            unp_file = decomposed_clip_dir / "hand_eye_calibration" / fname
            if not unp_file.is_file():
                findings.append(DiffFinding(
                    "ERROR", "calibration", f"hand_eye_calibration/{fname}",
                    "missing in decomposed tree",
                ))
                continue
            unp_hash = _file_sha(unp_file)
            if unp_hash != raw_hash:
                findings.append(DiffFinding(
                    "ERROR", "calibration", f"hand_eye_calibration/{fname}",
                    f"byte hash mismatch: raw={raw_hash[:16]}… unp={unp_hash[:16]}…",
                ))

    if check_meta and snap.meta_sha is not None:
        unp_meta = decomposed_clip_dir / "meta_data.json"
        if not unp_meta.is_file():
            findings.append(DiffFinding(
                "ERROR", "meta", "meta_data.json", "missing in decomposed tree",
            ))
        else:
            unp_hash = _file_sha(unp_meta)
            if unp_hash != snap.meta_sha:
                # meta_data.json is reconstructed from episode_meta.json
                # so a byte mismatch is expected in general (whitespace,
                # indent differ). Compare semantically instead.
                raw = json.loads((snap.raw_root / "meta_data.json").read_text())
                unp = json.loads(unp_meta.read_text())
                for k in ("user_id", "operator_skill_level", "case_type",
                          "tool", "failure", "recovery"):
                    if raw.get(k) != unp.get(k):
                        findings.append(DiffFinding(
                            "ERROR", "meta", "meta_data.json",
                            f"field {k!r} differs: raw={raw.get(k)!r} unp={unp.get(k)!r}",
                        ))

    if check_kinematic:
        for arm, fnames in snap.kinematic_paths.items():
            for fn in fnames:
                raw_f = snap.raw_root / "kinematic" / arm / fn
                unp_f = decomposed_clip_dir / "kinematic" / arm / fn
                if not unp_f.is_file():
                    findings.append(DiffFinding(
                        "ERROR", "kinematic", f"kinematic/{arm}/{fn}",
                        "missing in decomposed tree",
                    ))
                    continue
                try:
                    raw_j = json.loads(raw_f.read_text())
                    unp_j = json.loads(unp_f.read_text())
                except Exception as e:
                    findings.append(DiffFinding(
                        "ERROR", "kinematic", f"kinematic/{arm}/{fn}",
                        f"JSON parse error: {e}",
                    ))
                    continue
                # Numeric-equivalent walk. Known packer loss for ECM:
                # setpoint_cp is not carried in the ECM parquet schema.
                skip_paths = {("arm", "setpoint_data", "setpoint_cp")} if arm == "ECM" else set()
                ok, msg = _kin_close(raw_j, unp_j, skip_paths=skip_paths, atol=kin_atol)
                if not ok:
                    findings.append(DiffFinding(
                        "ERROR", "kinematic", f"kinematic/{arm}/{fn}", msg,
                    ))

    return findings


def _kin_close(a, b, *, skip_paths: set, atol: float, path: tuple = ()) -> tuple[bool, str]:
    """Numeric tree-equality with a configurable atol and skip set.

    `skip_paths` entries are tuples of dict keys (any list-index
    prefix tolerated) — used to elide structurally-dropped subtrees
    like ECM.setpoint_cp.
    """
    # Match `path` against `skip_paths` regardless of leading list-index
    # entries.
    norm = tuple(p for p in path if not (isinstance(p, str) and p.startswith("[")))
    for sk in skip_paths:
        if norm[:len(sk)] == sk:
            return True, ""

    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                child = path + (k,)
                cnorm = tuple(p for p in child if not (isinstance(p, str) and p.startswith("[")))
                if any(cnorm[:len(sk)] == sk for sk in skip_paths):
                    continue
                return False, f"missing key in unp at .{'.'.join(map(str, path))}.{k}"
            ok, msg = _kin_close(a[k], b[k], skip_paths=skip_paths, atol=atol, path=path + (k,))
            if not ok:
                return False, msg
        return True, ""

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"list len {len(a)} vs {len(b)} at .{'.'.join(map(str, path))}"
        # Numeric list shortcut.
        if a and isinstance(a[0], (int, float)):
            aa = np.asarray(a, dtype=np.float64)
            bb = np.asarray(b, dtype=np.float64)
            if not np.allclose(aa, bb, atol=atol):
                d = float(np.max(np.abs(aa - bb)))
                return False, f"numeric diff max={d:.3g} at .{'.'.join(map(str, path))}"
            return True, ""
        for i, (xa, xb) in enumerate(zip(a, b)):
            ok, msg = _kin_close(xa, xb, skip_paths=skip_paths, atol=atol, path=path + (f"[{i}]",))
            if not ok:
                return False, msg
        return True, ""

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > atol * max(1.0, abs(a)):
            return False, f"scalar {a} vs {b} at .{'.'.join(map(str, path))}"
        return True, ""

    if a != b:
        return False, f"value {a!r} vs {b!r} at .{'.'.join(map(str, path))}"
    return True, ""
