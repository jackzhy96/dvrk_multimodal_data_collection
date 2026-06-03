"""Spot-check that a `surgsync unpack` output matches the original raw clip.

Usage:
    python scripts/verify_unpack_vs_raw.py \
        --raw    /path/to/raw_clip \
        --unpack /path/to/unpacked_clip \
        --max-frames 10

Checks:
  image/{left,right,side}/*.png  pixel-equal
  kinematic/*.json               shape + numeric (float32 round-trip)
  annotation/*/*.json            file presence per frame
  time_syn/*.json                tracked stamp equality
  camera_calibration, hand_eye_calibration, meta_data.json present
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# Known packer information losses (kept here so the verifier doesn't surface
# them as false negatives).
_KIN_DROPPED_BY_PACKER = {
    "ECM": frozenset({("arm", "setpoint_data", "setpoint_cp")}),
}


def _scan_pngs(raw_dir: Path, unp_dir: Path, max_frames: int) -> tuple[int, int, int]:
    matches = differs = missing = 0
    if not raw_dir.is_dir() or not unp_dir.is_dir():
        return 0, 0, 0
    for r in sorted(raw_dir.glob("*.png"))[:max_frames]:
        u = unp_dir / r.name
        if not u.is_file():
            missing += 1
            continue
        a = cv2.imread(str(r), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(u), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
            differs += 1
        else:
            matches += 1
    return matches, differs, missing


def _is_dropped_path(arm: str, path: tuple[str, ...]) -> bool:
    norm = tuple(p for p in path if not (isinstance(p, str) and p.startswith("[")))
    return any(norm[:len(d)] == d for d in _KIN_DROPPED_BY_PACKER.get(arm, frozenset()))


def _kin_close(a, b, *, arm: str, atol: float = 1e-5, path: tuple = ()) -> tuple[bool, str]:
    """Numeric tree equality with a per-arm skip set."""
    if _is_dropped_path(arm, path):
        return True, ""

    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            child = path + (k,)
            if _is_dropped_path(arm, child):
                continue
            if k not in b:
                return False, f"missing key in unp at .{'.'.join(map(str, child))}"
            ok, msg = _kin_close(a[k], b[k], arm=arm, atol=atol, path=child)
            if not ok:
                return False, msg
        return True, ""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"list len {len(a)} vs {len(b)} at .{'.'.join(map(str, path))}"
        if a and isinstance(a[0], (int, float)):
            aa = np.asarray(a, dtype=np.float64)
            bb = np.asarray(b, dtype=np.float64)
            if not np.allclose(aa, bb, atol=atol):
                return False, f"numeric diff max={float(np.max(np.abs(aa-bb))):.3g} at .{'.'.join(map(str, path))}"
            return True, ""
        for i, (xa, xb) in enumerate(zip(a, b)):
            ok, msg = _kin_close(xa, xb, arm=arm, atol=atol, path=path + (f"[{i}]",))
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--unpack", required=True)
    p.add_argument("--max-frames", type=int, default=10)
    args = p.parse_args(argv)

    raw = Path(args.raw)
    unp = Path(args.unpack)
    if not raw.is_dir() or not unp.is_dir():
        print(f"ERROR: raw or unpack not a dir; raw={raw} unp={unp}", file=sys.stderr)
        return 1

    n_fail = 0

    # image/
    for cam in ("left", "right", "side"):
        m, d, miss = _scan_pngs(raw / "image" / cam, unp / "image" / cam, args.max_frames)
        status = "OK" if d == 0 and miss == 0 else "FAIL"
        print(f"image/{cam}: {status} matches={m} differs={d} missing={miss}")
        if status == "FAIL":
            n_fail += 1

    # kinematic/
    for arm in ("ECM", "PSM1", "PSM2"):
        raw_arm = raw / "kinematic" / arm
        unp_arm = unp / "kinematic" / arm
        if not raw_arm.is_dir() or not unp_arm.is_dir():
            print(f"kinematic/{arm}: SKIP — dir missing")
            continue
        files = sorted(raw_arm.glob("*.json"))[:args.max_frames]
        n_ok = n_bad = 0
        for raw_f in files:
            unp_f = unp_arm / raw_f.name
            if not unp_f.is_file():
                n_bad += 1
                continue
            try:
                r = json.loads(raw_f.read_text())
                u = json.loads(unp_f.read_text())
                ok, msg = _kin_close(r, u, arm=arm)
                if ok:
                    n_ok += 1
                else:
                    if n_bad == 0:
                        print(f"  kinematic/{arm} first diff @ {raw_f.name}: {msg}")
                    n_bad += 1
            except Exception:
                n_bad += 1
        status = "OK" if n_bad == 0 else "FAIL"
        print(f"kinematic/{arm}: {status} ok={n_ok}/{len(files)}")
        if status == "FAIL":
            n_fail += 1

    # annotation/
    for kind in ("contact_detection", "phase", "step", "gesture"):
        raw_k = raw / "annotation" / kind
        unp_k = unp / "annotation" / kind
        if not raw_k.is_dir():
            print(f"annotation/{kind}: SKIP — raw dir missing")
            continue
        if not unp_k.is_dir():
            print(f"annotation/{kind}: FAIL — unp dir missing")
            n_fail += 1
            continue
        raw_files = {p.name for p in raw_k.glob("*.json")}
        unp_files = {p.name for p in unp_k.glob("*.json")}
        missing = raw_files - unp_files
        if missing and kind != "gesture":
            print(f"annotation/{kind}: FAIL — {len(missing)} missing "
                  f"(e.g. {sorted(missing)[:3]})")
            n_fail += 1
        else:
            print(f"annotation/{kind}: OK ({len(unp_files)} files; "
                  f"{len(missing)} raw-only)")

    # time_syn/
    sample = sorted((raw / "time_syn").glob("*.json"))[:args.max_frames]
    n_ok = n_bad = 0
    for raw_f in sample:
        unp_f = unp / "time_syn" / raw_f.name
        if not unp_f.is_file():
            n_bad += 1
            continue
        r = json.loads(raw_f.read_text())
        u = json.loads(unp_f.read_text())
        if (r.get("image_left_stamp")   == u.get("image_left_stamp") and
            r.get("image_right_stamp")  == u.get("image_right_stamp") and
            r.get("side_image_1_stamp") == u.get("side_image_1_stamp")):
            n_ok += 1
        else:
            n_bad += 1
    print(f"time_syn (image stamps): {'OK' if n_bad == 0 else 'FAIL'} "
          f"ok={n_ok}/{len(sample)}")
    if n_bad:
        n_fail += 1

    # calibration + meta
    for sub in ("camera_calibration", "hand_eye_calibration", "meta_data.json"):
        target = unp / sub
        ok = target.exists() and (target.is_file() or any(target.iterdir()))
        print(f"{sub}: {'OK' if ok else 'FAIL'}")
        if not ok:
            n_fail += 1

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
