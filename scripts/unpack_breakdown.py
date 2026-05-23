"""Runtime + size breakdown for a `surgsync unpack` output.

Usage:
    python scripts/unpack_breakdown.py \
        --packed   /path/to/release \
        --unpacked /path/to/unpack \
        --raw      online_data=/path/to/icra/raw \
        --raw      offline_data=/path/to/open_h/raw \
        --log      /path/to/unpack.log
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def _fmt(n: int) -> str:
    if n >= 1 << 30: return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20: return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10: return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(c) for c in [h] + [r[i] for r in rows]) for i, h in enumerate(header)]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print(sep)
    for r in rows:
        cells = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(r))]
        print("  ".join(cells))


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_STREAM_LINE = re.compile(
    r"\|\s*(?P<stream>[\w./_-]+):\s+(?P<count>\d+)\s+(PNGs|JSONs)"
    r"\s+in\s+(?P<elapsed>[\d.]+)s(?:\s+\(([\d.]+)\s*fps\))?"
)


def _parse_log(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.is_file():
        return rows
    for line in log_path.read_text().splitlines():
        m = _STREAM_LINE.search(line)
        if not m:
            continue
        rows.append({
            "stream":    m.group("stream"),
            "count":     int(m.group("count")),
            "elapsed_s": float(m.group("elapsed")),
            "fps":       float(m.group(5)) if m.group(5) else None,
        })
    return rows


# Buckets used for the per-clip layout breakdown.
_BUCKETS = [
    ("image",                "image"),
    ("kinematic",            "kinematic"),
    ("annotation",           "annotation"),
    ("time_syn",             "time_syn"),
    ("camera_calibration",   "camera_calibration"),
    ("hand_eye_calibration", "hand_eye_calibration"),
    ("meta_data.json",       "meta_data.json"),
    ("preprocess/rectify_resize",      "preprocess/rectify_resize"),
    ("preprocess/depth_estimation",    "preprocess/depth_estimation"),
    ("preprocess/optical_flow",        "preprocess/optical_flow"),
    ("preprocess/kinematic_reproject", "preprocess/kinematic_reproject"),
]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packed",   required=True)
    p.add_argument("--unpacked", required=True)
    p.add_argument("--raw", action="append", default=[],
                   help="Raw root. Form: `<path>` (any dataset) or "
                        "`<dataset>=<path>` (scoped). Repeatable.")
    p.add_argument("--log", default=None,
                   help="Optional unpack log file to extract per-stream timings.")
    args = p.parse_args(argv)

    packed   = Path(args.packed)
    unpacked = Path(args.unpacked)

    scoped_raw: dict[str, Path] = {}
    unscoped_raw: list[Path] = []
    for entry in args.raw:
        if "=" in entry:
            ds, _, path_str = entry.partition("=")
            scoped_raw[ds.strip()] = Path(path_str)
        else:
            unscoped_raw.append(Path(entry))

    report_path = unpacked / "decompose_report.json"
    if not report_path.is_file():
        sys.exit(f"decompose_report.json not found at {report_path}")
    report = json.loads(report_path.read_text())

    # ---- 1. Runtime --------------------------------------------------
    from datetime import datetime
    t0 = datetime.fromisoformat(report["started_at_utc"])
    t1 = datetime.fromisoformat(report["finished_at_utc"])
    wall = (t1 - t0).total_seconds()
    sum_elapsed = sum(c["elapsed_s"] for c in report["clips"])

    print("=" * 78)
    print("RUNTIME — surgsync unpack")
    print("=" * 78)
    print(f"  Started:           {report['started_at_utc']}")
    print(f"  Finished:          {report['finished_at_utc']}")
    print(f"  Wall-clock total:  {wall:7.1f} s  ({wall/60:.1f} min)")
    print(f"  Σ per-clip elapsed:{sum_elapsed:7.1f} s  ({sum_elapsed/60:.1f} min)  "
          f"[serial-equivalent]")
    if wall > 0:
        print(f"  Effective parallelism: {sum_elapsed/wall:.2f}×")
    print(f"  Episodes:          ok={report['n_episodes_ok']}  "
          f"fail={report['n_episodes_fail']}  "
          f"skipped={report.get('n_episodes_skipped', 0)}")
    print()
    rows = [[
        f"{c['dataset_name']}/{c['clip_index']}",
        c["task"],
        str(c["n_frames"]),
        f"{c['elapsed_s']:.1f}",
    ] for c in report["clips"]]
    _print_table(["clip", "task", "frames", "elapsed_s"], rows)
    print()

    # ---- 2. Per-stream timings ---------------------------------------
    if args.log:
        stream_rows = _parse_log(Path(args.log))
        if stream_rows:
            print("=" * 78)
            print("PER-STREAM TIMINGS  (aggregated across all clips)")
            print("=" * 78)
            agg: dict[str, dict] = {}
            for r in stream_rows:
                a = agg.setdefault(r["stream"], {"count": 0, "elapsed_s": 0.0,
                                                  "n": 0, "fps_sum": 0.0, "fps_n": 0})
                a["count"]     += r["count"]
                a["elapsed_s"] += r["elapsed_s"]
                a["n"]         += 1
                if r["fps"] is not None:
                    a["fps_sum"] += r["fps"]
                    a["fps_n"]   += 1
            rows = []
            for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["elapsed_s"]):
                avg_fps = (a["fps_sum"] / a["fps_n"]) if a["fps_n"] else 0.0
                rows.append([
                    name, str(a["n"]), f"{a['count']}", f"{a['elapsed_s']:.1f}",
                    f"{avg_fps:.1f}" if a["fps_n"] else "—",
                ])
            _print_table(["stream", "n_clips", "Σ items", "Σ elapsed_s", "avg fps"], rows)
            print()

    # ---- 3. Per-clip size --------------------------------------------
    print("=" * 78)
    print("SIZE — packed vs unpacked (and raw, if given)")
    print("=" * 78)
    rows = []
    totals = {"packed": 0, "unpacked": 0, "raw": 0}
    for c in report["clips"]:
        ds, ci = c["dataset_name"], c["clip_index"]
        packed_dir   = packed / ds / "episodes" / c["task"] / ci
        unpacked_dir = unpacked / ds / ci

        cand_paths: list[Path] = []
        if ds in scoped_raw:
            cand_paths.append(scoped_raw[ds] / ci)
        cand_paths.extend(rr / ci for rr in unscoped_raw)

        raw_size = 0
        raw_path: Optional[Path] = None
        for cand in cand_paths:
            if cand.is_dir():
                raw_path = cand
                raw_size = _dir_size(cand)
                break

        pk = _dir_size(packed_dir)
        un = _dir_size(unpacked_dir)
        totals["packed"]   += pk
        totals["unpacked"] += un
        totals["raw"]      += raw_size
        rows.append([
            f"{ds}/{ci}",
            _fmt(pk),
            _fmt(un),
            f"{(un/pk):.2f}×" if pk else "0×",
            _fmt(raw_size) if raw_path else "—",
            f"{(un/raw_size):.2f}×" if raw_size else "—",
        ])
    rows.append([
        "TOTAL",
        _fmt(totals["packed"]),
        _fmt(totals["unpacked"]),
        f"{totals['unpacked']/totals['packed']:.2f}×" if totals["packed"] else "—",
        _fmt(totals["raw"]) if totals["raw"] else "—",
        f"{totals['unpacked']/totals['raw']:.2f}×" if totals["raw"] else "—",
    ])
    _print_table(["clip", "packed", "unpacked", "unp/pk", "raw", "unp/raw"], rows)
    print()

    # ---- 4. Per-bucket breakdown -------------------------------------
    print("=" * 78)
    print("UNPACKED PER-BUCKET BREAKDOWN  (Σ across all clips)")
    print("=" * 78)
    bucket_totals: dict[str, int] = {name: 0 for name, _ in _BUCKETS}
    for c in report["clips"]:
        clip_dir = unpacked / c["dataset_name"] / c["clip_index"]
        for name, rel in _BUCKETS:
            bucket_totals[name] += _dir_size(clip_dir / rel)
    grand = sum(bucket_totals.values()) or 1
    rows = [
        [name, _fmt(sz), f"{100*sz/grand:.1f}%"]
        for name, sz in sorted(bucket_totals.items(), key=lambda kv: -kv[1])
    ]
    rows.append(["TOTAL", _fmt(grand), "100.0%"])
    _print_table(["bucket", "size", "share"], rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
