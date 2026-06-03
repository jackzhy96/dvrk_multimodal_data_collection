# How to run the SurgSync unpacker

This is the operator guide for `surgsync unpack` — the decomposer
that takes a packed SurgSync dataset (the output of `surgsync build`)
and writes a directory tree that mirrors the **pre-pack raw layout**
under `data/`.

`unpack` is the literal inverse of `build`. See `HOW_to_RUN_pack.md` for
the forward direction.

---

## Prerequisites

You must already have:

1. **Conda env** built and active:
   ```bash
   conda activate dvrk_multimodal_process
   pip install -e .
   ```

2. **A packed SurgSync dataset** on disk — e.g.
   `<release_root>` for the sample data,
   or any other release built with `surgsync build`.

3. **Disk space**: unpacking restores the raw PNGs (bit-exact via FFV1)
   plus the preprocessing PNGs, so the unpacked tree is roughly the
   same size as the original `data/<dataset>/<clip>/` plus its
   `preprocess/`. As a reference: `online_data/2` (886 frames) takes
   ~12 GB when fully unpacked (3 cameras × bit-exact PNGs dominate).
   The sample release of 2 clips unpacks to ~57 GB.

4. **ffmpeg / ffprobe** from the conda env (the unpacker shells out to
   them via the same `encode/codec.py` module the packer uses).

---

## One-liner: unpack the sample release

```bash
conda activate dvrk_multimodal_process

surgsync unpack \
    "<release_root>" \
    --out "<unpack_root>"
```

What happens:

1. The dataset root is walked. Only episodes with a
   `.surgsync_complete.json` sentinel are considered shippable.
2. For each selected episode, the unpacker writes
   `<out_root>/<dataset_name>/<clip_index>/` containing every raw
   artifact the packer originally consumed:
   - `image/{left,right,side}/<i>.png` — bit-exact from `video_raw/*.mkv`.
   - `kinematic/{ECM,PSM1,PSM2}/<i>.json` — from the per-arm parquets.
   - `annotation/{contact_detection,phase,step,gesture}/<i>.json` —
     phase / step / gesture cells carry the **text description**
     (verbalized via `workflow_description.json`), not the original
     numeric id.
   - `time_syn/<i>.json` — reconstructed from
     `master_t0_ns + master_timestamp_ns + delta_to_master.<topic>_ns`.
   - `camera_calibration/`, `hand_eye_calibration/`, `meta_data.json`
     — copied verbatim from the packed `calibration/` + `episode_meta.json`.
   - `preprocess/rectify_resize/image/{left,right}/<i>.png` —
     visually-lossless (≥ 40 dB PSNR) from `video/*.mp4`.
   - `preprocess/rectify_resize/{camera_calibration,kinematic,time_syn}/`
     — copies of the relevant raw artifacts (matches the
     preprocessing stage's behavior of copying these through).
   - `preprocess/depth_estimation/depth_image/<i>.png` — bit-exact
     from `preprocess/depth.mkv`.
   - `preprocess/optical_flow/{left,right}/image/<i>.png` — bit-exact
     from `preprocess/flow_*.mkv`.
   - `preprocess/kinematic_reproject/{PSM1,PSM2}/{left,right}/image/<i>.png`
     — bit-exact from `preprocess/heatmap_*.mkv`.
   - `preprocess/kinematic_reproject/{PSM1,PSM2}/calibrated_kinematic/<i>.json`
     — from the per-arm parquet's `*_cp_calibrated.*` columns (only
     present when the hand-eye preprocessing stage ran for the
     original clip).
3. A `decompose_report.json` lands at the out_root with per-clip stats
   (frame counts, elapsed time, error string if any) and a fidelity
   tag for every artifact bucket.

### Runtime

Measured on an 8-core dev box with both packed and unpacked tree on
`<external drive>`. Sample release is 2 clips (886 + 947
frames, 3 cameras each, 1920×1080 raw):

| Configuration | Per-clip wall-clock | Sum-of-clips (serial-equiv) | Wall-clock end-to-end | Speedup |
|---|---:|---:|---:|---:|
| `--parallelism 2 --workers-per-clip 4`, full (raw + preprocess) | **150 s** | 300 s | **152 s** | 1.98× |

Per-stream throughput (measured, logged at INFO):

| Stream | Throughput |
|---|---:|
| `image/{left,right,side}` (FFV1 1080p → PNG bgr24) | ~20 fps |
| `preprocess/rectify_resize/image/*` (H.264 512×288 → PNG) | ~270 fps |
| `preprocess/depth_estimation/depth_image` (FFV1 → PNG) | ~700 fps |
| `preprocess/optical_flow/<cam>/image` (FFV1 → PNG) | ~730 fps |
| `preprocess/kinematic_reproject/<arm>/<cam>/image` (FFV1 gray8 → PNG) | ~1300 fps |
| `kinematic/<arm>/<src_idx>.json` (parquet batch → JSON) | ~2000 files/s |

Bottleneck: cv2 PNG-encode of the bit-exact `image/` frames at 1080p
(~70% of total wall-clock). FFV1 decode is fast; ffmpeg subprocess
startup is negligible. RAM is bounded by
`workers-per-clip × queue_depth_factor` (currently 4 × 4 = 16 frames
in flight ≈ 100 MB per stream regardless of clip length).

Tuning recommendations:
- **Single-clip latency**: keep `--parallelism 1`, bump
  `--workers-per-clip` to `(physical cores - 1)`.
- **Throughput on a sweep**: set `--parallelism` to roughly
  `(physical cores / 3)`. Each clip worker uses ~3 threads internally
  (1 ffmpeg + N png-writers + bookkeeping); going higher
  oversubscribes ffmpeg subprocesses and loses to SSD contention.
- For 10k-clip sweeps on an NVMe SSD, `--parallelism 6 --workers-per-clip 4`
  on a 24-core box is a good starting point. Watch `iostat -x 1` —
  if disk `%util` is pegged at 100, lower `--parallelism`.

### Resume / skip-if-done

`surgsync unpack` writes `.surgsync_unpacked.json` into every
successfully-finished clip dir. Re-running the same command **without
`--force`** skips clips whose sentinel is present (no decode, no
write, ~0 ms per clip). This is what makes long sweeps interruptible:

```bash
# First pass: unpacks every clip it can; killed mid-sweep is fine.
surgsync unpack /path/to/release --out /path/to/unpack --parallelism 4

# Second pass: skips everything done; finishes what was left.
surgsync unpack /path/to/release --out /path/to/unpack --parallelism 4

# Force re-do everything (wipes + re-writes):
surgsync unpack /path/to/release --out /path/to/unpack --parallelism 4 --force
```

The orchestrator also detects when two packed episodes share the
same `(dataset_name, clip_index)` across different task folders —
which would silently overwrite in the unpacked tree — and aborts
with an actionable error before touching the output. Use `--task` to
disambiguate, or re-pack with consistent task labels.

---

## Filtering: unpack only what you need

| Flag | Effect |
|---|---|
| `--clip online_data/2` | Only the named clip. Repeatable. |
| `--dataset-name online_data` | Only episodes from one top-level partition. Repeatable. |
| `--task single_interrupted_stitch` | Only one task. Repeatable. |
| `--episode-id online_data_2_1754609707325800839` | Match the deterministic episode id. Repeatable. |
| `--streams raw` | Skip the `preprocess/` tree (raw clip only — fastest). |
| `--streams preprocess` | Skip the raw clip body (only emit the preprocess tree). |
| `--force` | Wipe + re-write already-populated output clip dirs. |
| `--parallelism N` | Unpack N clips concurrently via ProcessPoolExecutor. |
| `--workers-per-clip K` | Thread-pool size for the PNG writer inside one clip. |

Filters AND together.

Examples:

```bash
# Only the online suturing clip; skip preprocess; overwrite if already there.
surgsync unpack <release_root> \
    --out <unpack_root> \
    --clip online_data/2 \
    --streams raw \
    --force

# Whole dataset, 4 clips in parallel, 4 PNG threads each.
surgsync unpack <release_root> \
    --out <unpack_root> \
    --parallelism 4 \
    --workers-per-clip 4
```

---

## Output layout

A successful unpack writes the following tree under `<out_root>`:

```
<out_root>/
├── decompose_report.json
├── offline_data/
│   └── 3/
│       └── ...
└── online_data/
    └── 2/
        ├── image/
        │   ├── left/0.png      ← bit-exact (FFV1)
        │   ├── right/0.png
        │   └── side/0.png      ← bit-exact, when packed
        ├── kinematic/
        │   ├── ECM/0.json
        │   ├── PSM1/0.json
        │   └── PSM2/0.json
        ├── annotation/
        │   ├── contact_detection/0.json    # {"PSM1": 0, "PSM2": 0}
        │   ├── phase/0.json                # {"phase": "<text>"}
        │   ├── step/0.json                 # {"step": "<text>"}
        │   └── gesture/0.json              # {"gesture": {"PSM1": "<text>", "PSM2": "<text>"}}
        ├── time_syn/0.json
        ├── camera_calibration/
        │   ├── left.yaml
        │   ├── right.yaml
        │   └── stereo_calib_params.json
        ├── hand_eye_calibration/
        │   ├── PSM1-registration-dVRK.json
        │   ├── PSM1-registration-open-cv.json
        │   ├── PSM2-registration-dVRK.json
        │   └── PSM2-registration-open-cv.json
        ├── meta_data.json
        └── preprocess/
            ├── rectify_resize/
            │   ├── image/{left,right}/0.png        ← visually lossless (H.264 CRF 18)
            │   ├── camera_calibration/             ← copied
            │   ├── kinematic/{ECM,PSM1,PSM2}/0.json
            │   └── time_syn/0.json
            ├── depth_estimation/
            │   └── depth_image/0.png               ← bit-exact (FFV1)
            ├── optical_flow/
            │   ├── left/image/0.png                ← bit-exact
            │   └── right/image/0.png               ← bit-exact
            └── kinematic_reproject/
                ├── PSM1/
                │   ├── left/image/0.png            ← bit-exact (gray8)
                │   ├── right/image/0.png
                │   └── calibrated_kinematic/0.json
                └── PSM2/
                    ├── left/image/0.png
                    ├── right/image/0.png
                    └── calibrated_kinematic/0.json
```

**Filenames**: every per-frame file uses the **source frame index**
from the original raw clip (carried through the pack via
`timestamp.parquet.source_frame_index`). Sequential
when the source was contiguous; sparse when the packer compacted
dropped frames.

---

## What is and isn't recoverable

| Artifact | Fidelity | Why |
|---|---|---|
| `image/` PNGs | pixel bit-exact | FFV1 → bgr24 PNG round-trip preserves every pixel. File-byte hashes will not match because cv2's PNG encoder picks different compression filters than whatever produced the original; decode-and-compare yields identical arrays. |
| `kinematic/*.json` | float-equivalent | Packed as Arrow `list<float32>`; JSON re-emits the same bits. |
| `annotation/*.json` | text form | Packer verbalized phase/step/gesture ids via `workflow_description.json`; unpack carries the same text. The original numeric id is **not** preserved by design. |
| `time_syn/` `image_left_stamp` + tracked topics | reconstructed | `master_t0_ns + master_ns + delta_to_master.<topic>_ns`. Bit-exact for tracked topics. |
| `time_syn/` `header_cv_stamp` / `reference_js_stamp` | reconstructed | Mirrored from the nearest tracked topic (the raw data shows they coincide; the packer drops them on the floor). |
| `camera_calibration/`, `hand_eye_calibration/` | bit-exact | Byte-copied from the packed `calibration/` folder. |
| `meta_data.json` | bit-exact for known keys | Reconstructed from `episode_meta.json` per `serde/meta_io.py`. |
| `preprocess/rectify_resize/image/` PNGs | visually lossless | H.264 CRF 18 — declared PSNR floor ≥ 40 dB. |
| `preprocess/depth_estimation/depth_image/*.png` | bit-exact | FFV1 bgr24 round-trip. |
| `preprocess/optical_flow/<cam>/image/*.png` | bit-exact | FFV1 bgr24 round-trip. |
| `preprocess/kinematic_reproject/<arm>/<cam>/image/*.png` | bit-exact | FFV1 gray8 round-trip. |
| `preprocess/.../calibrated_kinematic/*.json` | float-equivalent | From per-arm parquet's `*_cp_calibrated` columns. |
| Raw `.npy` derivatives (`depth_estimation/{depth,disparity,disparity_image,combined_image}`, `optical_flow/<cam>/optical_flow/`, `kinematic_reproject/<arm>/<cam>/heatmap/`) | **dropped** | The packer never stored the raw float arrays — derived modalities are shipped as the 8-bit visualization PNG only. Re-run the preprocessing pipeline (`scripts/run_all_stages.py`) from the unpacked `image/` folder to regenerate the numerical derivatives. |
| `preprocess/kinematic_reproject_drawframe/*` | **dropped** | The packer doesn't store the drawframe overlays. Re-run the preprocessing pipeline with the drawframe config enabled to regenerate. |

---

## Verifying an unpack

The per-run report carries everything:

```bash
jq . "<unpack_root>/decompose_report.json" | less
```

To spot-check that PNGs round-trip pixel-exact for a clip you also
have in raw form, use the bundled verifier script:

```bash
python scripts/verify_unpack_vs_raw.py \
    --raw    data/online_data/2 \
    --unpack "<unpack_root>/online_data/2" \
    --max-frames 10
```

It compares pixels (not file bytes) for image/, validates kinematic
JSON shape + numeric values, and confirms time_syn stamps match.

For a quick visual sanity check on the verbalized annotations:

```bash
jq . "<unpack_root>/online_data/2/annotation/phase/0.json"
jq . "<unpack_root>/online_data/2/annotation/step/0.json"
jq . "<unpack_root>/online_data/2/annotation/gesture/100.json"
```

You should see English descriptions, not bare integer strings.

---

## Python API

The CLI is a thin wrapper over `surgsync.decompose(...)`. To script
the unpack:

```python
from pathlib import Path
import surgsync   # or: from dvrk_data_processing import surgsync

report = surgsync.decompose(
    dataset_root=Path("<release_root>"),
    out_root=Path("<unpack_root>"),
    clips=["online_data/2"],          # filter (optional)
    streams=("raw", "preprocess"),    # default
    force=True,
    parallelism=2,
    workers_per_clip=4,
)
print(report.n_episodes_ok, "/", report.n_episodes_seen)
```

The reader API is also exposed for ad-hoc inspection without unpacking:

```python
ds = surgsync.open_dataset("<release_root>")
ep = ds["online_data/single_interrupted_stitch/2"]
print(ep.meta["operator_skill_level"])
for frame in ep.video_raw("stereo_left").iter_frames():
    ...   # frame is (H, W, 3) uint8 BGR
```

---

## Sweeping the other datasets

Same CLI; just point at the right packed root and out dir:

```bash
# Online ICRA data set (built earlier)
surgsync unpack \
    "<release_root>"  \
    --out "<unpack_root>" \
    --parallelism 4

# Offline data set (built earlier)
surgsync unpack \
    "<release_root>" \
    --out "<unpack_root>" \
    --parallelism 4
```

The corresponding raw sources are at
`<input data folder>` (online) and
`<input data folder>` (offline) — handy for
side-by-side comparison.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `output dir already populated: <path>; pass force=True to overwrite` | The target clip dir already has files from a prior run. | Pass `--force` to wipe and re-write, or unpack into a fresh `--out`. |
| `ffprobe not found on PATH` | Conda env not activated. | `conda activate dvrk_multimodal_process`. |
| All preprocess MKVs report as absent | The original pack was built with `include_preprocess=false` (or against a clip with no preprocessing outputs). | Re-pack the clip with preprocessing outputs available, or skip the preprocess stream via `--streams raw`. |
| `MISSING` files in a hash-compare against the raw clip | The raw clip had source-frame gaps that the packer compacted. Compare against `source_frame_index` mapping instead of expecting every raw filename to come back. | Walk `timestamp.parquet.source_frame_index` to see which raw indices are actually present after pack. |
| Decomposed annotation text differs from the raw numeric id | This is **by design** — the packer verbalizes ids through `workflow_description.json` at pack time, and the inverse direction emits the same text. The raw numeric id is not preserved. | Use `meta/tasks.jsonl` (in the packed dataset) to reverse-lookup numeric ids from text when needed. |
| `worker_per_clip=4` is slow on a USB SSD | exFAT / FUSE is slow at small-file writes; PNG fanout makes it worse. | Lower `--workers-per-clip` to 1 (less seek thrashing), or `--out` to an ext4/xfs volume. |
