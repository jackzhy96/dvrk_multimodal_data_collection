# How to run the SurgSync packer

This is the operator guide for `surgsync build` — the packer that
turns raw dVRK clips + the preprocessing outputs into a SurgSync
dataset on disk.

For the architectural background, read `tasks/M2-packing.md` and
`specs/code_design.md` first. This document is "how to drive it", not
"how it works".

---

## Prerequisites

You must already have:

1. **Conda env** built and active:
   ```bash
   bash env_setup/create_env.sh   # one-time, ~10–15 min
   conda activate dvrk_multimodal_process
   pip install -e .
   ```

2. **Raw data** at `data/{offline_data,online_data}/<idx>/` — one
   self-contained recording per `<idx>` folder (see
   `specs/raw_data_spec.md`).

3. **Preprocessing outputs** for every clip you want to pack. The
   preprocessing pipeline writes everything **inside the clip
   directory** under a `preprocess/` subtree (one sibling per stage):
   - `data/<dataset>/<idx>/preprocess/rectify_resize/` — stage 1
     (rectified+resized images, scaled calibration,
     `rectify_params.json`, copied kinematic + time_syn).
   - `data/<dataset>/<idx>/preprocess/{kinematic_reproject, kinematic_reproject_drawframe, depth_estimation, optical_flow}/` —
     stages 2 / 3 / 4 outputs.

   Run the preprocessing pipeline first if you don't have these:
   ```bash
   python scripts/run_all_stages.py path_config=jack_local_release \
       clips.source=dataset clips.dataset_name=online_data
   ```

   If preprocessing outputs are missing when `surgsync build` runs,
   the clip is logged as skipped with a clear "run
   `run_all_stages.py` first" message and the sweep continues.

4. **Disk space**: each packed episode is roughly 1–2 GB per minute of
   stereo+side raw video, dominated by `video_raw/` (FFV1, bit-exact).
   The sample `online_data/2` clip (~3 min, 886 frames) packs to ~5.2 GB.

---

## One-liner: pack one clip

```bash
conda activate dvrk_multimodal_process

surgsync build \
    clips.source=list \
    +clips.list='[online_data/2]' \
    path_config.dataset_root='/path/to/output/surgsync_release'
```

`tasks.default_task` defaults to `"auto"` — the packer reads each
clip's `annotation/phase/<frame>.json`, takes the dominant phase, and
maps it via `workflow_description.json:_task_routing` to pick the
folder name. No manual task argument is needed unless you want to
override it (see "Useful overrides" below).

What this does:
1. Reads `data/online_data/2/` (raw root) +
   `data/online_data/2/preprocess/rectify_resize/` (preprocessing stage 1) +
   `data/online_data/2/preprocess/{kinematic_reproject, depth_estimation, optical_flow, ...}/`
   (preprocessing stages 2/3/4). If preprocessing outputs are absent
   the clip still packs (raw video + parquets + raw calibration);
   `include_video_processed` and `include_preprocess` gate the
   preprocessing-dependent encoders.
2. Aligns every per-modality stream to the stereo-left master clock.
   Produces a signed `delta_to_master.<topic>_ns` column **per topic
   in `align.topics.TIMESTAMP_TOPICS`** (image_right, image_side,
   ECM/PSM1/PSM2 × measured_js/cp/cv, setpoint_js/cp, local_measured_cp,
   plus PSM jaw_measured/jaw_setpoint — 24 topics total). **The packer
   never drops or NULLs data based on tolerance**; every preprocessing
   value is preserved verbatim. Tolerance assessments surface only in
   `episode_meta.json.sync_stats.out_of_tol_counts` (informational).
3. Encodes raw videos, processed + geometry videos (when preprocessing ran),
   the five per-modality parquets (`timestamp.parquet`, `ECM.parquet`,
   `PSM1.parquet`, `PSM2.parquet`, `annotation.parquet`),
   `episode_meta.json`, `time_sync_stat.json` (per-topic latency
   detail), `modalities.json`, and `calibration/` files (originals
   from `raw/camera_calibration/` + optional `rectify_params.json`
   from preprocessing).
4. Marks the episode complete by atomically writing
   `.surgsync_complete.json` into the final dir. Build happens in
   place (no `.staging/` rename); a `.surgsync_running.json` marker
   sits in the dir while encoding is in flight, and an
   `.surgsync_failed.json` lands if anything raises mid-pack.

Runtime: ~5–6 min/clip serial, ~1.5 min/clip amortized at
`parallelism=4` on an 8+ core dev box. The encode itself is fast
(~1 min); the bulk on exFAT volumes is the OS page-cache flush during
the final fsync. ext4/xfs targets are sub-second.

---

## Sweep a whole dataset

```bash
surgsync build \
    clips.source=dataset clips.dataset_name=online_data \
    parallelism=4 \
    path_config.dataset_root='/path/to/output/surgsync_release'
```

Or pack offline + online together (sweeping every clip under both):

```bash
surgsync build \
    clips.source=all \
    parallelism=4 \
    path_config.dataset_root='/path/to/output/surgsync_release'
```

With `tasks.default_task=auto` (the build.yaml default), each clip
auto-routes to its phase-derived task folder. To force a specific
label per clip, use `tasks.overrides`:

```bash
surgsync build \
    clips.source=all \
    parallelism=4 \
    tasks.overrides.online_data/2=single_interrupted_stitch \
    tasks.overrides.offline_data/3=tissue_manipulation \
    path_config.dataset_root='/path/to/output/surgsync_release'
```

`tasks.overrides` maps `<dataset>/<clip_idx>` → task name; clips not
in the override map fall back to `tasks.default_task` (auto-inference
unless explicitly set to a literal task name).

---

## Useful overrides

| Override | Meaning |
|---|---|
| `tasks.default_task=auto` | (Default) Infer the task folder name per clip from the dominant phase in `annotation/phase/*.json`. Set to a literal task name (e.g. `single_interrupted_stitch`) to force every unrouted clip into one folder. |
| `tasks.overrides.<dataset>/<idx>=<task>` | Per-clip force. Always wins over auto-inference and `default_task`. |
| `parallelism=N` | Pack N clips concurrently via a `ProcessPoolExecutor`. Each worker also encodes its 3 video streams in parallel via threads. Set to ~`(num_cores // 3)` for the sweet spot — beyond that you over-subscribe ffmpeg subprocesses. |
| `force=true` | Re-pack already-finalized episodes (wipes them first). |
| `clean_staging=true` | Wipe any leftover incomplete episode dir (`.surgsync_running.json` or `.surgsync_failed.json` present, no `.surgsync_complete.json`) from a crashed prior run. |
| `release_option=A` | Pack Option A only (no preprocess MKVs); smaller release. |
| `include_preprocess=false` | Same effect: skip preprocess encoding. |
| `include_video_processed=false` | Skip the preprocessing-rectified H.264 MP4s. Needed when packing clips that have no preprocessing outputs yet. |
| `include_video_raw=false` | **Not recommended.** Disables bit-exact raw video. Breaks the unpack contract — the packed dataset can no longer reproduce the raw `image/` folder. |
| `fps=30.0` | Override the playback fps stamped into the MKV containers. Default is **10.0** to match the dVRK capture rate; does NOT change the master timeline, only video player metadata. |
| `align/tol_ms_kinematic_online=2.0` | Tighten kinematic tolerance — the default 100 ms covers standard rosbag-recorded data; tighten to 2 ms if your recording uses a tight real-time loop. |

Full config schema lives in `src/dvrk_data_processing/surgsync/config.py`
(dataclass `SurgSyncBuildCfg`) and `config/surgsync/build.yaml`.

---

## Output structure (after `surgsync build`)

A successful build writes the following tree under
`<dataset_root>`. The top-level partition is the **raw dataset name**
(e.g. `offline_data`, `online_data`, or any future name like
`synthetic_data`) — the same string used as the top-level folder under
`data/`. Inside, episodes partition by task name and then by raw clip
index. Paths round-trip trivially back to the source clip.

```
<dataset_root>/
├── meta/
│   └── dataset.json                          # release metadata; tasks list is the union of every task folder on disk
├── .logs/<run_id>.jsonl                      # structured build log (per-clip outcomes)
├── offline_data/
│   └── episodes/<task>/<clip_idx>/
│       └── ...
└── online_data/
    └── episodes/<task>/<clip_idx>/
        ├── .surgsync_complete.json           # sentinel manifest written last; absence ⇒ in-flight / failed
        ├── episode_meta.json                 # per-episode metadata + sync_stats
        │                                     #   episode_id = "<dataset_name>_<clip>_<master_t0_ns>"
        │                                     #   master_t0_ns = absolute ns of frame 0
        │                                     #   master_timestamp_ns in every parquet is rebased to clip start (frame 0 == 0)
        ├── time_sync_stat.json               # per-topic latency detail (median/mean/std/max + max_delta_frame_idx + n_present)
        ├── timestamp.parquet                 # master clock + ALL delta_to_master.<topic>_ns columns (24 topics) + contiguity
        ├── ECM.parquet                       # every field of kinematic/ECM/<i>.json — no delta columns (those live in timestamp.parquet)
        ├── PSM1.parquet                      # every field of kinematic/PSM1/<i>.json + jaw + freq — no delta columns
        ├── PSM2.parquet                      # same shape as PSM1
        ├── annotation.parquet                # contact/gesture/phase/step aligned to stereo-left timestamp; "None"/"null" strings normalized to JSON null
        ├── modalities.json                   # per-episode manifest of which streams + topics are present
        ├── video/                            # MP4/H.264 visually-lossless (rectified resolution; present iff preprocessing ran)
        │   ├── stereo_left.mp4
        │   └── stereo_right.mp4
        ├── video_raw/                        # MKV/FFV1 bit-exact (raw resolution; MANDATORY for decomposability)
        │   ├── stereo_left.mkv
        │   ├── stereo_right.mkv
        │   └── side.mkv                      # (present when the clip has a side camera)
        ├── preprocess/                       # FFV1 bit-exact over preprocessing viz PNGs (present iff preprocessing ran)
        │   ├── depth.mkv                     # bgr24 from depth_estimation/depth_image/<i>.png
        │   ├── flow_left.mkv                 # bgr24 from optical_flow/left/image/<i>.png
        │   ├── flow_right.mkv
        │   └── heatmap_PSM{1,2}_{left,right}.mkv   # gray8 from kinematic_reproject/
        └── calibration/
            ├── camera.json                   # convenience index (image size, fx, baseline)
            ├── left.yaml                     # ORIGINAL raw CRTK YAML — native camera resolution, verbatim
            ├── right.yaml                    # same
            ├── stereo_calib_params.json      # ORIGINAL raw stereo extrinsics + baseline (when present)
            ├── rectify_params.json           # P1/P2/Q at rectified resolution (only when preprocessing ran)
            └── hand_eye/
                ├── PSM1-registration-dVRK.json
                ├── PSM1-registration-open-cv.json
                ├── PSM2-registration-dVRK.json
                └── PSM2-registration-open-cv.json
```

**Notes**:
- The packer never transforms preprocessing values. Alignment deltas
  are computed and surfaced in `timestamp.parquet` +
  `episode_meta.json.sync_stats` for analysis, but never used to drop
  or NULL data.
- Lossy steps are limited to encoder choices (H.264 CRF 18 on
  processed video; uint16 quantization on geometry) — bounded and
  documented in their respective encoder docstrings.
- **Calibration source**: the YAMLs shipped in `calibration/` are the
  **original raw** files at native camera resolution. Consumers who
  need the rectified-resolution intrinsics derive them from raw +
  `rectify_params.json` on demand. The earlier behavior of shipping
  the preprocessing-scaled intermediate YAMLs has been dropped.
- **Sentinel protocol**: writes go directly to the final episode
  directory (no `.staging/` rename — that was the long pole on exFAT
  volumes). A `.surgsync_running.json` marker stamps the dir on entry
  carrying `{pid, host, started_at_utc, episode_id}`. On success,
  `.surgsync_complete.json` is written via atomic temp+rename
  carrying the same fields plus `length_frames` + `duration_s` + a
  `completed_at_utc`; the running marker is then removed. On
  exception, `.surgsync_failed.json` lands instead (with traceback)
  and the running marker is removed. All scanners — validator, index
  builder, unpack reader — treat **only** the complete sentinel as a
  shippable signal.

---

## Compression ratio (sample: online_data/2 — 886 frames, ~88 s)

| Bucket | Before (raw on disk) | After (packed) | Ratio |
|---|---:|---:|---:|
| Raw images (3 cams) | ~7.6 GB | ~4.5 GB (FFV1 bit-exact) | ~1.7× |
| Rectified images (stereo) | ~580 MB | ~30 MB (H.264 CRF 18) | ~20× |
| Disparity .npy | ~520 MB | ~40 MB (FFV1 uint16) | ~13× |
| Optical flow .npy (2 cams) | ~1.0 GB | ~30 MB (FFV1 gbrp16le) | ~33× |
| Heatmap .npy + PNG | ~1.3 GB | ~30 MB (FFV1 gray8 viz) | ~43× |
| Kinematic + annotation JSONs | ~50 MB | ~3 MB (per-modality parquets) | ~17× |

The dominant cost is `video_raw/` — that's deliberate (bit-exact raw is
required for the unpack stage to reproduce the original PNGs). Disable it
with `include_video_raw=false` for a much smaller release at the cost
of decomposability.

---

## Verifying a build

The packer logs every per-clip outcome to JSONL. To check what
happened:

```bash
cat <dataset_root>/.logs/*.jsonl | jq -c 'select(.event=="clip_end" or .event=="clip_skipped" or .event=="clip_failed")'
```

Inspect a parquet:

```bash
python - <<'PY'
import pyarrow.parquet as pq

# Cross-modal alignment lives in timestamp.parquet (every delta_to_master.*_ns
# column for every topic in align.topics.TIMESTAMP_TOPICS).
ts = pq.read_table('<dataset_root>/online_data/episodes/<task>/<clip_idx>/timestamp.parquet')
print(ts.num_rows, 'rows ×', len(ts.column_names), 'columns (expect 29 with the full topic catalog)')
print('frame 0 master_timestamp_ns:', ts.column('master_timestamp_ns')[0].as_py(), '(clip-relative)')
print('frame 0 PSM1.measured_cp delta:',
      ts.column('delta_to_master.PSM1.measured_cp_ns')[0].as_py(), 'ns')

# Per-arm state lives in {ECM,PSM1,PSM2}.parquet — no delta columns there anymore.
psm1 = pq.read_table('<dataset_root>/online_data/episodes/<task>/<clip_idx>/PSM1.parquet')
print('PSM1 joint position (frame 0):', psm1.column('measured_js.position')[0])
print('PSM1 measured_cp twist (frame 0):', psm1.column('measured_cp.velocity')[0])
PY
```

Inspect the episode metadata + per-topic latency:

```bash
jq . '<dataset_root>/online_data/episodes/<task>/<clip_idx>/episode_meta.json'
jq '.per_topic | to_entries[] | "\(.key)\t\(.value.median_delta_ms)\t\(.value.max_delta_ms)\t\(.value.n_present)"' \
    '<dataset_root>/online_data/episodes/<task>/<clip_idx>/time_sync_stat.json' -r
```

The complete sentinel doubles as a fast "did this clip ship?" check:

```bash
test -f '<dataset_root>/online_data/episodes/<task>/<clip_idx>/.surgsync_complete.json' \
    && jq . "$_" || echo "NOT COMPLETE — check .surgsync_running.json or .surgsync_failed.json"
```

---

## Codec selftest

Before a long sweep, run the codec round-trip selftest:

```bash
surgsync selftest
```

Should print `selftest OK`. This validates FFV1 bit-exact round-trip
for `gray16le` + `gbrp16le` and the H.264 pipeline. Run it once after
a fresh env setup and again any time you change ffmpeg versions.

---

## Indexing and validation

`surgsync build` now auto-runs the index builders at the end of every
build. After a successful build:

- `meta/episodes.parquet/task=<task>/part-00000.parquet` — one row per episode (Hive-partitioned)
- `meta/episodes.jsonl` — convenience copy for humans / grep
- `meta/index.parquet/task=<task>/part-00000.parquet` — frame-level cross-episode index
- `meta/stats.parquet` — per-column min/max/mean/std/q01/q99 + vocab_size for strings
- `meta/manifest.json` — SHA256 of every file (except itself)

To rebuild the indexes against an existing dataset (e.g. after manually
adding an episode dir):

```bash
surgsync index <dataset_root>
```

To validate a packed dataset:

```bash
surgsync validate --layer=all \
    --raw-clip data/online_data/2 \
    --episode '<dataset_root>/online_data/episodes/suturing/2' \
    --dataset-root '<dataset_root>'
```

Each layer surfaces ERROR / WARNING / INFO findings; exit code is 0
when no ERRORs were found, 2 otherwise.

## Limitations

These are spec'd but not in this release:

- **Preview encoder** (Option C): the H.264 8-bit colorized previews
  are not built; `include_preview=true` logs a stub message.
- **Vocab file** (`meta/tasks.jsonl`): **auto-generated** by every
  build by projecting `workflow_description/workflow_description.json`
  — one row per `_task_routing` entry, each row carrying its phase's
  exact `step` + `gesture` vocab. No hand-authored sidecar. Unpack
  round-trip of the verbalized text → numeric id is explicitly out
  of scope (the JSON is the source of truth, not the parquet cells).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MissingPreprocessingOutputError: ... intermediate_dir/image/left/ does not exist` | Preprocessing hasn't run for this clip yet **and** you requested a preprocessing-dependent encoder. | Either run `python scripts/run_all_stages.py` first, or pack with `include_video_processed=false include_preprocess=false`. |
| `FatalExportError: ... tasks.default_task='auto' but the clip's annotation/phase/ is missing` | The clip's `annotation/phase/<frame>.json` files don't exist or all map to unrouted phase ids. | Set an explicit `tasks.overrides.<dataset>/<idx>=<task>` for that clip, or restore the phase annotations. |
| `selftest FAILED: ffmpeg not found on PATH` | ffmpeg not in the active env. | `conda activate dvrk_multimodal_process`. |
| Build crashes with `No space left on device` | Output disk is full. | Re-point `path_config.dataset_root` to a disk with more room. |
| The pack hangs for several minutes on the **finalize** step per clip | Your `dataset_root` is on **exFAT** or **FUSE**, where the OS triggers a global page-cache flush on the sentinel's fsync. The build is correct; this is the filesystem catching up on buffered writes. | If you can, move `dataset_root` to ext4/xfs/btrfs — finalize becomes sub-second. Otherwise it's the cost of doing business on the SSD. |
| `gesture.PSM1 contains text not from the workflow_description table: 'None'` from `surgsync validate` | The raw annotation cell carried the literal string `"None"` and predates the ingest normalization (or the dataset was packed before that fix). | Re-pack the clip — the ingest layer now collapses `"None"`/`"null"`/`""` (case-insensitive) to JSON null, which the validator skips. |
| `surgsync validate` reports `task <X> present under episodes/ but not in dataset.json.tasks` | The `dataset.json.tasks` list got out of sync with the on-disk episode tree (e.g. you ran an incremental pack of a single task). | Re-run any `surgsync build` — `_write_dataset_meta` discovers existing tasks on disk and unions them with the current run's. |
| All `state.PSM*.setpoint_cp.*` columns are NULL on offline clips | The offline recorder has no Cartesian setpoint by design. The corresponding `delta_to_master.PSM*.setpoint_cp_ns` is also NULL, and `time_sync_stat.json` reports `n_present=0` for that topic. **Not a bug.** | Use `measured_*` columns for offline clips. |
| Large `sync_stats.out_of_tol_counts` for `*.setpoint_*` | Setpoint topics often lag the camera by 10–30 ms in non-real-time recordings. The data is still preserved (no rows are NULL'd based on tolerance). | Bump `align.tol_ms_kinematic_<variant>` if the count concerns you, or filter on the delta columns at training time. |
