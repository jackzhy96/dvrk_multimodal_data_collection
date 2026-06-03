# HOW_to_RUN

Single end-to-end runbook for preprocessing, packing, and unpacking.
For deeper notes on any stage, see the per-stage files at the repo
root:

- `HOW_to_RUN_Process.md` — every preprocessing stage and config knob.
- `HOW_to_RUN_pack.md` — every `surgsync build` flag.
- `HOW_to_RUN_unpack.md` — every `surgsync unpack` flag + fidelity table.

---

## 0. Prerequisites

```bash
git clone --recursive https://github.com/jackzhy96/dvrk_multimodal_data_collection.git
cd dvrk_multimodal_data_collection
bash env_setup/create_env.sh           # ~10–15 min (flash-attn compile)
conda activate dvrk_multimodal_process
pip install -e .
```

Sample clips are already in the repo:

- `data/online_data/2/` — 886 frames, online recorder (has setpoint_cp).
- `data/offline_data/3/` — 947 frames, offline recorder.

FoundationStereo weights for depth are not bundled; download per
`env_setup/INSTALL.md` if you want to run the depth-estimation stage.

---

## 1. Preprocessing — Process raw clips

The preprocessing pipeline turns the raw `image/` + `kinematic/` tree
into the derived modalities (rectified images, kinematic heatmaps,
depth, optical flow). Outputs land under `<clip>/preprocess/<stage>/`.

### One command, all four stages

```bash
python scripts/run_all_stages.py \
    path_config=jack_local_release \
    clips.source=list +clips.list='[online_data/2]'
```

Useful overrides:

| Flag | Effect |
|---|---|
| `clips.source=dataset clips.dataset_name=online_data` | Sweep every clip under a dataset. |
| `clips.source=all` | Sweep every clip under every dataset. |
| `stages='[rectify_resize,kinematic_reproject]'` | Run a subset of stages. |
| `force=true` | Re-run stages even if outputs exist. |

### Per-stage equivalents

Each stage is a standalone Hydra entry point. Run any of them individually:

```bash
# 1. Rectify + resize  (CPU)
cd src/dvrk_data_processing/raw_image_processing
python gen_rectify_resize.py \
    --config-name=config_rr_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2

# 2. Kinematic reproject  (CPU)
cd ../kinematic_mapping
python gen_kinematic_heatmap_handeye.py \
    --config-name=config_kp_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2

# 3. Depth estimation  (GPU, FoundationStereo)
cd ../depth_estimation
python gen_depth_estimate.py \
    --config-name=config_de_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2

# 4. Optical flow  (GPU, RAFT)
cd ../optical_flow
python gen_optical_flow_raft.py \
    --config-name=config_of_raft_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2
```

After preprocessing you should see, per clip:

```
data/<dataset>/<idx>/preprocess/
├── rectify_resize/
│   ├── image/{left,right}/<i>.png
│   ├── camera_calibration/{left.yaml,right.yaml,rectify_params.json}
│   ├── kinematic/   time_syn/
├── kinematic_reproject/{PSM1,PSM2}/{left,right}/{image,heatmap}/<i>.{png,npy}
├── kinematic_reproject/{PSM1,PSM2}/calibrated_kinematic/<i>.json
├── depth_estimation/{depth,disparity,depth_image,disparity_image,combined_image}/
└── optical_flow/{left,right}/{image,optical_flow}/<i>.{png,npy}
```

Stage knobs live in `config/preprocess/<stage>.yaml`. Full reference:
`HOW_to_RUN_Process.md`.

---

## 2. Packing — Pack to SurgSync

### One clip

```bash
surgsync build \
    clips.source=list \
    +clips.list='[online_data/2]' \
    path_config.dataset_root=/path/to/output/release
```

### Whole dataset, in parallel

```bash
surgsync build \
    clips.source=dataset clips.dataset_name=online_data \
    parallelism=4 \
    path_config.dataset_root=/path/to/output/release
```

### Useful overrides

| Flag | Effect |
|---|---|
| `tasks.default_task=auto` | (Default) Infer the task folder name per clip from its dominant phase. |
| `tasks.overrides.<dataset>/<idx>=<task>` | Force a specific task label for one clip. Repeatable. |
| `parallelism=N` | Pack N clips concurrently. ~`cores/3` is a good starting point. |
| `force=true` | Re-pack already-finalized episodes. |
| `release_option=A` | Skip geometry (preprocess) MKVs. |
| `include_preprocess=false` | Same effect. |
| `include_video_processed=false` | Skip rectified H.264 video. |
| `fps=30.0` | Playback fps stamped into MKVs. Default 10. |

### Validate before shipping

```bash
surgsync validate --layer=all \
    --raw-clip data/online_data/2 \
    --episode /path/to/release/online_data/episodes/single_interrupted_stitch/2 \
    --dataset-root /path/to/release
```

Exit 0 means clean; exit 2 means at least one ERROR. Full reference:
`HOW_to_RUN_pack.md`.

### Indexing

`surgsync build` runs the index builders at the end of every build. To
rebuild them manually:

```bash
surgsync index /path/to/release
```

### Release docs

```bash
surgsync release /path/to/release \
    --bump-version=minor \
    --notes 'Added 12 new suturing clips.'
```

Writes `README.md` + `CHANGELOG.md` into the release root and bumps
`meta/dataset.json:data_version` per semver.

---

## 3. Unpacking — Read or unpack a SurgSync release

### Read in Python

```python
import dvrk_data_processing.surgsync as ss

ds = ss.open_dataset("/path/to/release")
ep = ss.open_episode(ds.episodes[0].path)

print(ep.episode_id, ep.task, ep.length)
print(ep.psm1.column("measured_js.position")[0].as_py())
print(ep.annotation.column("phase")[0].as_py())  # verbalized text

for frame in ep.video_raw("stereo_left").iter_frames():
    ...  # (H, W, 3) uint8 BGR
    break
ep.close()
```

Reader cookbook with filtering / PyTorch DataLoader / preprocess
decoding: `docs/surgsync/loader_cookbook.md`.

### Unpack to the raw-style tree

```bash
surgsync unpack /path/to/release \
    --out /path/to/unpacked \
    --parallelism 4
```

### Useful flags

| Flag | Effect |
|---|---|
| `--clip <dataset>/<idx>` | Only one clip. Repeatable. |
| `--task <name>` | Only one task. Repeatable. |
| `--dataset-name <name>` | Only one top-level partition. Repeatable. |
| `--streams raw` | Only the raw clip tree (skip preprocess). |
| `--streams preprocess` | Only the preprocess tree. |
| `--force` | Overwrite already-populated output. |
| `--parallelism N` | Pack N clips concurrently. |
| `--workers-per-clip K` | PNG-writer threads per clip. Default 4. |

What you get under `<out>/<dataset>/<idx>/`:

```
image/{left,right,side}/<i>.png        # bit-exact (pixel) round-trip
kinematic/{ECM,PSM1,PSM2}/<i>.json     # float32-equivalent
annotation/{phase,step,gesture,contact_detection}/<i>.json
                                       # text descriptions (not numeric ids)
time_syn/<i>.json                      # tracked stamps bit-exact
camera_calibration/, hand_eye_calibration/, meta_data.json
                                       # byte-exact / reconstructed
preprocess/                            # rectified + depth + flow + heatmap
                                       # (PNGs only — raw .npy not recoverable)
```

Resume-friendly: re-run without `--force` and clips with the
`.surgsync_unpacked.json` sentinel are skipped in <10 ms.

### Verify against the raw original

```bash
python scripts/verify_unpack_vs_raw.py \
    --raw    data/online_data/2 \
    --unpack /path/to/unpacked/online_data/2 \
    --max-frames 10
```

Exit 0 means image pixels, kinematic JSON contents, time_syn stamps,
and calibration all match. Exit 2 means at least one bucket failed.

### Runtime + size breakdown

```bash
python scripts/unpack_breakdown.py \
    --packed   /path/to/release \
    --unpacked /path/to/unpacked \
    --raw      online_data=/path/to/raw_online \
    --raw      offline_data=/path/to/raw_offline \
    --log      /path/to/unpack.log
```

Prints wall-clock, per-clip elapsed, per-stream throughput (when a log
is given), and packed-vs-unpacked-vs-raw size tables.

Full reference: `HOW_to_RUN_unpack.md`.

---

## 4. End-to-end smoke (preprocessing → packing → unpacking) on a fresh clip

```bash
# 1. Process the clip.
python scripts/run_all_stages.py \
    path_config=jack_local_release \
    clips.source=list +clips.list='[online_data/2]'

# 2. Pack it.
surgsync build \
    clips.source=list +clips.list='[online_data/2]' \
    path_config.dataset_root=/tmp/release

# 3. Validate.
surgsync validate --layer=dataset --dataset-root /tmp/release

# 4. Unpack.
surgsync unpack /tmp/release --out /tmp/unpack

# 5. Verify the round-trip against the raw.
python scripts/verify_unpack_vs_raw.py \
    --raw    data/online_data/2 \
    --unpack /tmp/unpack/online_data/2

# 6. Stamp release docs.
surgsync release /tmp/release --bump-version=patch --notes 'First release.'
```

---

## 5. Tests

```bash
conda activate dvrk_multimodal_process
pytest tests/surgsync/                            # full suite (~25 min)
pytest tests/surgsync/serde/                      # serde round-trips
pytest tests/surgsync/decompose/                  # unpack round-trip
pytest tests/surgsync/integration/test_full_smoke.py
                                                  # full pack→unpack smoke (~13 min)
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ffprobe not found on PATH` | `conda activate dvrk_multimodal_process`. |
| `MissingPreprocessingOutputError` during `surgsync build` | Run `scripts/run_all_stages.py` first, or pass `include_preprocess=false include_video_processed=false`. |
| `output dir already populated` from `surgsync unpack` | Pass `--force`, or point `--out` at a fresh directory. |
| Unpack PNG bytes don't hash-match the raw originals | Expected — PNG file bytes differ across encoders. Use `scripts/verify_unpack_vs_raw.py`, which compares decoded pixels. |
| `data_version '1.0' is not parseable` | Should not happen — `surgsync release` accepts `M`, `M.m`, or `M.m.p`. Re-pull to update. |
| Slow finalize on `surgsync build` | Output disk is exFAT; move `dataset_root` to ext4/xfs. |
