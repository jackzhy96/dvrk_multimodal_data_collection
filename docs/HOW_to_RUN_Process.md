# HOW_to_RUN_Process.md

Operational walkthrough for the **preprocessing pipeline**. Covers every
stage from raw inputs to per-stage outputs, the sample data layout, the
batch orchestrator, and the unit + E2E smoke tests.

For the higher-level project README see `README.md`.

---

## 0. Prerequisites

```bash
# environment
conda activate dvrk_multimodal_process
# package (idempotent)
pip install -e .
```

Place your raw clips under `data/`, one folder per clip:

```
data/
├── offline_data/3/         # offline recorder — no setpoint_cp in kinematic JSON
└── online_data/2/          # online recorder — has setpoint_cp
```

Each clip folder contains `image/{left,right,…}/`, `kinematic/{PSM1,PSM2,ECM}/`,
`camera_calibration/{left.yaml,right.yaml,stereo_calib_params.json}`,
`hand_eye_calibration/`, `time_syn/`, `annotation/`.

**FoundationStereo weights** (needed for stage 3, depth) are NOT bundled.
Download `23-51-11/model_best_bp2.pth` per `env_setup/INSTALL.md` and place it
at:

```
FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

If the weights are absent, the depth stage is skipped with a clear message —
the other three stages still run end-to-end.

---

## 1. Hydra config layout (one-time read)

Every pipeline script is a Hydra entry point composed from three layers:

1. **Running config** at `config/config_<op>_<user>.yaml` — picks `path_config`
   + `preprocess` via the `defaults:` block.
2. **Path config** at `config/path_config/<user>_local*.yaml` — `data_dir`,
   `data_name`, `data_index`.
3. **Preprocess config** at `config/preprocess/<stage>.yaml` — algorithm knobs.

For multi-clip runs use the preprocessing path config:

```yaml
# config/path_config/jack_local_release.yaml
data_dir: ".../dvrk_multimodal_data_collection/data"
data_name: "offline_data"
data_index: "3"
raw_dir:          "${.data_dir}/${.data_name}/${.data_index}"
intermediate_dir: "${.data_dir}/intermediate/${.data_name}/${.data_index}"
processed_dir:    "${.data_dir}/preprocess/${.data_name}/${.data_index}"
```

You can switch clips on the command line via Hydra overrides
(`path_config.data_index=3`, `path_config.data_name=online_data`).

---

## 2. Run each stage manually

Activate the env first (`conda activate dvrk_multimodal_process`). All
commands assume you're at the repo root.

### Stage 1 — Rectify + Resize  (CPU)

```bash
cd src/dvrk_data_processing/raw_image_processing
python gen_rectify_resize.py \
    --config-name=config_rr_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2
```

Reads `${path_config.raw_dir}/image/{left,right}/` and writes:

```
intermediate_dir/
├── camera_calibration/{left.yaml,right.yaml,rectify_params.json,stereo_calib_params.json}
├── image/{left,right}/<frame>.png       # rectified + resized
├── kinematic/                            # copied verbatim
└── time_syn/                             # copied verbatim
```

> **DO NOT** call `gen_resize_rectify.py` — it is **deprecated**. Calling it
> raises a `RuntimeError` pointing at `gen_rectify_resize.py`.

### Stage 2 — Kinematic re-projection  (CPU)

```bash
cd src/dvrk_data_processing/kinematic_mapping
python gen_kinematic_heatmap_handeye.py \
    --config-name=config_kp_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2
```

Output (folder renamed `kinematic_map/` → `kinematic_reproject/`):

```
processed_dir/kinematic_reproject/<PSM>/<cam>/{image,heatmap}/<frame>.{png,npy}
processed_dir/kinematic_reproject/<PSM>/calibrated_kinematic/<frame>.json   # handeye only
processed_dir/kinematic_reproject_drawframe/<PSM>/<cam>/<frame>.png         # sibling tree
```

Feature gates (in `config/preprocess/kinematic_reproject.yaml`):
- `calibrated_kinematic.enable: true` (default) — emit per-frame 6-DoF pose JSON.
- `drawframe.enable: true` (default) — render tool-tip axes on rectified images.

Disable either via CLI override:

```bash
python gen_kinematic_heatmap_handeye.py \
    +preprocess.drawframe.enable=false \
    +preprocess.calibrated_kinematic.enable=false
```

### Stage 3 — Depth estimation  (GPU; FoundationStereo)

```bash
cd src/dvrk_data_processing/depth_estimation
python gen_depth_estimate.py \
    --config-name=config_de_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2
```

Output:

```
processed_dir/depth_estimation/
├── disparity/<i>.npy        # raw FoundationStereo disparity (float32, pixels)
├── disparity_image/<i>.png  # vis_disparity colorization
├── combined_image/<i>.png   # [left | disparity] side-by-side
├── depth/<i>.npy            # meters (float32, NaN where invalid)
└── depth_image/<i>.png      # INFERNO colormap of depth/<i>.npy
```

Depth knobs (`config/preprocess/depth_estimation.yaml`):
- `compute_depth: true` — write `depth/<i>.npy` alongside disparity.
- `depth_eps: 1e-3` — disparity ≤ this → depth = NaN (never clamped).
- `depth_viz_range_m: [0.02, 0.5]` — INFERNO colormap window in meters.
- `stereo_calib_filename: stereo_calib_params.json` — relative to the
  intermediate_dir camera_calibration folder.

### Stage 4 — Optical flow (RAFT)  (GPU)

```bash
cd src/dvrk_data_processing/optical_flow
python gen_optical_flow_raft.py \
    --config-name=config_of_raft_jack \
    path_config=jack_local_release \
    path_config.data_name=online_data \
    path_config.data_index=2
```

Output:

```
processed_dir/optical_flow/<cam>/optical_flow/<i>.{npy,flo}     # flow frame[i] → frame[i+1]
processed_dir/optical_flow/<cam>/image/<i>.png                  # color viz
```

> The Farneback variant `gen_optical_flow.py` is tagged **legacy** in its
> top-of-file banner; prefer RAFT.

---

## 3. Batch orchestrator

For multi-clip release runs, use the single entry point that sweeps all
four stages over a list of clips in stage-first order (model loads once
per stage, not once per clip). The execution order is **fixed** and matches
the preprocessing stage map:

```
1. rectify_resize        (CPU, always runs first — produces intermediate_dir)
2. kinematic_reproject   (CPU, depends on intermediate_dir from stage 1)
3. depth_estimation      (GPU, depends on intermediate_dir from stage 1)
4. optical_flow_raft     (GPU, depends on intermediate_dir from stage 1)
```

Stages 2-4 are independent of each other but all depend on stage 1 having
finished for that clip. The orchestrator enforces this by running stage 1
across every requested clip before stage 2 starts, etc.

### Recipe — process every `<index>` under one dataset folder

This is the most common use. Say your dataset lives at
`data/offline_data/{1,2,3,…}/` and you want every subfolder processed:

```bash
python scripts/run_all_stages.py \
    path_config=jack_local_release \
    clip_indices.source=dataset \
    clip_indices.dataset_name=offline_data
```

`source=dataset` auto-discovers every subfolder under
`<data_dir>/<dataset_name>/` and sweeps stages 1 → 2 → 3 → 4 across them.
Subfolders are sorted numerically when every name parses as an integer
(so `1, 2, 10` runs in that order, not `1, 10, 2`).

### Recipe — process an explicit list of clips

```bash
python scripts/run_all_stages.py \
    path_config=jack_local_release \
    path_config.data_name=offline_data \
    clip_indices.source=list \
    'clip_indices.list=["1","2","3"]'
```

(Pass `clip_indices.list` as a Hydra list literal — note the outer single
quotes around the whole override and the JSON-style brackets.)

### Recipe — sweep multiple datasets via glob

```bash
python scripts/run_all_stages.py \
    path_config=jack_local_release \
    clip_indices.source=glob \
    'clip_indices.glob_patterns=["offline_data/*","online_data/*"]'
```

Each pattern is joined against `path_config.data_dir`. The `data_name` for
each discovered clip is taken from its parent folder (`offline_data` or
`online_data` in the example).

### Recipe — defaults from the YAML file

When invoked with no overrides, `scripts/run_all_stages.py` reads
`config/run_all_stages.yaml` and uses its defaults. Edit the YAML in place
if your sweep is stable; pass overrides on the command line when you want
to vary something for a single run.

```bash
# uses defaults — clip_indices.list = ["3"], path_config = jack_local_release
python scripts/run_all_stages.py
```

### Recipe — skip the GPU stages on a CPU-only machine

```bash
python scripts/run_all_stages.py \
    stages.depth_estimation.enable=false \
    stages.optical_flow_raft.enable=false
```

The GPU stages also auto-skip when `skip_if_no_gpu=true` (the default for
depth + RAFT) and `nvidia-smi` is unavailable on the host.

### Behavior cheat sheet

- **Stage-first ordering** — every clip finishes stage 1 before stage 2 runs,
  then every clip finishes stage 2 before stage 3 runs, and so on. This is
  not configurable.
- **Resumability** — a (stage, clip) is skipped when the expected output
  directory already exists and is non-empty. Use `force=true` to override.
  This means re-running after a crash picks up where it left off — safe to
  invoke the same command twice.
- **Per-clip failures are non-fatal** — if `gen_*` exits non-zero on one
  clip, the orchestrator logs the failure and moves on. Other clips and
  later stages still run.
- **Structured log** — JSONL is appended to
  `${path_config.data_dir}/.run_all_stages/<run_id>.jsonl` so you can audit
  outcomes after a long run (one line per stage-start, stage-end, clip-start,
  clip-skipped, clip-end event).

---

## 4. Tests

### Synthetic unit tests (CPU only, fast)

```bash
python tests/processing/test_unit_synthetic.py
```

Covers:
- rotation composition for `compute_calibrated_tip_pose` and the
  asymmetric `compute_calibrated_setpoint_pose` (setpoint must skip `T_W_B`).
- `disparity_to_depth_m` round-trip on a hand-built disparity
  field, plus NaN handling (zero disparity → NaN; NaN disparity → NaN).
- analytical projection of a known axis triad through a known K
  matrix, plus the behind-camera (z ≤ 0) early-out.

Expected output: `8/8 tests passed`.

### End-to-end smoke test (needs sample data + ideally GPU)

```bash
# all stages on both sample clips
python tests/processing/test_pipeline_e2e.py

# CPU-only box: skip the GPU stages
python tests/processing/test_pipeline_e2e.py --skip-depth --skip-flow
```

What it checks per clip:
- Stage 1: `image/left/`, `image/right/`, `camera_calibration/`,
  `rectify_params.json` all populated.
- Stage 2: per-PSM heatmap directories; `calibrated_kinematic/` schema
  (positions length-3, quat length-4 xyzw); `setpoint_cp_calibrated` is
  **present** on the online clip and **absent** on the offline clip; no
  ECM `calibrated_kinematic/` directory exists; `kinematic_reproject_drawframe/`
  populated for both PSMs and both cameras.
- Stage 3 (when not skipped): `disparity/`, `depth/`, `depth_image/` populated;
  at least one valid (non-NaN) depth value within a physically plausible range
  (0 < depth < 5 m).
- Stage 4 (when not skipped): per-camera `optical_flow/` populated.

Expected runtime: < 10 minutes on a dev box with GPU.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: gen_resize_rectify.py is deprecated …` | Called the deprecated stage-1 script. | Switch to `gen_rectify_resize.py`. |
| Stage 2 errors `Cannot find Hand-Eye file …PSMx-registration-dVRK.json` | The clip's `hand_eye_calibration/` folder is missing the expected JSONs. | Confirm the per-arm registration files are present under `${raw_dir}/hand_eye_calibration/`. |
| Stage 3 errors `Pretrained model not found …model_best_bp2.pth` | FoundationStereo weights absent. | Download per `env_setup/INSTALL.md` into `FoundationStereo/pretrained_models/23-51-11/`. |
| Stage 3 depth values are all NaN | `depth_eps` set too high, or disparity model failed on a textureless region. | Inspect `disparity/<i>.npy` first; if disparity is reasonable, lower `depth_eps`. |
| Stage 4 fails with CUDA OOM | RAFT batch size too large for the GPU. | `+preprocess.model_config.batch_size=1` (already the default). |
| Drawframe origin is visibly off the tool tip in `kinematic_reproject_drawframe/` | Hand-eye calibration drift (Risk A-3) or wrong `tool_tip_offset` for that arm. | Re-run the hand-eye calibration; verify `tool_tip_offset` in `config_kp_jack.yaml`. |
| `kinematic_reproject_drawframe/` is empty | Tool tip is behind the camera (logged as `behind_camera_count`) or the feature is disabled. | Check the script's final log line; flip `drawframe.enable` in the preprocess YAML. |
| `run_all_stages.py` skips every clip | Outputs already present and `force=false`. | Either delete the output trees or pass `force=true`. |
