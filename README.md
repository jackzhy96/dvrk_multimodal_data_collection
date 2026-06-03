# dVRK Multi-Modal Data Collection

End-to-end toolkit for the **da Vinci Research Kit (dVRK)** multi-modal
surgical dataset: from raw clip on disk → derived modalities (depth, optical
flow, kinematic heatmaps) → packed SurgSync dataset → trainable Python API
→ round-trippable back to the original raw layout.

The packed format (**SurgSync**) is the shippable artifact; the unpack
direction is the literal inverse so consumers can always recover the raw
PNGs + JSONs from a packed release.

Author: Haoying (Jack) Zhou — `hzhou62@jh.edu` / `hzhou6@wpi.edu` ·
[github.com/jackzhy96](https://github.com/jackzhy96).

---

## Contents

- [What's in this repo](#whats-in-this-repo)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Install](#install)
- [Raw clip layout](#raw-clip-layout)
- [Quick start](#quick-start)
- [Stage-by-stage operator guide](#stage-by-stage-operator-guide)
  - [Preprocessing — Process raw clips](#preprocessing--process-raw-clips)
  - [Packing — Pack to SurgSync](#packing--pack-to-surgsync)
  - [Unpacking — Read or unpack a SurgSync dataset](#unpacking--read-or-unpack-a-surgsync-dataset)
  - [Cleanup — Release docs](#cleanup--release-docs)
- [Reader API cheat sheet](#reader-api-cheat-sheet)
- [Repo layout](#repo-layout)
- [Hardware & data-collection tooling](#hardware--data-collection-tooling)
- [Tests](#tests)
- [Further reading](#further-reading)
- [License + contact](#license--contact)

---

## What's in this repo

| Capability | CLI | Doc |
|---|---|---|
| Run preprocessing on raw clips | `python scripts/run_all_stages.py …` | [`docs/HOW_to_RUN_Process.md`](docs/HOW_to_RUN_Process.md) |
| Pack clips into a SurgSync release | `surgsync build …` | [`docs/HOW_to_RUN_pack.md`](docs/HOW_to_RUN_pack.md) |
| Validate a packed release | `surgsync validate …` | `docs/HOW_to_RUN_pack.md` § validate |
| Read a packed release in Python | `surgsync.open_dataset(...)` | [`docs/surgsync/loader_cookbook.md`](docs/surgsync/loader_cookbook.md) |
| Unpack a release back to raw layout | `surgsync unpack …` | [`docs/HOW_to_RUN_unpack.md`](docs/HOW_to_RUN_unpack.md) |
| Emit README + CHANGELOG into a release | `surgsync release …` | This README + section below |
| Extend the format (new modality / column) | — | [`docs/surgsync/extension_policy.md`](docs/surgsync/extension_policy.md) |
| **Single-page runbook for all of the above** | — | [`docs/HOW_to_RUN.md`](docs/HOW_to_RUN.md) |

Everything is one installable package, `dvrk_data_processing`, with a single
entry-point binary, `surgsync`.

---

## Pipeline at a glance

```
            ┌─────────────────────────────┐
            │   raw clip on disk          │
            │   data/<dataset>/<idx>/     │
            │     image/  kinematic/      │
            │     annotation/  time_syn/  │
            │     camera_calibration/     │
            │     hand_eye_calibration/   │
            │     meta_data.json          │
            └──────────────┬──────────────┘
                           │  Preprocessing — scripts/run_all_stages.py
                           │  (rectify+resize, kinematic reproject,
                           │   depth, optical flow)
                           ▼
            ┌─────────────────────────────┐
            │   data/<dataset>/<idx>/     │
            │     preprocess/             │
            │       rectify_resize/       │
            │       kinematic_reproject/  │
            │       depth_estimation/     │
            │       optical_flow/         │
            └──────────────┬──────────────┘
                           │  Packing — surgsync build
                           │  (parquet + FFV1 + H.264 + meta)
                           ▼
            ┌─────────────────────────────┐
            │  SurgSync release           │
            │  <root>/                    │
            │    meta/                    │
            │    <dataset>/episodes/      │
            │      <task>/<idx>/          │
            │        *.parquet  video/    │
            │        video_raw/           │
            │        preprocess/          │
            │        calibration/         │
            │        episode_meta.json    │
            └─────────┬──────────┬────────┘
                      │          │
   surgsync.open_*    │          │  surgsync unpack
   (reader API)       │          │  (inverse — bit-exact round-trip)
                      ▼          ▼
              PyTorch /       raw-style tree on disk
              VLA training    (image/, kinematic/, annotation/,
                              calibration/, meta_data.json,
                              time_syn/, preprocess/)
```

**Four stages — preprocessing, packing, unpacking, cleanup — all
complete and tested.**

---

## Install

> 📌 Clone with submodules — the FoundationStereo backbone for depth
> estimation is a git submodule.

```bash
git clone --recursive https://github.com/jackzhy96/dvrk_multimodal_data_collection.git
cd dvrk_multimodal_data_collection
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Build the conda env (Python 3.9 + the full processing/pack/unpack stack)
and install the package in editable mode:

```bash
bash env_setup/create_env.sh        # ~10–15 min (flash-attn compile)
conda activate dvrk_multimodal_process
pip install -e .
```

The package on disk is `dvrk_data_processing`; the project name is
`dvrk_multimodal_data_collection`; the CLI binary registered by
`pyproject.toml` is `surgsync`.

For full install notes (FoundationStereo weights, troubleshooting,
ffmpeg version, GPU prerequisites), see [`env_setup/INSTALL.md`](env_setup/INSTALL.md).

---

## Raw clip layout

The pipeline operates on raw clips placed under `data/`. Each clip is a
self-contained recording in the canonical layout below — e.g.
`data/online_data/2/` (online recorder, has Cartesian `setpoint_cp`) or
`data/offline_data/3/` (offline recorder, no `setpoint_cp`, jaw nested
inside `arm`):

```
<clip>/
├── image/{left,right,side}/<i>.png      # 1080p PNG
├── kinematic/{ECM,PSM1,PSM2}/<i>.json   # per-frame
├── annotation/{contact_detection,phase,step,gesture}/<i>.json
├── time_syn/<i>.json                    # per-modality timestamps
├── camera_calibration/{left.yaml,right.yaml,stereo_calib_params.json}
├── hand_eye_calibration/<arm>-registration-<dVRK|open-cv>.json
└── meta_data.json
```

---

## Quick start

After installing, the shortest end-to-end demo is:

```bash
# 1. Run preprocessing on a raw clip (here online_data/2; CPU + GPU stages).
python scripts/run_all_stages.py \
    path_config=jack_local_release \
    clips.source=list +clips.list='[online_data/2]'

# 2. Pack it into a SurgSync release.
surgsync build \
    clips.source=list \
    +clips.list='[online_data/2]' \
    path_config.dataset_root=/tmp/my_release

# 3. Open it in Python.
python -c "
import dvrk_data_processing.surgsync as ss
ds = ss.open_dataset('/tmp/my_release')
ep = ss.open_episode(ds.episodes[0].path)
print(ep, ep.annotation.column('phase')[0].as_py())
"

# 4. Unpack it back to the raw-style tree.
surgsync unpack /tmp/my_release --out /tmp/my_unpack

# 5. Stamp release docs.
surgsync release /tmp/my_release --bump-version=minor \
    --notes 'Initial v1.0 release.'
```

For the verbose operator walkthroughs see the `HOW_to_RUN_*.md`
files under [`docs/`](docs/).

---

## Stage-by-stage operator guide

### Preprocessing — Process raw clips

The preprocessing pipeline turns the raw `image/` + `kinematic/` tree
into the derived modalities the packer consumes. Every preprocessing
output lives under `<clip>/preprocess/<stage>/` so the raw tree stays
untouched.

| Stage | Script | Output |
|---|---|---|
| 1. Rectify + resize | `gen_rectify_resize.py` | `preprocess/rectify_resize/image/{left,right}/*.png` + scaled calibration |
| 2. Kinematic reproject | `gen_kinematic_heatmap_handeye.py` | `preprocess/kinematic_reproject/<PSM>/<cam>/{image,heatmap}/*.png` + calibrated 6-DoF JSONs |
| 3. Depth estimation (GPU) | `gen_depth_estimate.py` | `preprocess/depth_estimation/{depth,disparity,depth_image,disparity_image,combined_image}/` |
| 4. Optical flow — RAFT (GPU) | `gen_optical_flow_raft.py` | `preprocess/optical_flow/<cam>/{optical_flow,image}/*.{npy,png}` |

The four stages share the same Hydra config layout
(`config/preprocess/<stage>.yaml` + `config/path_config/<name>_local*.yaml`)
and can be invoked individually or all at once:

```bash
python scripts/run_all_stages.py path_config=jack_local_release \
    clips.source=dataset clips.dataset_name=online_data
```

Full walkthrough — config keys, per-stage flags, GPU prerequisites —
in [`docs/HOW_to_RUN_Process.md`](docs/HOW_to_RUN_Process.md).

### Packing — Pack to SurgSync

`surgsync build` reads the preprocessing outputs (plus the raw
`image/`, `kinematic/`, `annotation/`, `meta_data.json`) and writes
the canonical SurgSync layout.

```bash
surgsync build clips.source=all parallelism=4 \
    path_config.dataset_root=/path/to/output/release
```

What you get under `<release>/<dataset>/episodes/<task>/<idx>/`:

| Artifact | Codec | Fidelity |
|---|---|---|
| `video_raw/*.mkv` | FFV1 | bit-exact raw frames (mandatory for unpack) |
| `video/*.mp4` | H.264 CRF 18 | visually lossless rectified frames (PSNR ≥ 40 dB) |
| `preprocess/{depth,flow_*,heatmap_*}.mkv` | FFV1 bgr24/gray8 | bit-exact preprocessing viz frames |
| `timestamp.parquet` | parquet zstd | master clock + per-topic deltas |
| `{ECM,PSM1,PSM2}.parquet` | parquet zstd | every kinematic field |
| `annotation.parquet` | parquet zstd | phase/step/gesture **verbalized to text** |
| `calibration/*.{yaml,json}` | verbatim | byte-exact raw camera + hand-eye files |
| `episode_meta.json` + `modalities.json` + `time_sync_stat.json` | JSON | typed metadata |

Plus a top-level `meta/` directory with `dataset.json`, `tasks.jsonl`,
Hive-partitioned `episodes.parquet` / `index.parquet`, `stats.parquet`,
and SHA-256 `manifest.json`.

Full guide: [`docs/HOW_to_RUN_pack.md`](docs/HOW_to_RUN_pack.md).

### Unpacking — Read or unpack a SurgSync dataset

Two consumption paths share the same Python API.

**Read in place** (training, analysis, ad-hoc inspection):

```python
import dvrk_data_processing.surgsync as ss

ds = ss.open_dataset("/path/to/release")
for ep_ref in ds.filter(task="single_interrupted_stitch"):
    ep = ss.open_episode(ep_ref.path)
    print(ep.episode_id, ep.task, ep.length)
    pos = ep.psm1.column("measured_js.position")[0].as_py()
    phase = ep.annotation.column("phase")[0].as_py()  # verbalized text
    for frame in ep.video_raw("stereo_left").iter_frames():
        ...  # (H, W, 3) uint8 BGR
        break
    ep.close()
```

Cookbook with full worked examples (filtering, video decode, PyTorch
DataLoader): [`docs/surgsync/loader_cookbook.md`](docs/surgsync/loader_cookbook.md).

**Unpack back to disk** (tools that expect the original raw layout):

```bash
surgsync unpack /path/to/release \
    --out /path/to/unpacked \
    --parallelism 4
```

What you get: the pre-pack on-disk tree, with images pixel-bit-exact,
calibration byte-exact, kinematic JSONs numerically equivalent, and
annotation JSONs carrying the **text** descriptions (the packer
verbalized ids via `workflow_description/workflow_description.json`
at pack time).

Resume-friendly via `.surgsync_unpacked.json` sentinel; collision
detection if two task folders share the same clip index; bounded
PNG-encoder memory so very large clips don't OOM.

Full guide: [`docs/HOW_to_RUN_unpack.md`](docs/HOW_to_RUN_unpack.md).

### Cleanup — Release docs

```bash
surgsync release /path/to/release \
    --bump-version=minor \
    --notes 'Added 12 new suturing clips; re-encoded after re-calibration.'
```

Writes `README.md` + `CHANGELOG.md` into the release root populated
from `meta/dataset.json` + per-episode counts, and bumps
`meta/dataset.json:data_version` per semver (`patch` / `minor` /
`major`). The CHANGELOG appends; the README is a fresh render each
time.

---

## Reader API cheat sheet

Every public symbol — also exported from `surgsync` at the top level:

```python
import dvrk_data_processing.surgsync as ss

ss.open_dataset(path)            # → Dataset
ss.open_episode(path)            # → Episode
ss.decompose(dataset_root, out)  # → DecomposeReport
ss.SCHEMA_VERSION                # str

# Classes:
ss.Dataset, ss.Episode, ss.VideoView
ss.DecomposeReport, ss.DecomposedClipReport
```

`Episode` carries lazy properties:
`meta`, `episode_id`, `task`, `length`, `master_t0_ns`,
`recorder_variant`, `source_clip`, `modalities`, `time_sync_stat`,
`timestamps` (parquet), `ecm` / `psm1` / `psm2` / `annotation`
(parquets), `video(name)` / `video_raw(name)` / `preprocess(name)`
(returning `VideoView`), `calibration` (`CalibrationBundle`).

Every parquet is loaded on first access and cached on the instance.
Video frames stream via ffmpeg subprocesses; bounded memory regardless
of clip length.

---

## Repo layout

```
.
├── src/dvrk_data_processing/
│   ├── raw_image_processing/        # preprocessing stage 1 — rectify + resize
│   ├── kinematic_mapping/           # preprocessing stage 2 — heatmaps + calibrated kinematics
│   ├── depth_estimation/            # preprocessing stage 3 — FoundationStereo (GPU)
│   ├── optical_flow/                # preprocessing stage 4 — RAFT / Farneback
│   ├── data_annotation/             # PyQt5 GUIs for phase / event / contact annotation
│   ├── utils/                       # Hydra dataclass schemas + filesystem helpers
│   └── surgsync/                    # packer + reader + decomposer + release docs
│       ├── schema/                  #   Arrow + pydantic — single source of truth
│       ├── serde/                   #   JSON ↔ record converters
│       ├── ingest/                  #   raw → typed records (pack forward)
│       ├── align/                   #   master clock + per-topic delta computation
│       ├── encode/                  #   typed records → on-disk artifacts (pack)
│       ├── validate/                #   raw-clip / episode / dataset validators
│       ├── pipeline/                #   per-clip + per-release orchestration + release docs
│       ├── index/                   #   meta/ index builders (episodes, frames, stats, manifest)
│       ├── load/                    #   open_dataset, open_episode, video decode (unpack read)
│       ├── decompose/               #   packed → raw-style tree (unpack)
│       └── cli.py                   #   build, validate, index, unpack, release, selftest
│
├── config/
│   ├── path_config/                 # per-user path mappings (data_dir, dataset_root)
│   ├── preprocess/                  # per-preprocessing-stage hyperparameter configs
│   ├── surgsync/                    # SurgSync build/encode/align configs
│   ├── config_<op>_<user>.yaml      # preprocessing running configs (one per op × user)
│   └── run_all_stages.yaml          # preprocessing batch orchestrator
│
├── scripts/
│   ├── run_all_stages.py            # preprocessing batch entry point
│   ├── unpack_breakdown.py          # unpack runtime + size analyzer
│   ├── verify_unpack_vs_raw.py      # unpack round-trip diff
│   ├── data_reorg.py, data_remap.py, data_trim.py, …
│   └── (one-off conversion / inspection utilities)
│
├── docs/                            # all documentation
│   ├── HOW_to_RUN.md                # single-page runbook
│   ├── HOW_to_RUN_Process.md  HOW_to_RUN_pack.md  HOW_to_RUN_unpack.md   # operator guides
│   └── surgsync/                    # consumer docs: loader_cookbook.md, extension_policy.md
│
├── tests/surgsync/                  # 150+ pytest tests
│   ├── schema/  serde/  ingest/  align/  encode/  index/  validate/
│   ├── pipeline/                    # release docs generator
│   ├── decompose/                   # unpack round-trip
│   └── integration/                 # end-to-end pack → unpack smoke
│
├── workflow_description/            # phase / step / gesture vocab JSONs
├── FoundationStereo/                # git submodule (depth backbone)
├── env_setup/                       # create_env.sh + INSTALL.md
├── data/                            # raw clips you supply (gitignored)
├── assets/                          # endoscope-holder CAD (SolidWorks + STL)
├── dvrk_config/                     # dVRK console JSONs + capacitive contact-sensor design
├── replay/  rosbag_record/  video_launch/   (data-collection-time tooling — see "Hardware" below)
├── pyproject.toml  LICENSE  README.md
```

---

## Hardware & data-collection tooling

The dataset was captured on a physical dVRK. The hardware designs and
collection-time launch files used to produce the raw clips ship alongside the
processing code so the capture rig can be reproduced.

- **Endoscope holder (CAD)** — [`assets/endoscope_holder_CAD/`](assets/endoscope_holder_CAD/):
  SolidWorks parts (`.SLDPRT`) and print-ready `.STL` for the custom endoscope
  holder, adapter, and cannula used to mount the stereo endoscope.
- **Capacitive contact sensor** — [`dvrk_config/contact_sensor/`](dvrk_config/contact_sensor/):
  a simple Arduino-based capacitive contact-detection system (one sensor per
  PSM) that produces the `contact_detection` annotation channel. Includes the
  Arduino sketch (`.ino`), a wiring + threshold-tuning guide
  ([`README.md`](dvrk_config/contact_sensor/README.md)), and the dVRK
  `sawRobotIO1394` digital-input XML.
- **dVRK console configs** — [`dvrk_config/`](dvrk_config/): the console / SUJ
  system JSON files for the PSM / ECM / MTM setup used during collection.
- **Video capture launch** — [`video_launch/`](video_launch/): `gscam` V4L2
  launch files for the endoscope feed. They drive stereo capture through the
  dVRK video stack — see [jhu-dvrk/dvrk_video](https://github.com/jhu-dvrk/dvrk_video).
- **Replay / recording** — [`replay/`](replay/) and
  [`rosbag_record/`](rosbag_record/): trajectory replay and ROS bag recording
  helpers used at collection time.

---

## Tests

The full pytest suite lives under `tests/surgsync/`. To run it:

```bash
conda activate dvrk_multimodal_process
pytest tests/surgsync/
```

Counts as of the latest pass:

| Category | Count | Runtime |
|---|---:|---:|
| Schema, serde, ingest, align, encode, index, validate (packing) | 121 | ~6 min |
| Decompose round-trip (unpacking) | 7 | ~5 min |
| Release docs generator (cleanup) | 21 | <1 s |
| End-to-end pack → unpack smoke | 1 | ~13 min |
| **Total** | **150** | **~25 min** |

Smoke + integration tests need a raw sample clip at
`data/online_data/2/`; they're auto-skipped if absent.

---

## Further reading

- [`docs/HOW_to_RUN.md`](docs/HOW_to_RUN.md) — single-page runbook
  for the whole pipeline.
- [`docs/HOW_to_RUN_Process.md`](docs/HOW_to_RUN_Process.md) — preprocessing
  operator guide.
- [`docs/HOW_to_RUN_pack.md`](docs/HOW_to_RUN_pack.md) — packing operator
  guide.
- [`docs/HOW_to_RUN_unpack.md`](docs/HOW_to_RUN_unpack.md) — unpacking
  operator guide + fidelity table.
- [`docs/surgsync/loader_cookbook.md`](docs/surgsync/loader_cookbook.md) — Python
  reader API worked examples.
- [`docs/surgsync/extension_policy.md`](docs/surgsync/extension_policy.md) — how
  to add a new modality, column, or stream.
- [`env_setup/INSTALL.md`](env_setup/INSTALL.md) — install troubleshooting.

External:
- [FoundationStereo](https://github.com/NVlabs/FoundationStereo) — the
  depth-estimation backbone (Wen et al., CVPR 2025).
- [RAFT](https://github.com/princeton-vl/RAFT) — the optical-flow
  backbone (Teed & Deng, ECCV 2020).

---

## License + contact

License: see [`LICENSE`](LICENSE).

Maintainer: **Haoying (Jack) Zhou** — hzhou62@jh.edu / hzhou6@wpi.edu ·
[github.com/jackzhy96](https://github.com/jackzhy96). Open an issue on
the repo for bug reports, feature requests, or questions about the
data format.
