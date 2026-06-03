# SurgSync loader cookbook

Worked examples for the reader API. Every snippet runs against a
real packed dataset — adjust the path and `task` literal to match
your release.

For the operator-facing pack / unpack guides, see
[`HOW_to_RUN_pack.md`](../HOW_to_RUN_pack.md) and
[`HOW_to_RUN_unpack.md`](../HOW_to_RUN_unpack.md).

---

## 0. Imports + open a dataset

Every example assumes:

```python
from pathlib import Path
import dvrk_data_processing.surgsync as surgsync

DATASET_ROOT = Path("/media/jackzhy/Extreme SSD/surgsync_release")
ds = surgsync.open_dataset(DATASET_ROOT)
print(ds)
# Dataset(root='/media/jackzhy/Extreme SSD/surgsync_release', episodes=2, tasks=['single_interrupted_stitch'])
```

The constructor walks `<root>/<dataset>/episodes/<task>/<clip>/` once
and skips anything missing the `.surgsync_complete.json` sentinel.
Parquets are not loaded; per-episode opens are lazy.

---

## 1. List + filter episodes

### Every episode

```python
for ep_ref in ds.episodes:
    print(ep_ref.dataset_name, ep_ref.task, ep_ref.clip_index)
```

### Filter by task

```python
suturing = ds.filter(task="single_interrupted_stitch")
print(f"{len(suturing)} suturing episodes")
```

### Filter by task + operator skill level

`skill_level` lives on `episode_meta.json`, not on the path. So we
filter by task first (cheap, path-based) then open each candidate to
check skill:

```python
intermediate_suturing = []
for ep_ref in ds.filter(task="single_interrupted_stitch"):
    ep = surgsync.open_episode(ep_ref.path)
    try:
        if ep.meta.get("operator_skill_level") == "Intermediate":
            intermediate_suturing.append(ep_ref)
    finally:
        ep.close()
print(f"{len(intermediate_suturing)} intermediate-skill suturing episodes")
```

### Filter by tool

`tool` is a nested dict in `episode_meta.json` (`{"PSM1": "...", "PSM2": "..."}`).
Use the same open-then-check pattern:

```python
needle_drivers = []
for ep_ref in ds.episodes:
    ep = surgsync.open_episode(ep_ref.path)
    try:
        if ep.meta.get("tool", {}).get("PSM1") == "Large_Needle_Driver":
            needle_drivers.append(ep_ref)
    finally:
        ep.close()
```

---

## 2. Open one episode + walk per-frame

```python
ep = surgsync.open_episode(suturing[0].path)
print(ep)
# Episode('online_data_2_1754609707325800839', task='single_interrupted_stitch', frames=886)

# Iterate frame indices.
for i in range(ep.length):
    # PSM1 joint position (6-vector) at frame i.
    pos = ep.psm1.column("measured_js.position")[i].as_py()
    # Phase verbalized text (already english at pack time).
    phase = ep.annotation.column("phase")[i].as_py()
    # Master clock (clip-relative ns).
    t_ns  = ep.timestamps.column("master_timestamp_ns")[i].as_py()
    if i == 0:
        print(f"frame {i}: t={t_ns} ns  phase={phase!r}  PSM1={pos}")
    if i == 3:
        break

ep.close()
```

Parquets are cached on the `Episode` instance the first time you
touch them. Repeated access is free.

---

## 3. Decode video frames lazily

Three video sources per episode:

- `ep.video_raw('stereo_left' | 'stereo_right' | 'side')` — FFV1
  bit-exact raw camera frames.
- `ep.video('stereo_left' | 'stereo_right')` — H.264 visually-lossless
  rectified frames (only when preprocessing ran).
- `ep.preprocess('depth' | 'flow_left' | 'flow_right' |
  'heatmap_PSM1_left' | …)` — FFV1 bit-exact visualization frames
  from preprocessing outputs.

Each returns a `VideoView | None`. Iterate frames sequentially via
`view.iter_frames()` — one ffmpeg subprocess per call, no random
access.

```python
ep = surgsync.open_episode(suturing[0].path)
view = ep.video_raw("stereo_left")
if view is None:
    raise RuntimeError("stereo_left raw video not present")

for i, frame in enumerate(view.iter_frames()):
    # frame: np.ndarray, shape (H, W, 3), dtype uint8, BGR order.
    print(f"frame {i}: shape={frame.shape} dtype={frame.dtype} "
          f"mean={frame.mean():.1f}")
    if i == 2:
        break
ep.close()
```

### Probe video shape without decoding

```python
view = ep.video_raw("stereo_left")
info = view.probe()
# {'width': 1920, 'height': 1080, 'pix_fmt': 'bgr0',
#  'codec_name': 'ffv1', 'n_frames': 0}    # n_frames=0 unless count_frames=True
print(info)
```

### Decode all preprocess streams

```python
ep = surgsync.open_episode(suturing[0].path)
for stream in ep.available_videos()["preprocess"]:
    view = ep.preprocess(stream)
    first = next(view.iter_frames())
    print(stream, "first frame shape:", first.shape)
ep.close()
```

---

## 4. Annotation text + structured-vocab lookup

The packed `annotation.parquet` already verbalizes phase / step /
gesture cells to text. Bare ids are not preserved by design; if you
need the id → text mapping, read it from `meta/tasks.jsonl`.

```python
ep = surgsync.open_episode(suturing[0].path)
ann = ep.annotation
for i in range(3):
    print({
        "phase":        ann.column("phase")[i].as_py(),
        "step":         ann.column("step")[i].as_py(),
        "gesture.PSM1": ann.column("gesture.PSM1")[i].as_py(),
        "contact.PSM1": ann.column("contact.PSM1")[i].as_py(),
    })
ep.close()
```

### Use the task vocab table directly

```python
# Dataset-level vocab table (one row per task).
for row in ds.task_vocab:
    print(row["task"], "→ phase:",
          row["phase_description"][:60], "...")
    # row["step_vocab"]:    {<id>: <text>}
    # row["gesture_vocab"]: {<id>: <text>}
```

---

## 5. Build a per-task PyTorch DataLoader

The packer ships per-modality parquets; there's no built-in
`SurgSyncTorchDataset` yet. The two canonical patterns below show
how to assemble one in ~30 lines.

### Pattern A — random-access from one episode

Best for short clips that fit comfortably in RAM. The parquets stay
on disk; only the row(s) you ask for are materialized.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class OneEpisode(Dataset):
    """Minimal per-frame access; image is decoded lazily on __getitem__."""

    def __init__(self, episode_path: Path, image_stream: str = "stereo_left"):
        self.ep = surgsync.open_episode(episode_path)
        # Pre-collect every frame once. Each entry is (np.ndarray, BGR).
        view = self.ep.video_raw(image_stream)
        self._frames = list(view.iter_frames())   # cache full clip
        self._psm1   = self.ep.psm1
        self._ann    = self.ep.annotation

    def __len__(self):
        return self.ep.length

    def __getitem__(self, i):
        frame = self._frames[i]
        # NHW C → C H W and BGR → RGB for the typical torchvision pipeline.
        img = torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1)
        return {
            "image":  img,                                                # uint8 (3, H, W)
            "action": torch.tensor(
                self._psm1.column("setpoint_js.position")[i].as_py(),
                dtype=torch.float32,
            ),
            "language": self._ann.column("phase")[i].as_py() or "",
        }


ep_ref = ds.filter(task="single_interrupted_stitch")[0]
batch = next(iter(DataLoader(OneEpisode(ep_ref.path), batch_size=4)))
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in batch.items()})
```

### Pattern B — streaming across many episodes

Best for large datasets. Builds the `(episode_path, frame_idx)`
index once at init; opens each episode on demand. Add an LRU around
`open_episode` if you re-touch the same episode across many `__getitem__`
calls (cheap parquets + ffmpeg subprocess per `iter_frames` makes
plain dict caching effective).

```python
from functools import lru_cache

class StreamingSurgSync(Dataset):
    def __init__(self, ds: surgsync.Dataset, task: str):
        self.ds = ds
        self.refs = ds.filter(task=task)
        # Build a flat (ep_ref, frame_idx) index.
        self.index = []
        for r in self.refs:
            ep = surgsync.open_episode(r.path)
            try:
                for i in range(ep.length):
                    self.index.append((r.path, i))
            finally:
                ep.close()

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=4)
    def _open(self, path):
        return surgsync.open_episode(path)

    def __getitem__(self, k):
        path, i = self.index[k]
        ep = self._open(path)
        # NOTE: video_raw().iter_frames() is sequential; random access
        # by frame i would require seeking via ffprobe + -ss filters.
        # For training loops that randomly sample frames, pre-extract
        # PNGs (or use `surgsync unpack`) instead of decoding on demand.
        return {
            "action": torch.tensor(
                ep.psm1.column("setpoint_js.position")[i].as_py(),
                dtype=torch.float32,
            ),
        }


dataset = StreamingSurgSync(ds, task="single_interrupted_stitch")
loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
for batch in loader:
    print(batch["action"].shape)
    break
```

Caveat: `num_workers > 0` forks. The forked process gets a fresh
file-handle cache; `lru_cache` survives because it's per-process.
Don't share live ffmpeg subprocesses across forks.

---

## 6. Decompose back to raw layout

If your downstream tool expects the pre-pack on-disk shape, use unpack:

```python
report = surgsync.decompose(
    dataset_root=DATASET_ROOT,
    out_root=Path("/tmp/unpacked"),
    clips=["online_data/2"],
    streams=("raw", "preprocess"),
    force=True,
    parallelism=2,
)
print(f"unpacked {report.n_episodes_ok}/{report.n_episodes_seen} clips")
# Re-running with `force=False` skips clips whose output is sentinel'd:
report = surgsync.decompose(DATASET_ROOT, Path("/tmp/unpacked"),
                            clips=["online_data/2"], force=False)
assert report.n_episodes_skipped == 1
```

After decompose, the tree at `/tmp/unpacked/online_data/2/` matches
the original raw `data/online_data/2/` layout, with:

- bit-exact `image/{left,right,side}/*.png`
- text-verbalized `annotation/{phase,step,gesture}/*.json` (matching
  the parquet; the numeric id is not preserved)
- reconstructed `time_syn/*.json`
- byte-exact `camera_calibration/`, `hand_eye_calibration/`, `meta_data.json`

See [`HOW_to_RUN_unpack.md`](../HOW_to_RUN_unpack.md) for the full
fidelity table and the resume / collision-detection semantics.

---

## 7. End-of-cookbook: generate release docs

After building or extending a dataset:

```bash
surgsync release /path/to/dataset_root \
    --bump-version=patch \
    --notes "Re-encoded clip 42 after re-calibration."
```

This writes `README.md` + `CHANGELOG.md` into the dataset root
populated from `meta/dataset.json` + the per-episode metadata. See
the source at
[`src/dvrk_data_processing/surgsync/pipeline/release.py`](../../src/dvrk_data_processing/surgsync/pipeline/release.py).
