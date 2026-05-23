# SurgSync extension policy

How to evolve the dataset format without breaking consumers. This is
the H-4 risk closure from `architecture_risks.md`: changes ship
through documented patterns rather than ad-hoc edits to packer code.

Scope: schema fields, modality additions, dense streams, per-episode
metadata, and `schema_version` bumps. For the architectural
contracts these extensions rest on, read
[`code_design.md`](../../specs/code_design.md) and
[`final_data_spec.md`](../../specs/final_data_spec.md) first.

---

## TL;DR — the rules

1. **`schema_version` is semver.** Bump *major* on a breaking change
   (consumers using the old schema break), *minor* on an additive
   change (old consumers still load fine), *patch* on a doc-only fix.
2. **Schemas live in one place per artifact.** Arrow / pydantic schemas
   under `src/dvrk_data_processing/surgsync/schema/`. JSON
   forward+inverse converters under `surgsync/serde/`. Encoders
   import from there — never duplicate column lists inline.
3. **Every new on-disk artifact has a forward AND inverse direction.**
   The packer writes it; the decomposer recovers it. Without both,
   the invertibility contract weakens silently.
4. **Every new field is documented in `meta/dataset.json` conventions
   when consumer-visible.** The README generator picks it up.

---

## Adding a new modality (audio, force sensor, eye tracking)

Use case: you record a microphone alongside the cameras and want it
in the packed dataset.

### Steps

1. **Spec the on-disk raw form** in `specs/raw_data_spec.md` —
   filename pattern, sample rate, file format. Submit as a PR
   before writing code so reviewers can debate the shape.

2. **Add a new ingest module** under `src/dvrk_data_processing/surgsync/ingest/`:
   ```python
   # ingest/audio.py
   def load_audio(clip: RawClip) -> AudioTrack: ...
   ```
   Read-only; mirrors the pattern of `ingest/kinematics.py`.

3. **Add a serde module** if the modality is per-frame JSON or
   carries content that the inverse direction needs to reproduce:
   ```python
   # serde/audio_io.py
   def audio_to_table(track: AudioTrack) -> pa.Table: ...
   def table_to_audio_files(table: pa.Table, dst: Path) -> None: ...
   ```

4. **Add an encoder** under `src/dvrk_data_processing/surgsync/encode/`:
   ```python
   # encode/audio.py
   def write_audio(track: AudioTrack, dst: Path, *, fps: float) -> int: ...
   ```
   Always emit under `<episode>/<modality_name>/...`. For audio: maybe
   `<episode>/audio/track.flac`.

5. **Add a decoder hook to `Episode`** in `src/dvrk_data_processing/surgsync/load/episode.py`:
   ```python
   def audio(self) -> Optional[AudioTrack]:
       p = self.path / "audio" / "track.flac"
       return AudioTrack(p) if p.is_file() else None
   ```

6. **Add a decompose writer** under
   `src/dvrk_data_processing/surgsync/decompose/`:
   ```python
   # decompose/audio.py — call from decompose/raw.py or new domain
   def write_audio_domain(episode, out_clip_dir): ...
   ```

7. **Extend `meta/dataset.json` conventions** with anything
   consumer-visible (sample rate, channel layout). The
   `Modalities` pydantic model gains an optional `audio: list[str]`
   field; add it with `Field(default_factory=list)` so existing
   datasets parse unchanged.

8. **Bump `schema_version` minor.** Old consumers still load; new
   consumers gain the field.

9. **Tests** under `tests/surgsync/`:
   - `tests/surgsync/ingest/test_audio.py` — read a fixture file.
   - `tests/surgsync/serde/test_audio_io.py` — forward+inverse round-trip
     byte-equivalent.
   - `tests/surgsync/encode/test_audio.py` — encoded file is decodable.
   - `tests/surgsync/decompose/test_audio.py` — decompose recovers
     the raw form.

### What NOT to do

- Don't shoehorn the new modality into an existing parquet (e.g.
  adding audio columns into `annotation.parquet`). That's a schema
  major bump; the modality split is the simpler change.
- Don't write directly to disk from a notebook. Every artifact goes
  through the `encode/` layer so validators + manifest hashing
  cover it automatically.

---

## Adding per-frame columns (`observation.*`, `state.*` extensions)

Use case: add `state.PSM1.ee_force` (a 6-vector) recorded from a wrist
force sensor.

### Steps

1. **Extend the Arrow schema** in `src/dvrk_data_processing/surgsync/schema/psm.py`:
   ```python
   pa.field("ee_force", pa.list_(pa.float32()), nullable=True),
   ```
   Always `nullable=True` for new fields — old episodes have no such
   column, and we never want to forbid older builds from loading.

2. **Extend the serde converter** in `surgsync/serde/kinematic_io.py`:
   add an `ee_force: Optional[list[float]]` field on `KinematicSample`
   and round-trip it on both `load_arm_frame_json` and
   `kinematic_sample_to_raw_dict`.

3. **Extend the parquet writer** in
   `surgsync/encode/per_modality_parquet.py` to populate the column
   from `sample.ee_force`.

4. **Extend the decompose reader** in `surgsync/decompose/raw.py`
   `_row_to_kinematic_sample` to pull the new column from the
   parquet row dict.

5. **Bump `schema_version` minor.** The new column has a default
   value (NULL) so old consumers ignore it cleanly.

6. **Migration note in CHANGELOG**: explain when the column starts
   being populated. Old builds without the column are still valid.

### Trade-off

If the new column is huge per row (e.g. an embedding vector), consider
a sidecar parquet instead of adding to PSM1/PSM2. The packer doesn't
care; consumers pay only for the columns they read.

---

## Adding per-frame dense streams (new video, new geometry)

Use case: add a thermal camera; consumers want lossy color video for
viewing and lossless raw frames for analysis.

### Steps

1. **Pick the codec contract**:
   - Lossy-but-visually-lossless (H.264 CRF 18): document the PSNR floor
     in the encoder docstring (the packer contract requires this).
   - Lossless (FFV1): bit-exact round-trip, larger file.

2. **Extend `encode/video_raw.py`** for the lossless side or
   `encode/video_processed.py` for the lossy side. Both call into
   `encode/codec.py`.

3. **Extend `Episode.video_raw()` / `.video()` / `.available_videos()`**
   in `load/episode.py` to expose the new stream name.

4. **Extend `decompose/raw.py` `write_raw_images()`** (or the
   preprocess writer if the new stream is derived) to decode and
   write PNGs back. Use `_bounded_png_pool` to keep memory flat.

5. **Add the stream to `meta/dataset.json.modalities.video`** in the
   `DatasetMeta` pydantic model defaults.

6. **Bump `schema_version` minor** if old consumers can still read
   the dataset (they ignore the unknown video stream). Bump *major*
   if the new stream is required (probably never; new streams are
   additive).

### Dense numerical streams (depth, flow as float arrays)

These were the original Option B encoding (16-bit FFV1 + scale/offset).
The v1 packer ships them as 8-bit visualization PNGs only — see the
docstring of `encode/preprocess.py` for the rationale. To revive
the 16-bit path:

- Re-add the per-frame scale/offset columns to `timestamp.parquet`
  (additive, minor bump).
- Add a `decode_geometry_frame()` helper in `load/codec_decode.py`
  that consumes the scale/offset to recover float32.
- Wire `Episode.geometry()` to return the decoded array.
- Extend `decompose/preprocess.py` to write `.npy` files alongside
  the existing PNG path.

---

## Adding per-episode metadata (`episode_meta.json` extension)

Use case: track which surgeon performed the procedure (a string
field).

### Steps

1. **Extend `schema/episode_meta.py:EpisodeMeta`** with the new
   field. Always `Optional[<type>] = None` for additive fields.

2. **Wire it through the packer** in
   `encode/episode_meta.py:write_episode_meta` — accept the new
   value as a kwarg, default to None, write it into the JSON.

3. **Wire it through the ingest path** if it comes from
   `meta_data.json`: extend `serde/meta_io.py:ClipMeta` so the
   forward direction pulls it, and have `pipeline/per_clip.py` pass
   it to `write_episode_meta`.

4. **Wire it through the decompose path** in
   `decompose/raw.py:write_calibration_and_meta` so the recovered
   `meta_data.json` carries the new field.

5. **Bump `schema_version` minor.**

6. **Update the README generator** (`pipeline/release.py`) if the
   field is interesting at the release-overview level (e.g. "skills
   represented in this release").

---

## Bumping `schema_version`

| Change | Bump |
|---|---|
| Rename a column. | **major** — old consumers crash. |
| Drop a column. | **major**. |
| Add a column with a default. | **minor**. |
| Add an optional field on a pydantic model. | **minor**. |
| Add a new modality (folder). | **minor** — old consumers ignore. |
| Tighten validation (an existing dataset now fails a check). | **major** — a former-valid dataset now invalid is a breaking change for ops. |
| Loosen validation. | **patch**. |
| Doc-only change. | **patch**. |

The `SCHEMA_VERSION` constant lives at
[`schema/__init__.py`](../../src/dvrk_data_processing/surgsync/schema/__init__.py).
The packer reads it on every build and stamps it into
`meta/dataset.json.schema_version` and every `episode_meta.json`.
Readers check that constant — minor bumps are accepted; major
bumps require the reader to update.

---

## Where each change shows up in the test suite

| Change | Tests to add / update |
|---|---|
| New ingest module | `tests/surgsync/ingest/test_<modality>.py` |
| New encoder | `tests/surgsync/encode/test_<modality>.py` |
| New serde converter | `tests/surgsync/serde/test_<modality>_io.py` (forward+inverse) |
| New `Episode` accessor | `tests/surgsync/decompose/test_decompose_smoke.py` smoke covers the lazy property |
| New decompose writer | `tests/surgsync/decompose/test_<modality>.py` |
| Schema bump | the existing end-to-end smoke test catches obvious regressions; add a targeted round-trip test for the new field |

---

## What lives where

```
src/dvrk_data_processing/surgsync/
├── schema/         # Arrow + pydantic — single source of truth per artifact
├── serde/          # JSON ↔ record converters; round-trip-tested
├── ingest/         # raw → in-memory typed records (read-only)
├── align/          # timestamp matching + master clock
├── encode/         # in-memory records → on-disk artifacts (pack)
├── pipeline/       # per_clip / per_release orchestration + release docs
├── validate/       # raw_clip + episode + dataset validators
├── load/           # `open_dataset`, `open_episode`, video decoders (read)
└── decompose/      # packed → on-disk raw layout (unpack)
```

Cross-imports rules (enforced informally):

- `ingest/` may import `schema/`, `serde/`. Never `encode/`, `pipeline/`.
- `encode/` may import `schema/`, `serde/`, `align/`. Never `ingest/`.
- `decompose/` may import `load/`, `serde/`. Never `encode/`, `ingest/`.
- `pipeline/` is the glue layer; it imports from everywhere.
