"""SurgSync encoders — every module here writes one artifact under one
finalized episode directory.

Module layout:
- `codec.py` is the single ffmpeg shell-out point (no other module
  shells out to ffmpeg directly).
- `video_processed.py` / `video_raw.py` / `preprocess.py` / `preview.py`
  encode video streams.
- `frames_parquet.py` writes the per-episode parquet via the schema
  built in `surgsync.schema`.
- `episode_meta.py` writes `episode_meta.json` via `surgsync.serde.meta_io`.
- `calibration.py` writes the per-episode `calibration/` folder.
"""
