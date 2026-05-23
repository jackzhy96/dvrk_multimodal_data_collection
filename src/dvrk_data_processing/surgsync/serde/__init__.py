"""Source-of-truth JSON ↔ in-memory record converters.

Both the packer (ingest, encode/episode_meta) and the unpacker
(decompose) route their JSON serialization through these modules so
the on-disk schemas stay consistent across forward and inverse
directions.

Modules:
  kinematic_io  — per-frame kinematic JSON ↔ records
  annotation_io — per-frame annotation JSON ↔ records
  meta_io       — meta_data.json ↔ ClipMeta record / episode_meta.json mapping
"""
