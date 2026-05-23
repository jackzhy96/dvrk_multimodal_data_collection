"""pydantic model for `meta/manifest.json` (`code_design.md` § 3.4).

SHA256 manifest for bit-rot detection. Written last by the indexing
stage so it covers every other file under `<dataset_root>` except itself.
"""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha256: str
    size_bytes: int


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    generated_at_utc: str
    data_version: str
    algorithm: str = "sha256"
    files: dict[str, ManifestFile]
    total_files: int
    total_size_bytes: int
