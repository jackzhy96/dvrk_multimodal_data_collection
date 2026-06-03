"""pydantic model for `meta/dataset.json`.

Top-level dataset metadata. One file per dataset root; loaded eagerly by
both readers and validators.
"""
from __future__ import annotations
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Modalities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video: list[str] = Field(default_factory=lambda: ["stereo_left", "stereo_right", "side"])
    # `preprocess` lists every stream that ships under `<episode>/preprocess/`
    # — depth, flow, heatmap visualizations etc. Matches the folder name
    # so the dataset.json modality list and the on-disk layout agree.
    preprocess: list[str] = Field(default_factory=lambda: [
        "depth", "flow_left", "flow_right",
        "heatmap_PSM1_left", "heatmap_PSM1_right",
        "heatmap_PSM2_left", "heatmap_PSM2_right",
    ])
    state: list[str] = Field(default_factory=lambda: ["ECM", "PSM1", "PSM2"])
    action: list[str] = Field(default_factory=lambda: ["PSM1", "PSM2"])
    annotation: list[str] = Field(default_factory=lambda: ["contact", "gesture", "phase", "step"])


class AlignmentTolMs(BaseModel):
    """Tolerance numbers exposed to consumers for transparency. `online`
    is a single float (strict); `offline` is a dict-of-overrides because
    different modalities use different tolerances."""
    model_config = ConfigDict(extra="forbid")
    online: float = 2.0
    offline: dict[str, Union[float, str]] = Field(default_factory=lambda: {
        "image_side": 33.0,
        "kinematic": "1000/source_frequency_hz",
    })


class Conventions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    master_clock: str = "stereo_left_capture_ros_header_stamp"
    alignment_policy: str = "nearest_neighbor_within_tolerance"
    alignment_tol_ms: AlignmentTolMs = Field(default_factory=AlignmentTolMs)
    frame_index_basis: str = "master_clock"
    quaternion_order: str = "xyzw"
    length_unit: str = "m"
    angle_unit: str = "rad"
    image_size: list[int] = Field(default_factory=lambda: [512, 288])
    bimanual_action_concat: bool = False
    image_normalization: str = "imagenet"


class PipelineVersions(BaseModel):
    """Same shape as the per-episode block, kept separate so dataset.json
    and episode_meta.json can be migrated independently."""
    model_config = ConfigDict(extra="forbid")
    rectify_resize: Optional[str] = None
    kinematic_handeye: Optional[str] = None
    depth_estimation: Optional[str] = None
    optical_flow_raft: Optional[str] = None


class DatasetMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "SurgSync"
    schema_version: str
    data_version: str
    release_option: str   # "A" | "B" | "C"
    created_at_utc: str

    modalities: Modalities = Field(default_factory=Modalities)
    conventions: Conventions = Field(default_factory=Conventions)
    pipeline_versions: PipelineVersions = Field(default_factory=PipelineVersions)

    tasks: list[str] = Field(default_factory=list)
    tasks_jsonl_path: str = "meta/tasks.jsonl"
    episodes_index_path: str = "meta/episodes.parquet"
    frames_index_path: str = "meta/index.parquet"
    manifest_path: str = "meta/manifest.json"
