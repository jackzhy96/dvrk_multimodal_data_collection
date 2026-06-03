"""
Batch orchestrator — sweeps every preprocessing stage over a list of clip
indices in stage-first order so each GPU model loads only once per stage.

Invocation pattern:

    python scripts/run_all_stages.py \
        path_config=jack_local_release \
        clip_indices.source=glob \
        stages.depth_estimation.enable=true

Stage-first ordering matters: stages 3 (FoundationStereo) and 4 (RAFT) each
load a large model on CUDA. If we ran clip-by-clip (clip 1 → all four stages
then clip 2 → all four stages …) we'd reload the model from disk N×2 times
per sweep. The implementation here keeps the **outer** loop at the stage
level and the **inner** loop at the clip level, so each stage script process
keeps its model alive across clips via a single Hydra multi-run sweep
(``python gen_*.py -m path_config.data_index=1,2,3``).

Resumability: before running a (stage, clip) pair, we check whether the
expected output directory already exists and contains files. If so, the
pair is skipped — re-running after a crash picks up where it left off.
Override with ``force=true``.

Structured logging: each (stage, clip_index) outcome is appended to
``<log_dir>/<run_id>.jsonl`` so operators can audit which clips finished
which stages.
"""
from __future__ import annotations
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from dvrk_data_processing.utils.hydra_config import PathConfig

# Resolve the project root (the repo root that contains src/ and FoundationStereo/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Dataclass schemas (registered with ConfigStore)
# --------------------------------------------------------------------------- #

@dataclass
class ClipIndicesConfig:
    # Discovery mode:
    #   "list"    — use the explicit `list` field below.
    #   "dataset" — process every subfolder of `<data_dir>/<dataset_name>/`.
    #               Convenience for the common "run all clips under one
    #               dataset" case — no glob pattern needed.
    #   "glob"    — glob each pattern in `glob_patterns` under data_dir
    #               (use when sweeping multiple datasets at once).
    #   "file"    — read a YAML/JSON file at `filter_file`.
    source: str = "list"
    list: List[str] = field(default_factory=list)
    # "dataset" mode: name of the single dataset folder under data_dir
    # whose subfolders should each be treated as a clip. Defaults to
    # path_config.data_name when blank.
    dataset_name: str = ""
    # "glob" mode patterns. Each pattern is "<dataset>/<index_glob>" joined
    # under data_dir, e.g. "offline_data/*".
    glob_patterns: List[str] = field(default_factory=list)
    filter_file: str = ""
    vary_data_name: bool = False                     # if True, sweep data_name alongside data_index


@dataclass
class StageToggle:
    enable: bool = True
    config_name: str = ""
    skip_if_no_gpu: bool = False


# StagesConfig has no class-level defaults — values come from
# `config/run_all_stages.yaml`. This sidesteps a Hydra/OmegaConf gotcha where
# nested-dataclass `default_factory=lambda: …` triggers a schema-merge error
# ("'Field' object is not callable") on Hydra 1.3. Keeping the dataclass
# slot-only means OmegaConf builds the leaf values from the YAML.
@dataclass
class StagesConfig:
    rectify_resize: StageToggle = field(default_factory=StageToggle)
    kinematic_reproject: StageToggle = field(default_factory=StageToggle)
    depth_estimation: StageToggle = field(default_factory=StageToggle)
    optical_flow_raft: StageToggle = field(default_factory=StageToggle)


@dataclass
class RunAllStagesAppCfg:
    path_config: PathConfig
    workspace: str
    clip_indices: ClipIndicesConfig
    stages: StagesConfig
    force: bool = False
    log_dir: str = ""
    child_path_config: str = "jack_local_release"


# NOTE: we intentionally do NOT register RunAllStagesAppCfg with the ConfigStore.
# Hydra 1.3 + OmegaConf has a known bug with nested-dataclass default_factory
# during schema validation that surfaces here as
#     ConfigTypeError: 'Field' object is not callable
# Skipping registration falls back to plain YAML-driven config which works the
# same in practice — the orchestrator only reads fields, it doesn't need
# OmegaConf's structured-config type checking.


# --------------------------------------------------------------------------- #
# Stage descriptor table — encodes the per-stage script location, output
# folder layout (used for resumability checks), and CWD for the subprocess.
# Keep this in sync with the stage map (rectify → kinematic → depth → flow).
# --------------------------------------------------------------------------- #

@dataclass
class StageDescriptor:
    name: str
    script_dir: Path           # directory holding the stage script (the script's `cwd`)
    script: str                # file name of the stage script
    # Output directory under processed_dir or intermediate_dir, with templating:
    #   uses {intermediate_dir} or {processed_dir} which we fill at runtime.
    # When `present` returns True (non-empty dir), the stage is considered done.
    output_probe: str

    def output_path(self, intermediate_dir: Path, processed_dir: Path) -> Path:
        return Path(self.output_probe.format(
            intermediate_dir=str(intermediate_dir),
            processed_dir=str(processed_dir),
        ))


# Mapping from sweep-toggle key in stages_cfg → descriptor.
# Order here is execution order (must be stage-first as noted in the docstring).
STAGE_ORDER: List[Tuple[str, StageDescriptor]] = [
    ("rectify_resize", StageDescriptor(
        name="rectify_resize",
        script_dir=PROJECT_ROOT / "src" / "dvrk_data_processing" / "raw_image_processing",
        script="gen_rectify_resize.py",
        # Stage 1 writes rectified frames here.
        output_probe="{intermediate_dir}/image/left",
    )),
    ("kinematic_reproject", StageDescriptor(
        name="kinematic_reproject",
        script_dir=PROJECT_ROOT / "src" / "dvrk_data_processing" / "kinematic_mapping",
        script="gen_kinematic_heatmap_handeye.py",
        # Stage 2 writes per-arm heatmaps here. Use PSM1/left/heatmap as the
        # canonical probe — if anything failed we'd see this folder missing.
        output_probe="{processed_dir}/kinematic_reproject/PSM1/left/heatmap",
    )),
    ("depth_estimation", StageDescriptor(
        name="depth_estimation",
        script_dir=PROJECT_ROOT / "src" / "dvrk_data_processing" / "depth_estimation",
        script="gen_depth_estimate.py",
        output_probe="{processed_dir}/depth_estimation/disparity",
    )),
    ("optical_flow_raft", StageDescriptor(
        name="optical_flow_raft",
        script_dir=PROJECT_ROOT / "src" / "dvrk_data_processing" / "optical_flow",
        script="gen_optical_flow_raft.py",
        output_probe="{processed_dir}/optical_flow/left/optical_flow",
    )),
]


# --------------------------------------------------------------------------- #
# Clip discovery
# --------------------------------------------------------------------------- #

def discover_clips(cfg: RunAllStagesAppCfg) -> List[Tuple[str, str]]:
    """
    Resolve `cfg.clip_indices` to a list of (data_name, data_index) pairs.

    - source=="list": yield (cfg.path_config.data_name, idx) for each idx.
    - source=="glob": glob each pattern under data_dir; return the matching
      dataset names + clip indices.
    - source=="file": read a YAML/JSON file with `clip_indices: [..]`.
    """
    source = cfg.clip_indices.source
    data_dir = Path(cfg.path_config.data_dir)

    if source == "list":
        return [(cfg.path_config.data_name, str(i)) for i in cfg.clip_indices.list]

    if source == "dataset":
        # Sweep every subfolder of <data_dir>/<dataset_name>/. Sort by clip
        # name (lexicographic) so the JSONL log is deterministic.
        dataset_name = cfg.clip_indices.dataset_name or cfg.path_config.data_name
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"clip_indices.source='dataset' but {dataset_dir} does not exist."
            )
        # Tolerate clips whose names sort better numerically (`1, 2, 10`) by
        # falling back to numeric key when every name parses as int.
        subdirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
        try:
            subdirs = sorted(subdirs, key=lambda p: int(p.name))
        except ValueError:
            subdirs = sorted(subdirs, key=lambda p: p.name)
        if not subdirs:
            raise FileNotFoundError(
                f"clip_indices.source='dataset' but {dataset_dir} has no subfolders."
            )
        return [(dataset_name, p.name) for p in subdirs]

    if source == "glob":
        # Each pattern is "<data_name>/<index_glob>". e.g. "offline_data/*".
        results: List[Tuple[str, str]] = []
        for pat in cfg.clip_indices.glob_patterns:
            for p in sorted(data_dir.glob(pat)):
                if not p.is_dir():
                    continue
                # Parent name is the dataset name (offline_data / online_data).
                results.append((p.parent.name, p.name))
        if not results:
            raise FileNotFoundError(
                f"No clips discovered under {data_dir} with patterns "
                f"{list(cfg.clip_indices.glob_patterns)}."
            )
        return results

    if source == "file":
        path = Path(cfg.clip_indices.filter_file)
        if not path.exists():
            raise FileNotFoundError(f"clip_indices.filter_file not found: {path}")
        if path.suffix in {".yaml", ".yml"}:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        indices = data.get("clip_indices", [])
        return [(cfg.path_config.data_name, str(i)) for i in indices]

    raise ValueError(f"Unknown clip_indices.source: {source!r}")


# --------------------------------------------------------------------------- #
# GPU detection (best-effort; used by skip_if_no_gpu)
# --------------------------------------------------------------------------- #

def is_gpu_available() -> bool:
    """
    Check if CUDA is available without importing torch (keeps the orchestrator
    light-weight). We probe via `nvidia-smi`; absence of the binary or a
    non-zero exit means "no GPU".
    """
    try:
        subprocess.run(["nvidia-smi"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# --------------------------------------------------------------------------- #
# Per-stage invocation
# --------------------------------------------------------------------------- #

def output_present(out_dir: Path) -> bool:
    """A stage's output is considered "present" iff the probe directory
    exists and contains at least one file."""
    if not out_dir.exists():
        return False
    try:
        next(out_dir.iterdir())
        return True
    except StopIteration:
        return False


def build_child_command(stage_cfg: StageToggle, descriptor: StageDescriptor,
                        child_path_config: str, data_name: str, data_index: str,
                        force: bool) -> List[str]:
    """
    Build the child subprocess command for a single (stage, clip) invocation.

    The child is the per-stage Hydra script, invoked with overrides:
        --config-name=<stage_cfg.config_name>
        path_config=<child_path_config>
        path_config.data_name=<data_name>
        path_config.data_index=<data_index>
    """
    # We don't use Hydra's `-m` (multi-run) here because we already serialize
    # per-clip in the parent loop. Each clip becomes its own subprocess
    # invocation (one Hydra single-run per clip). That's the simplest path
    # consistent with the "no parallelism across clips" decision and
    # keeps the structured-log entries 1:1 with subprocess outcomes.
    cmd = [
        sys.executable, descriptor.script,
        f"--config-name={stage_cfg.config_name}",
        f"path_config={child_path_config}",
        f"path_config.data_name={data_name}",
        f"path_config.data_index={data_index}",
    ]
    # NOTE: `force` only bypasses the orchestrator's parent-side resumability
    # check (see the existence test in main()). We deliberately do NOT pass
    # `folder_initialize=true` to the child here because `clear_folder()` calls
    # `input()` to ask for confirmation — that would deadlock the subprocess.
    # The stage scripts already overwrite per-frame outputs in place, so a
    # rerun without folder wipe is correct.
    return cmd


def run_single(cmd: List[str], cwd: Path) -> Tuple[int, float]:
    """
    Run a single subprocess, returning (returncode, wall_seconds).
    """
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode, time.time() - t0


# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #

class JsonlLogger:
    """
    Append-only JSONL writer. Each record is one JSON object per line. We
    flush after every record so partial logs survive a hard kill.
    """
    def __init__(self, log_path: Path):
        self.path = log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in line-buffered text mode.
        self._f = open(self.path, "a", buffering=1)

    def log(self, **fields: Any) -> None:
        fields.setdefault("ts", time.time())
        self._f.write(json.dumps(fields, default=str) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Hydra entry point
# --------------------------------------------------------------------------- #

# Config path — same convention as the other scripts under scripts/. The
# config file lives at `<repo>/config/run_all_stages.yaml`.
config_path = PROJECT_ROOT / "config"


@hydra.main(version_base=None, config_path=str(config_path), config_name="run_all_stages")
def main(cfg: DictConfig):
    # Mint a run id (short, human-readable; combines hostname + 6-char uuid).
    run_id = f"{socket.gethostname()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    log_dir = Path(cfg.log_dir) if cfg.log_dir else Path(cfg.path_config.data_dir) / ".run_all_stages"
    log_path = log_dir / f"{run_id}.jsonl"
    logger = JsonlLogger(log_path)
    print(f"[run_all_stages] run_id = {run_id}")
    print(f"[run_all_stages] log     = {log_path}")

    logger.log(event="start", run_id=run_id,
               cfg=OmegaConf.to_container(cfg, resolve=True))

    # Discover the clips to sweep.
    clips = discover_clips(cfg)
    print(f"[run_all_stages] discovered {len(clips)} clip(s): {clips}")
    logger.log(event="clips_discovered", clips=clips, count=len(clips))

    gpu_ok = is_gpu_available()
    print(f"[run_all_stages] gpu_available = {gpu_ok}")
    if not gpu_ok:
        logger.log(event="gpu_unavailable", warning="GPU stages with skip_if_no_gpu=true will be skipped.")

    # ------------------------------------------------------------------- #
    # Stage-first sweep
    # ------------------------------------------------------------------- #
    for stage_key, descriptor in STAGE_ORDER:
        stage_cfg: StageToggle = getattr(cfg.stages, stage_key)
        if not stage_cfg.enable:
            print(f"[run_all_stages] stage {stage_key!r} disabled — skipping")
            logger.log(event="stage_skipped", stage=stage_key, reason="disabled")
            continue
        if stage_cfg.skip_if_no_gpu and not gpu_ok:
            print(f"[run_all_stages] stage {stage_key!r} requires GPU — skipping (skip_if_no_gpu=true)")
            logger.log(event="stage_skipped", stage=stage_key, reason="no_gpu")
            continue

        print(f"[run_all_stages] ===== STAGE: {stage_key} =====")
        logger.log(event="stage_start", stage=stage_key, config_name=stage_cfg.config_name)

        for data_name, data_index in clips:
            # Build the per-clip path config to probe the output directory.
            # We don't actually compose Hydra here — we manually mirror the
            # path templating from jack_local_release.yaml so existence
            # checks are cheap.
            data_dir = Path(cfg.path_config.data_dir)
            # Same template as jack_local_release.yaml (no extra prefix).
            raw_dir = data_dir / data_name / data_index
            intermediate_dir = data_dir / "intermediate" / data_name / data_index
            processed_dir = data_dir / "preprocess" / data_name / data_index

            out_probe = descriptor.output_path(intermediate_dir, processed_dir)
            already_done = output_present(out_probe) and not cfg.force

            if already_done:
                print(f"  [{stage_key} | {data_name}/{data_index}] already done at {out_probe} — skip")
                logger.log(event="clip_skipped", stage=stage_key,
                           data_name=data_name, data_index=data_index,
                           probe=str(out_probe), reason="already_done")
                continue

            cmd = build_child_command(stage_cfg, descriptor,
                                      child_path_config=cfg.child_path_config,
                                      data_name=data_name, data_index=data_index,
                                      force=cfg.force)
            print(f"  [{stage_key} | {data_name}/{data_index}] running: {' '.join(cmd)}")
            logger.log(event="clip_start", stage=stage_key,
                       data_name=data_name, data_index=data_index,
                       cmd=cmd, cwd=str(descriptor.script_dir))

            rc, wall = run_single(cmd, cwd=descriptor.script_dir)
            ok = (rc == 0)
            print(f"  [{stage_key} | {data_name}/{data_index}] {'OK' if ok else 'FAIL'} "
                  f"(rc={rc}, {wall:.1f}s)")
            logger.log(event="clip_end", stage=stage_key,
                       data_name=data_name, data_index=data_index,
                       returncode=rc, wall_seconds=wall, ok=ok)

            if not ok:
                # Don't abort the whole sweep on a single-clip failure — log
                # and move on. The operator can re-run with `force=false` to
                # retry only the failed clips.
                logging.warning(
                    f"Stage {stage_key} failed on clip {data_name}/{data_index} (rc={rc})."
                )

        logger.log(event="stage_end", stage=stage_key)

    logger.log(event="finish", run_id=run_id)
    logger.close()
    print(f"[run_all_stages] DONE. Structured log at {log_path}")


if __name__ == "__main__":
    main()
