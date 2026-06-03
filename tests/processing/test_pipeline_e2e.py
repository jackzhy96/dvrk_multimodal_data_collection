"""
End-to-end smoke test of all four preprocessing stages.

Runs stage 1 → 2 → 3 → 4 on the sample clips ``offline_data/3/`` and
``online_data/2/`` via the existing Hydra CLI entry points. Catches
regressions from the rename / calibrated_kinematic / depth / drawframe
changes.

Run with::

    cd <repository root>
    python tests/processing/test_pipeline_e2e.py

The script is intentionally **not** a pytest fixture — the project's tests/
tree is a collection of standalone validators. The
script returns exit code 0 on success and non-zero on any check failure.

Stage 3 (depth) is skipped with a clear message if FoundationStereo weights
aren't downloaded under ``FoundationStereo/pretrained_models/``.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "dvrk_data_processing"
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
FS_WEIGHTS = REPO_ROOT / "FoundationStereo" / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #

def run_subprocess(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run a subprocess and capture output. Returns (rc, stdout, stderr)."""
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, env=env or os.environ.copy(),
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_rectify_resize(data_name: str, data_index: str, overrides: List[str]) -> None:
    cmd = [
        sys.executable, "gen_rectify_resize.py",
        "--config-name=config_rr_jack",
        f"path_config=jack_local_release",
        f"path_config.data_name={data_name}",
        f"path_config.data_index={data_index}",
    ] + overrides
    rc, out, err = run_subprocess(cmd, cwd=SRC_DIR / "raw_image_processing")
    if rc != 0:
        raise RuntimeError(f"rectify_resize failed (rc={rc}):\nSTDOUT:\n{out}\nSTDERR:\n{err}")


def run_kinematic(data_name: str, data_index: str, overrides: List[str]) -> None:
    cmd = [
        sys.executable, "gen_kinematic_heatmap_handeye.py",
        "--config-name=config_kp_jack",
        f"path_config=jack_local_release",
        f"path_config.data_name={data_name}",
        f"path_config.data_index={data_index}",
    ] + overrides
    rc, out, err = run_subprocess(cmd, cwd=SRC_DIR / "kinematic_mapping")
    if rc != 0:
        raise RuntimeError(f"kinematic_reproject failed (rc={rc}):\nSTDOUT:\n{out}\nSTDERR:\n{err}")


def run_depth(data_name: str, data_index: str, overrides: List[str]) -> None:
    cmd = [
        sys.executable, "gen_depth_estimate.py",
        "--config-name=config_de_jack",
        f"path_config=jack_local_release",
        f"path_config.data_name={data_name}",
        f"path_config.data_index={data_index}",
    ] + overrides
    rc, out, err = run_subprocess(cmd, cwd=SRC_DIR / "depth_estimation")
    if rc != 0:
        raise RuntimeError(f"depth_estimation failed (rc={rc}):\nSTDOUT:\n{out}\nSTDERR:\n{err}")


def run_optical_flow(data_name: str, data_index: str, overrides: List[str]) -> None:
    cmd = [
        sys.executable, "gen_optical_flow_raft.py",
        "--config-name=config_of_raft_jack",
        f"path_config=jack_local_release",
        f"path_config.data_name={data_name}",
        f"path_config.data_index={data_index}",
    ] + overrides
    rc, out, err = run_subprocess(cmd, cwd=SRC_DIR / "optical_flow")
    if rc != 0:
        raise RuntimeError(f"optical_flow failed (rc={rc}):\nSTDOUT:\n{out}\nSTDERR:\n{err}")


# --------------------------------------------------------------------------- #
# Assertions on the expected layout
# --------------------------------------------------------------------------- #

def assert_nonempty_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise AssertionError(f"{label}: directory missing: {path}")
    if not any(path.iterdir()):
        raise AssertionError(f"{label}: directory empty: {path}")
    print(f"  OK  {label}: {path}")


def check_stage1_layout(intermediate_dir: Path) -> None:
    assert_nonempty_dir(intermediate_dir / "image" / "left", "stage1: image/left")
    assert_nonempty_dir(intermediate_dir / "image" / "right", "stage1: image/right")
    assert_nonempty_dir(intermediate_dir / "camera_calibration", "stage1: camera_calibration")
    rectify_json = intermediate_dir / "camera_calibration" / "rectify_params.json"
    if not rectify_json.exists():
        raise AssertionError(f"stage1: missing rectify_params.json: {rectify_json}")
    print(f"  OK  stage1: rectify_params.json present")


def check_stage2_layout(processed_dir: Path, expect_setpoint: bool) -> None:
    kr = processed_dir / "kinematic_reproject"
    for psm in ("PSM1", "PSM2"):
        assert_nonempty_dir(kr / psm / "left" / "heatmap", f"stage2: {psm}/left/heatmap")
        assert_nonempty_dir(kr / psm / "right" / "heatmap", f"stage2: {psm}/right/heatmap")
        # calibrated_kinematic: JSONs only for PSMs (never ECM).
        ck_dir = kr / psm / "calibrated_kinematic"
        assert_nonempty_dir(ck_dir, f"stage2: {psm}/calibrated_kinematic")

        # Sanity-check the first JSON: schema + setpoint presence/absence.
        sample = next(iter(sorted(ck_dir.glob("*.json"))))
        with open(sample) as f:
            payload = json.load(f)
        assert "measured_cp_calibrated" in payload, f"{sample}: missing measured_cp_calibrated"
        mc = payload["measured_cp_calibrated"]
        assert len(mc["position"]) == 3, f"{sample}: position must be length-3"
        assert len(mc["orientation"]) == 4, f"{sample}: orientation must be length-4 (xyzw)"
        if expect_setpoint:
            assert "setpoint_cp_calibrated" in payload, (
                f"{sample}: expected setpoint_cp_calibrated for online clip"
            )
        else:
            assert "setpoint_cp_calibrated" not in payload, (
                f"{sample}: offline clip should omit setpoint_cp_calibrated, got it"
            )
        print(f"  OK  stage2: {psm}/calibrated_kinematic schema (sample: {sample.name})")

    # No ECM calibrated_kinematic — explicit absence is a strict requirement.
    if (kr / "ECM" / "calibrated_kinematic").exists():
        raise AssertionError("stage2: ECM/calibrated_kinematic exists but must not be emitted.")

    # drawframe sibling tree.
    drf = processed_dir / "kinematic_reproject_drawframe"
    for psm in ("PSM1", "PSM2"):
        for cam in ("left", "right"):
            assert_nonempty_dir(drf / psm / cam, f"stage2: drawframe/{psm}/{cam}")


def check_stage3_layout(processed_dir: Path) -> None:
    base = processed_dir / "depth_estimation"
    assert_nonempty_dir(base / "disparity", "stage3: disparity")
    assert_nonempty_dir(base / "depth", "stage3: depth")
    # depth_image is only created when save_visualization=true (default).
    # Tolerate its absence to keep this test resilient to config overrides.
    if (base / "depth_image").exists():
        assert_nonempty_dir(base / "depth_image", "stage3: depth_image")

    # depth/<i>.npy values: at least one non-NaN, and physically plausible.
    import numpy as np
    sample_npy = sorted((base / "depth").glob("*.npy"))[0]
    depth = np.load(sample_npy)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise AssertionError(f"stage3: {sample_npy} is all-NaN")
    p_lo, p_hi = float(finite.min()), float(finite.max())
    if not (0.0 < p_lo < 5.0 and 0.0 < p_hi < 5.0):
        raise AssertionError(
            f"stage3: depth range looks unphysical: [{p_lo:.4f}, {p_hi:.4f}] m"
        )
    print(f"  OK  stage3: depth range [{p_lo:.4f}, {p_hi:.4f}] m")


def check_stage4_layout(processed_dir: Path) -> None:
    base = processed_dir / "optical_flow"
    for cam in ("left", "right"):
        assert_nonempty_dir(base / cam / "optical_flow", f"stage4: {cam}/optical_flow")


# --------------------------------------------------------------------------- #
# Per-clip orchestration
# --------------------------------------------------------------------------- #

def run_clip(data_name: str, data_index: str, expect_setpoint: bool,
             skip_depth: bool, skip_flow: bool) -> None:
    """Run the four stages back-to-back on a single sample clip."""
    t_clip = time.time()
    print(f"\n=== clip {data_name}/{data_index} ===")

    # Path layout matches `config/path_config/jack_local_release.yaml`.
    intermediate_dir = DATA_DIR / "intermediate" / data_name / data_index
    processed_dir = DATA_DIR / "preprocess" / data_name / data_index

    # Stage 1 — rectify+resize. Force re-write so we don't pick up stale state.
    print("-- stage 1: rectify_resize")
    run_rectify_resize(data_name, data_index, ["+preprocess.folder_initialize=false"])
    check_stage1_layout(intermediate_dir)

    # Stage 2 — kinematic + calibrated_kinematic + drawframe.
    print("-- stage 2: kinematic_reproject")
    run_kinematic(data_name, data_index, [
        "+preprocess.calibrated_kinematic.enable=true",
        "+preprocess.drawframe.enable=true",
    ])
    check_stage2_layout(processed_dir, expect_setpoint=expect_setpoint)

    # Stage 3 — depth.
    if skip_depth:
        print("-- stage 3: SKIPPED (FoundationStereo weights missing)")
    else:
        print("-- stage 3: depth_estimation")
        run_depth(data_name, data_index, [])
        check_stage3_layout(processed_dir)

    # Stage 4 — RAFT optical flow.
    if skip_flow:
        print("-- stage 4: SKIPPED (no GPU)")
    else:
        print("-- stage 4: optical_flow_raft")
        run_optical_flow(data_name, data_index, [])
        check_stage4_layout(processed_dir)

    print(f"=== clip {data_name}/{data_index} OK ({time.time() - t_clip:.1f}s)\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocessing end-to-end smoke test.")
    parser.add_argument("--skip-depth", action="store_true",
                        help="Force-skip the depth stage (otherwise auto-detected from FS weights).")
    parser.add_argument("--skip-flow", action="store_true",
                        help="Force-skip the optical flow stage (e.g. CPU-only machine).")
    args = parser.parse_args()

    if not (DATA_DIR / "offline_data" / "3").exists():
        print(f"ERROR: sample clip offline_data/3 missing under {DATA_DIR}")
        return 1
    if not (DATA_DIR / "online_data" / "2").exists():
        print(f"ERROR: sample clip online_data/2 missing under {DATA_DIR}")
        return 1

    skip_depth = args.skip_depth or not FS_WEIGHTS.exists()
    if skip_depth and not args.skip_depth:
        print(f"WARN: FoundationStereo weights not found at {FS_WEIGHTS} — skipping stage 3.")

    # Best-effort GPU check (the depth and flow stages need CUDA).
    skip_flow = args.skip_flow
    if not skip_flow:
        # Allow the stage script to raise its own error if no GPU; we don't
        # pre-empt it here so that GPU-equipped boxes still exercise the path.
        pass

    t0 = time.time()
    try:
        # online_data/2 has setpoint_cp (online recorder).
        run_clip("online_data", "2", expect_setpoint=True,
                 skip_depth=skip_depth, skip_flow=skip_flow)
        # offline_data/3 has setpoint_data.setpoint_js but NO setpoint_cp.
        run_clip("offline_data", "3", expect_setpoint=False,
                 skip_depth=skip_depth, skip_flow=skip_flow)
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 2
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 3

    print(f"E2E smoke test PASSED in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
