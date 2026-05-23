"""Preprocess encoder tests — PNG → FFV1 bgr24/gray8 bit-exact round-trip."""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import pytest

from dvrk_data_processing.surgsync.encode.codec import (
    decode_video_frames, probe_frame_count,
)
from dvrk_data_processing.surgsync.encode.preprocess import (
    write_depth, write_flow, write_heatmap, write_preprocess,
)


REPO = Path(__file__).resolve().parents[3]
ONLINE_PROCESSED = REPO / "data" / "online_data" / "2" / "preprocess"


def _synth_color_png(path: Path, h: int = 32, w: int = 48, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _synth_gray_png(path: Path, h: int = 32, w: int = 48, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_write_depth_bit_exact_synthetic(tmp_path: Path):
    """write_depth uses FFV1 bgr24 → decode must round-trip the source
    PNGs bit-exactly."""
    n = 5
    src_dir = tmp_path / "src" / "depth_estimation" / "depth_image"
    for i in range(n):
        _synth_color_png(src_dir / f"{i}.png", seed=i)

    dst = tmp_path / "depth.mkv"
    n_real = write_depth(tmp_path / "src", dst,
                         fps=10.0, n_frames_expected=n)
    assert n_real == n
    assert probe_frame_count(dst) == n

    decoded = list(decode_video_frames(dst, pix_fmt="bgr24", width=48, height=32))
    for i, frame in enumerate(decoded):
        original = cv2.imread(str(src_dir / f"{i}.png"), cv2.IMREAD_COLOR)
        assert np.array_equal(frame, original), f"frame {i} not bit-exact"


def test_write_flow_bit_exact_synthetic(tmp_path: Path):
    n = 4
    src_dir = tmp_path / "src" / "optical_flow" / "left" / "image"
    for i in range(n):
        _synth_color_png(src_dir / f"{i}.png", seed=i + 10)
    dst = tmp_path / "flow_left.mkv"
    n_real = write_flow(tmp_path / "src", "left", dst,
                        fps=10.0, n_frames_expected=n)
    assert n_real == n
    decoded = list(decode_video_frames(dst, pix_fmt="bgr24", width=48, height=32))
    for i, frame in enumerate(decoded):
        original = cv2.imread(str(src_dir / f"{i}.png"), cv2.IMREAD_COLOR)
        assert np.array_equal(frame, original)


def test_write_heatmap_bit_exact_synthetic(tmp_path: Path):
    n = 3
    src_dir = tmp_path / "src" / "kinematic_reproject" / "PSM1" / "left" / "image"
    for i in range(n):
        _synth_gray_png(src_dir / f"{i}.png", seed=i + 20)
    dst = tmp_path / "heatmap.mkv"
    n_real = write_heatmap(tmp_path / "src", "PSM1", "left", dst,
                           fps=10.0, n_frames_expected=n)
    assert n_real == n
    decoded = list(decode_video_frames(dst, pix_fmt="gray8", width=48, height=32))
    for i, frame in enumerate(decoded):
        original = cv2.imread(str(src_dir / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(frame, original)


def test_missing_source_skips_silently(tmp_path: Path):
    """Optional-data contract: an absent source dir returns 0 and writes
    no MKV. The caller decides whether to surface that as has_preprocess=false."""
    n = write_depth(tmp_path, tmp_path / "depth.mkv", fps=10.0, n_frames_expected=5)
    assert n == 0
    assert not (tmp_path / "depth.mkv").exists()


def test_write_preprocess_handles_partial_streams(tmp_path: Path):
    """Only one of three streams present — the dispatcher writes that one
    and skips the others without raising."""
    n = 3
    src = tmp_path / "src"
    # Only flow_left present.
    for i in range(n):
        _synth_color_png(src / "optical_flow" / "left" / "image" / f"{i}.png", seed=i)

    written = write_preprocess(src, tmp_path / "dst", fps=10.0, n_frames_expected=n)
    assert "flow_left" in written
    assert "depth" not in written
    assert "flow_right" not in written
    assert (tmp_path / "dst" / "flow_left.mkv").exists()
    assert not (tmp_path / "dst" / "depth.mkv").exists()


def test_missing_frame_padded_with_zeros(tmp_path: Path):
    """If a frame is missing from the source dir, the MKV gets a zero
    placeholder so its frame count matches the master timeline."""
    n = 5
    src_dir = tmp_path / "src" / "depth_estimation" / "depth_image"
    # Only frames 0, 2, 4 present — 1 and 3 missing.
    for i in (0, 2, 4):
        _synth_color_png(src_dir / f"{i}.png", seed=i)

    dst = tmp_path / "depth.mkv"
    n_real = write_depth(tmp_path / "src", dst, fps=10.0, n_frames_expected=n)
    assert n_real == 3
    assert probe_frame_count(dst) == n


# ---------------------------------------------------------------------------
# Smoke against real sample data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ONLINE_PROCESSED / "depth_estimation" / "depth_image").exists(),
    reason="preprocessing depth_image output not present",
)
def test_depth_against_sample_data(tmp_path: Path):
    n = 10
    dst = tmp_path / "depth.mkv"
    n_real = write_depth(ONLINE_PROCESSED, dst, fps=10.0, n_frames_expected=n)
    assert n_real == n
    assert probe_frame_count(dst) == n


@pytest.mark.skipif(
    not (ONLINE_PROCESSED / "optical_flow" / "left" / "image").exists(),
    reason="preprocessing flow image output not present",
)
def test_flow_against_sample_data(tmp_path: Path):
    n = 10
    dst = tmp_path / "flow_left.mkv"
    n_real = write_flow(ONLINE_PROCESSED, "left", dst, fps=10.0, n_frames_expected=n)
    # Preprocessing flow has N-1 PNGs for N frames; the last is padded.
    assert n_real >= n - 1
    assert probe_frame_count(dst) == n
