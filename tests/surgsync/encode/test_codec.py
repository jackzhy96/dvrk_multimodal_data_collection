"""Codec round-trip tests.

The full `roundtrip_selftest` runs all three checks; the individual
helpers below let pytest attribute a specific failure to the right
codec/pixel-format combination instead of one mega-test.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from dvrk_data_processing.surgsync.encode.codec import (
    encode_mkv_ffv1,
    encode_h264_crf,
    decode_video_frames,
    roundtrip_selftest,
    probe_frame_count,
)


def test_ffv1_gray16_bit_exact(tmp_path: Path):
    h, w, n = 32, 48, 5
    rng = np.random.default_rng(seed=42)
    frames = [rng.integers(0, 65536, size=(h, w), dtype=np.uint16) for _ in range(n)]
    dst = tmp_path / "gray16.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="gray16le")

    decoded = list(decode_video_frames(dst, pix_fmt="gray16le", width=w, height=h))
    assert len(decoded) == n
    for a, b in zip(frames, decoded):
        assert np.array_equal(a, b)


def test_ffv1_bgr24_bit_exact(tmp_path: Path):
    """Raw-image FFV1 round-trip — bgr24 pix_fmt, the path used for
    `video_raw/`."""
    h, w, n = 16, 24, 4
    rng = np.random.default_rng(seed=7)
    frames = [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]
    dst = tmp_path / "raw.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="bgr24")

    decoded = list(decode_video_frames(dst, pix_fmt="bgr24", width=w, height=h))
    assert len(decoded) == n
    for a, b in zip(frames, decoded):
        assert np.array_equal(a, b)


def test_ffv1_gray8_bit_exact(tmp_path: Path):
    """8-bit grayscale — heatmap path."""
    h, w, n = 16, 24, 4
    rng = np.random.default_rng(seed=9)
    frames = [rng.integers(0, 256, size=(h, w), dtype=np.uint8) for _ in range(n)]
    dst = tmp_path / "gray8.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="gray8")

    decoded = list(decode_video_frames(dst, pix_fmt="gray8", width=w, height=h))
    assert len(decoded) == n
    for a, b in zip(frames, decoded):
        assert np.array_equal(a, b)


def test_ffv1_flow_gbrp16_bit_exact(tmp_path: Path):
    """Optical flow packed into 16-bit planar GBR — bit-exact."""
    h, w, n = 32, 40, 3
    rng = np.random.default_rng(seed=99)
    frames = [rng.integers(0, 65536, size=(3, h, w), dtype=np.uint16) for _ in range(n)]
    dst = tmp_path / "flow.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="gbrp16le")

    decoded = list(decode_video_frames(dst, pix_fmt="gbrp16le", width=w, height=h))
    assert len(decoded) == n
    for a, b in zip(frames, decoded):
        assert np.array_equal(a, b)


def test_h264_crf18_encoder_pipeline_smoke(tmp_path: Path):
    """End-to-end H.264 encode/decode produces an array of the right
    shape and a reasonable PSNR on synthetic content.

    Note: synthetic monochrome content (R=G=B) on yuv420p with BT.709
    limited-range conversion has an inherent ~38 dB floor unrelated to
    the encoder's quality setting. The contract "visually lossless on
    real surgical video" (PSNR ≥ 40 dB) is checked by the integration
    smoke test against actual rectified frames; this unit test asserts
    only that the encoder pipeline runs cleanly and produces the
    declared content shape.
    """
    h, w, n = 128, 128, 4
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        127.5
        + 80.0 * np.sin(2 * np.pi * xx / 64.0)
        + 40.0 * np.cos(2 * np.pi * yy / 80.0)
    ).clip(0, 255).astype(np.uint8)
    frames = [np.stack([base, base, base], axis=-1) for _ in range(n)]
    dst = tmp_path / "h264.mkv"
    encode_h264_crf(frames, dst, fps=30.0, crf=18)

    decoded = list(decode_video_frames(dst, pix_fmt="bgr24", width=w, height=h))
    assert len(decoded) == n
    psnrs = []
    for a, b in zip(frames, decoded):
        mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
        psnrs.append(float("inf") if mse == 0 else 10 * np.log10(65025.0 / mse))
    # 30 dB floor — comfortably above the colorspace-conversion noise but
    # would catch a real encoder regression (e.g. CRF 51 or wrong codec).
    assert min(psnrs) >= 30.0, f"PSNR floor unreasonably low: {min(psnrs):.2f} dB"


def test_probe_frame_count(tmp_path: Path):
    """ffprobe nb_read_frames returns the right number."""
    h, w, n = 16, 16, 7
    rng = np.random.default_rng(seed=33)
    frames = [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]
    dst = tmp_path / "probe.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="bgr24")
    assert probe_frame_count(dst) == n


def test_h264_rejects_odd_dimensions(tmp_path: Path):
    """H.264 yuv420p needs even W/H. Encoder must reject early instead of
    silently rescaling."""
    frames = [np.zeros((17, 32, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="even dimensions"):
        encode_h264_crf(frames, tmp_path / "bad.mkv", fps=30.0)


def test_full_selftest_passes(tmp_path: Path):
    """The end-to-end selftest exposed by `surgsync.encode.codec selftest`."""
    roundtrip_selftest(tmp_path)
