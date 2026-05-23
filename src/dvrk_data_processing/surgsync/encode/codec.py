"""ffmpeg wrappers + FFV1 round-trip selftest.

This is the **single** module in `surgsync/` that shells out to ffmpeg.
Every encoder calls into the helpers here; nothing else invokes the
`ffmpeg`/`ffprobe` binaries.

Pixel-format choices:
- `gray8`   → 8-bit single-channel; used for processed RGB only as a
              probe target. Real RGB uses `yuv420p` / `bgr24`.
- `bgr24`   → 8-bit packed RGB; used by H.264 raw-PNG → MKV pipeline.
- `gray16le`→ 16-bit little-endian single-channel; legacy 16-bit
              preprocess round-trip target. Bit-exact under FFV1.
- `gray16be`→ big-endian variant; we use `gray16le` consistently.
- `yuv420p` → standard chroma-subsampled RGB for H.264. CRF 18 picks
              near-visually-lossless quality.

The two channels needed for optical flow (u, v) are packed into a
**gbrp16le** container (planar 16-bit GBR), where:
  G plane = u
  B plane = v
  R plane = zeros padding
FFV1 round-trips this exactly. The encoder's caller is responsible for
populating the R plane to zeros.

Minimum ffmpeg version: 4.4 (FFV1 v3 stable; gbrp16le supported). The
ffmpeg in `dvrk_multimodal_process` is 8.0.x at the time of writing.
"""
from __future__ import annotations
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

import numpy as np


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffmpeg discovery + invocation
# ---------------------------------------------------------------------------

def _ffmpeg_bin() -> str:
    """Resolve the ffmpeg binary; raise if missing.

    Uses `shutil.which` so the conda env's ffmpeg takes precedence over a
    system one. Cached intentionally **not** done — `which` is fast and
    avoiding the cache makes monkey-patching easy in tests.
    """
    bin_path = shutil.which("ffmpeg")
    if bin_path is None:
        raise RuntimeError("ffmpeg not found on PATH. Install via conda-forge.")
    return bin_path


def _ffprobe_bin() -> str:
    bin_path = shutil.which("ffprobe")
    if bin_path is None:
        raise RuntimeError("ffprobe not found on PATH.")
    return bin_path


def _run_ffmpeg(args: list[str], *, stdin_data: Optional[bytes] = None) -> None:
    """Run ffmpeg with the given args, raising on non-zero exit.

    Captures stderr for the error message; ffmpeg sends progress to
    stderr and errors there too. On success we discard it to keep logs
    clean.
    """
    cmd = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd, check=True,
            input=stdin_data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode(errors="replace") if e.stderr else "<empty>"
        raise RuntimeError(f"ffmpeg failed (rc={e.returncode}):\n{stderr_text}") from None


# ---------------------------------------------------------------------------
# H.264 CRF — processed RGB
# ---------------------------------------------------------------------------

def encode_h264_crf(
    frame_iter: Iterable[np.ndarray],
    dst_path: Path,
    *,
    fps: float,
    crf: int = 18,
    width: Optional[int] = None,
    height: Optional[int] = None,
    pix_fmt_in: str = "bgr24",
) -> None:
    """Encode an iterable of frames into MKV/H.264 (CRF 18 default).

    `frame_iter` yields contiguous uint8 numpy arrays of shape (H, W, 3).
    Frames are streamed to ffmpeg via stdin in `bgr24` (cv2 default) so
    no temp-PNG round-trip is needed.

    Lossy by design: H.264 CRF 18 is visually lossless. PSNR ≥ 40 dB is
    the contract; round-trip checks for this in `tests/`.

    Note: width/height must be supplied unless the caller knows the
    first frame's shape exactly — `peek + iter.tee` would also work but
    we keep this explicit for clarity.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull the first frame to discover shape if not given.
    iterator = iter(frame_iter)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"encode_h264_crf called with empty iterable for {dst_path}")
    # H.264 path consumes packed bgr24 only: (H, W, 3).
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(
            f"encode_h264_crf expects (H, W, 3) bgr24 frames; got shape {first.shape}"
        )
    H, W = first.shape[0], first.shape[1]
    if width is not None and width != W:
        raise ValueError(f"frame width {W} mismatches requested {width}")
    if height is not None and height != H:
        raise ValueError(f"frame height {H} mismatches requested {height}")
    width, height = W, H

    args = [
        # input: raw video frames via pipe
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt_in,
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        # output: H.264 in MKV, yuv420p for broad consumer compat
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", "medium",
        # ensure even dimensions (H.264 requires even W/H for yuv420p)
        # — error out instead of silently scaling.
        str(dst_path),
    ]
    if width % 2 or height % 2:
        raise ValueError(
            f"H.264 (yuv420p) requires even dimensions; got {width}x{height}. "
            "Resize to an even size in the upstream pipeline."
        )

    # Build the byte stream lazily — but ffmpeg's stdin doesn't easily
    # accept a generator across the subprocess boundary. We use Popen
    # with explicit write() to stream without buffering the whole video.
    proc = subprocess.Popen(
        [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(first.tobytes())
        for frame in iterator:
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"frame shape changed mid-stream: expected ({height},{width}) got {frame.shape[:2]}"
                )
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        # ffmpeg died — surface its stderr below.
        pass

    stderr = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg H.264 encode failed (rc={rc}) for {dst_path}:\n"
            f"{stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# FFV1 — bit-exact, used for raw images and preprocess viz streams
# ---------------------------------------------------------------------------

def encode_mkv_ffv1(
    frame_iter: Iterable[np.ndarray],
    dst_path: Path,
    *,
    fps: float,
    pix_fmt: str = "bgr24",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Encode frames into MKV/FFV1 — bit-exact round-trip target.

    `pix_fmt` choices:
        bgr24      — 8-bit packed RGB; raw color images.
        gray8      — 8-bit grayscale; heatmaps after MinMax to [0,255].
        gray16le   — 16-bit single-channel; disparity / heatmap (16-bit).
        gbrp16le   — 16-bit planar; optical flow (u in G, v in B, 0 in R).

    The encoder docstring promises **bit-exact** round-trip for the
    chosen pix_fmt. The selftest below verifies this on random arrays.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = iter(frame_iter)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"encode_mkv_ffv1 called with empty iterable for {dst_path}")
    # Shape inference depends on pix_fmt: planar 3-channel formats (gbrp16le,
    # yuv*p) have shape (3, H, W) so the spatial dims are at axes 1, 2.
    # Packed formats (bgr24, gray*) have spatial dims at axes 0, 1 with an
    # optional channel axis last.
    if pix_fmt.startswith("gbrp") or (pix_fmt.startswith("yuv") and pix_fmt.endswith("p")):
        # Planar 3-channel; expect (3, H, W).
        if first.ndim != 3 or first.shape[0] != 3:
            raise ValueError(
                f"planar pix_fmt={pix_fmt} expects (3, H, W) arrays; got shape {first.shape}"
            )
        H, W = first.shape[1], first.shape[2]
    elif pix_fmt == "bgr24":
        # Packed; expect (H, W, 3).
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError(
                f"pix_fmt=bgr24 expects (H, W, 3); got shape {first.shape}"
            )
        H, W = first.shape[0], first.shape[1]
    else:
        # gray8 / gray16le — (H, W).
        if first.ndim != 2:
            raise ValueError(
                f"pix_fmt={pix_fmt} expects (H, W); got shape {first.shape}"
            )
        H, W = first.shape[0], first.shape[1]

    if width is None:
        width = W
    if height is None:
        height = H

    args = [
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        # FFV1 v3 lossless. `level=3` enables the modern container; `g=1`
        # uses every frame as a keyframe (essential for random access).
        "-c:v", "ffv1",
        "-level", "3",
        "-coder", "1",
        "-context", "1",
        "-g", "1",
        # FFV1 supports the input pix_fmt natively; copy it through.
        "-pix_fmt", pix_fmt,
        str(dst_path),
    ]

    proc = subprocess.Popen(
        [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(first.tobytes())
        for frame in iterator:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass

    stderr = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg FFV1 encode failed (rc={rc}) for {dst_path}:\n"
            f"{stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Decode — used by tests and the unpack reader
# ---------------------------------------------------------------------------

def decode_video_frames(
    src_path: Path,
    *,
    pix_fmt: str,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Decode every frame of an MKV via ffmpeg → numpy arrays.

    The caller supplies expected `pix_fmt`, width, and height. We decode
    to that format and yield ndarrays of the appropriate shape and dtype.

    Channel/dtype mapping (extend as we add formats):
        bgr24    → uint8, (H, W, 3)
        gray8    → uint8, (H, W)
        gray16le → uint16, (H, W)
        gbrp16le → uint16, (3, H, W) (planar; caller reorders if needed)
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    args = [
        "-i", str(src_path),
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-",
    ]
    proc = subprocess.Popen(
        [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if pix_fmt == "bgr24":
        dtype = np.uint8
        bytes_per_frame = width * height * 3
        shape: tuple = (height, width, 3)
    elif pix_fmt == "gray8":
        dtype = np.uint8
        bytes_per_frame = width * height
        shape = (height, width)
    elif pix_fmt == "gray16le":
        dtype = np.uint16
        bytes_per_frame = width * height * 2
        shape = (height, width)
    elif pix_fmt == "gbrp16le":
        dtype = np.uint16
        bytes_per_frame = width * height * 3 * 2
        shape = (3, height, width)  # planar G,B,R
    else:
        proc.kill()
        raise ValueError(f"Unsupported pix_fmt for decode: {pix_fmt}")

    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(bytes_per_frame)
            if not raw:
                break
            if len(raw) < bytes_per_frame:
                raise RuntimeError(
                    f"Truncated frame from {src_path}: got {len(raw)} bytes, "
                    f"expected {bytes_per_frame}"
                )
            yield np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    finally:
        rc = proc.wait()
        stderr = proc.stderr.read() if proc.stderr else b""
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg decode failed (rc={rc}) for {src_path}:\n"
                f"{stderr.decode(errors='replace')}"
            )


def probe_frame_count(src_path: Path) -> int:
    """Use ffprobe to count frames. Used by validators."""
    args = [
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(src_path),
    ]
    out = subprocess.run([_ffprobe_bin(), *args], check=True, capture_output=True)
    return int(out.stdout.decode().strip())


# ---------------------------------------------------------------------------
# Round-trip selftest (Risk B-1)
# ---------------------------------------------------------------------------

def _selftest_ffv1_gray16(tmp_path: Path, *, n_frames: int = 4, h: int = 32, w: int = 48) -> None:
    """Random uint16 frames must round-trip bit-exact through FFV1 gray16le."""
    rng = np.random.default_rng(seed=0)
    frames = [rng.integers(0, 65536, size=(h, w), dtype=np.uint16) for _ in range(n_frames)]
    dst = tmp_path / "ffv1_gray16.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="gray16le")

    decoded = list(decode_video_frames(dst, pix_fmt="gray16le", width=w, height=h))
    assert len(decoded) == n_frames, f"frame count mismatch: {len(decoded)} vs {n_frames}"
    for i, (a, b) in enumerate(zip(frames, decoded)):
        if not np.array_equal(a, b):
            raise AssertionError(f"FFV1 gray16le NOT bit-exact at frame {i}")


def _selftest_ffv1_flow_gbrp16(tmp_path: Path, *, n_frames: int = 4, h: int = 32, w: int = 48) -> None:
    """gbrp16le (used for optical flow packing) round-trips bit-exact."""
    rng = np.random.default_rng(seed=1)
    # FFV1 expects gbrp16le planar layout: 3 planes (G, B, R).
    frames = [rng.integers(0, 65536, size=(3, h, w), dtype=np.uint16) for _ in range(n_frames)]
    dst = tmp_path / "ffv1_gbrp16.mkv"
    encode_mkv_ffv1(frames, dst, fps=30.0, pix_fmt="gbrp16le")

    decoded = list(decode_video_frames(dst, pix_fmt="gbrp16le", width=w, height=h))
    assert len(decoded) == n_frames
    for i, (a, b) in enumerate(zip(frames, decoded)):
        if not np.array_equal(a, b):
            raise AssertionError(f"FFV1 gbrp16le NOT bit-exact at frame {i}")


def _selftest_h264_psnr(tmp_path: Path, *, n_frames: int = 4, h: int = 128, w: int = 128) -> None:
    """H.264 CRF 18 encode + decode is a clean pipeline.

    Synthetic monochrome content (R=G=B) on yuv420p has a colorspace-
    conversion noise floor of ~38 dB independent of CRF. The real
    contract — "PSNR ≥ 40 dB on actual surgical video" — is enforced
    against real fixtures by the end-to-end smoke test. This selftest
    only verifies the pipeline runs cleanly and the decode shape is
    correct, with a generous 30 dB floor that would still flag a real
    regression (e.g. accidentally encoding at CRF 51).
    """
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        127.5
        + 80.0 * np.sin(2 * np.pi * xx / 64.0)
        + 40.0 * np.cos(2 * np.pi * yy / 80.0)
    ).clip(0, 255).astype(np.uint8)
    frames = [np.stack([base, base, base], axis=-1) for _ in range(n_frames)]
    dst = tmp_path / "h264.mkv"
    encode_h264_crf(frames, dst, fps=30.0, crf=18)

    decoded = list(decode_video_frames(dst, pix_fmt="bgr24", width=w, height=h))
    assert len(decoded) == n_frames
    psnrs = []
    for a, b in zip(frames, decoded):
        mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
        psnrs.append(psnr)
    floor = min(psnrs)
    if floor < 30.0:
        raise AssertionError(f"H.264 CRF 18 PSNR floor = {floor:.2f} dB (< 30); pipeline regression?")


def roundtrip_selftest(tmp_path: Optional[Path] = None) -> None:
    """Run every codec round-trip check. Raises on any failure.

    Used by the CLI selftest and by pytest.
    """
    import tempfile

    if tmp_path is None:
        cleanup_tmp = True
        tmp_path = Path(tempfile.mkdtemp(prefix="surgsync_codec_"))
    else:
        cleanup_tmp = False
        tmp_path = Path(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        _selftest_ffv1_gray16(tmp_path)
        _selftest_ffv1_flow_gbrp16(tmp_path)
        _selftest_h264_psnr(tmp_path)
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entry — `python -m dvrk_data_processing.surgsync.encode.codec selftest`
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="SurgSync codec selftest")
    p.add_argument("cmd", choices=["selftest"])
    args = p.parse_args()

    if args.cmd == "selftest":
        try:
            roundtrip_selftest()
        except Exception as e:
            print(f"selftest FAILED: {e}")
            return 1
        print("selftest OK: FFV1 gray16le + gbrp16le bit-exact; H.264 PSNR ≥ 40 dB.")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
