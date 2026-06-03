"""Smoke + round-trip tests for the unpack decomposer.

These tests assume the repo's sample raw clips live at
`data/{offline_data/3, online_data/2}` and that the matching packed
release sits on the external SSD. The packed-data tests are
auto-skipped when the SSD isn't mounted (so CI / clean checkouts
don't have to carry it).

Test scope:
1. `Episode` + `Dataset` open without touching anything beyond
   `episode_meta.json` and the tree walk respectively.
2. `decompose(..., streams=("raw",))` over one clip produces every
   raw artifact bucket.
3. The decomposed `image/<cam>/*.png` files are **bit-exact** versus
   the raw clip's originals (FFV1 round-trip contract).
4. The decomposed `annotation/{phase,step,gesture}/<i>.json` carries
   English text — proves the verbalization-only contract the user
   requested.
5. The fidelity tags on the per-clip report match the documented
   defaults (`bit_exact` for image / calibration, `text_form` for
   annotation, `reconstructed` for time_syn).
"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import dvrk_data_processing.surgsync as surgsync


REPO = Path(__file__).resolve().parents[3]
PACKED_ROOT  = Path("<release_root>")
RAW_ONLINE_2 = REPO / "data" / "online_data" / "2"


pytestmark = pytest.mark.skipif(
    not PACKED_ROOT.is_dir(),
    reason="sample packed dataset not present at "
           f"{PACKED_ROOT}; mount the SSD or build a dataset first.",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pixels_equal(a: Path, b: Path) -> bool:
    """Decode two PNG files and compare pixel-by-pixel.

    File-byte hash comparison would fail trivially because PNG
    compression (filter selection + chunk ordering) is not
    deterministic across encoders, even when the pixel data is
    identical. The packer writes its PNGs through cv2; the original
    raw clip was written by a different encoder. We care about the
    pixel content surviving the round-trip — not the byte sequence.
    """
    ia = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)
    ib = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)
    if ia is None or ib is None:
        return False
    if ia.shape != ib.shape:
        return False
    return np.array_equal(ia, ib)


@pytest.fixture(scope="module")
def packed_dataset() -> surgsync.Dataset:
    return surgsync.open_dataset(PACKED_ROOT)


def test_open_dataset_finds_sample_clips(packed_dataset: surgsync.Dataset):
    """Constructing the dataset only walks the tree and reads
    `meta/dataset.json` — no per-episode parquet load."""
    keys = {ep.key for ep in packed_dataset.episodes}
    # At minimum the two known sample clips should be discoverable.
    assert any("online_data/" in k and k.endswith("/2") for k in keys)
    assert any("offline_data/" in k and k.endswith("/3") for k in keys)


def test_open_episode_only_reads_meta(packed_dataset: surgsync.Dataset):
    """`open_episode` touches `episode_meta.json` only — the parquet
    tables are loaded lazily."""
    ep_ref = next(e for e in packed_dataset.episodes if e.clip_index == "2")
    ep = surgsync.open_episode(ep_ref.path)
    # Probe the lazy-cache attributes — they should still be None
    # before any property access.
    assert ep._timestamp_table is None
    assert ep._psm1_table is None
    # Touching a property triggers the load.
    _ = ep.timestamps
    assert ep._timestamp_table is not None


def test_decompose_raw_only_one_clip_round_trip(tmp_path: Path):
    """End-to-end: decompose one clip's raw domain, then check the
    most important round-trip claims (image bit-exact, annotation
    text-form, time_syn reconstructed)."""
    out_root = tmp_path / "unpack"
    report = surgsync.decompose(
        dataset_root=PACKED_ROOT,
        out_root=out_root,
        clips=["online_data/2"],
        streams=("raw",),
        force=True,
        parallelism=1,
        workers_per_clip=2,
    )
    assert report.n_episodes_ok == 1, report.clips[0].error
    clip_dir = out_root / "online_data" / "2"

    # ---- structure: every raw bucket present --------------------------
    for sub in ("image/left", "image/right", "kinematic/PSM1",
                "annotation/phase", "annotation/step",
                "annotation/contact_detection",
                "time_syn", "camera_calibration", "hand_eye_calibration"):
        assert (clip_dir / sub).is_dir(), f"missing {sub}"
    assert (clip_dir / "meta_data.json").is_file()

    # ---- annotation is text-form, not numeric id ---------------------
    phase_files = sorted((clip_dir / "annotation" / "phase").glob("*.json"))[:3]
    assert phase_files, "no phase JSONs emitted"
    for p in phase_files:
        d = json.loads(p.read_text())
        text = d["phase"]
        # User contract: workflow JSONs present text descriptions, not
        # bare ids. Reject pure-digit ids ("1", "2") — any reasonable
        # description has spaces in it.
        assert text is None or " " in text, (
            f"phase JSON {p} carries id-like value {text!r}; expected text description"
        )

    # ---- raw image bit-exact vs the source clip's PNGs ----------------
    # We compare pixels (not file bytes) — PNG file bytes differ
    # because cv2's encoder picks different filters than whatever
    # produced the original raw PNGs. The contract is "every pixel
    # survives", which IS bit-exact at the pixel level.
    if RAW_ONLINE_2.is_dir():
        sample_pngs = sorted((RAW_ONLINE_2 / "image" / "left").glob("*.png"))[:3]
        for raw_png in sample_pngs:
            unp = clip_dir / "image" / "left" / raw_png.name
            assert unp.is_file(), f"missing decompose output {unp}"
            assert _pixels_equal(raw_png, unp), (
                f"image PNG {raw_png.name} not pixel-exact after pack+unpack"
            )

    # ---- time_syn looks structurally sane ----------------------------
    ts_files = sorted((clip_dir / "time_syn").glob("*.json"))[:1]
    assert ts_files
    ts = json.loads(ts_files[0].read_text())
    assert "image_left_stamp" in ts
    assert "Kinematics_set_1" in ts
    assert "PSM1" in ts["Kinematics_set_1"]
    assert "measured_data" in ts["Kinematics_set_1"]["PSM1"]

    # ---- meta_data.json carries the known raw keys -------------------
    md = json.loads((clip_dir / "meta_data.json").read_text())
    for key in ("user_id", "operator_skill_level", "case_type", "tool",
                "failure", "recovery"):
        assert key in md

    # ---- fidelity tags in the report match the documented defaults ---
    fidelity = report.clips[0].fidelity
    assert fidelity["image"] == "bit_exact"
    assert fidelity["annotation"] == "text_form"
    assert fidelity["time_syn"] == "reconstructed"
    assert fidelity["calibration"] == "bit_exact"


def test_decompose_preprocess_only_one_clip(tmp_path: Path):
    """Smoke: --streams preprocess produces only the preprocess tree.

    Doesn't check fidelity in depth (covered by the encoder selftest);
    just verifies the writer emits at least one PNG in each documented
    bucket and that the raw domain bucket stays empty.
    """
    out_root = tmp_path / "unpack_pp"
    report = surgsync.decompose(
        dataset_root=PACKED_ROOT,
        out_root=out_root,
        clips=["online_data/2"],
        streams=("preprocess",),
        force=True,
        parallelism=1,
        workers_per_clip=2,
    )
    assert report.n_episodes_ok == 1, report.clips[0].error
    clip_dir = out_root / "online_data" / "2"

    # raw image/ should NOT have been written
    assert not (clip_dir / "image").exists()
    # preprocess subdirs should exist
    for sub in ("rectify_resize/image/left",
                "depth_estimation/depth_image",
                "optical_flow/left/image",
                "kinematic_reproject/PSM1/left/image"):
        d = clip_dir / "preprocess" / sub
        assert d.is_dir() and any(d.glob("*.png")), f"missing PNGs under {d}"


def test_force_required_when_output_populated(tmp_path: Path):
    """Without `force`, decompose must refuse to overwrite a populated
    output dir."""
    out_root = tmp_path / "unpack"
    out_clip = out_root / "online_data" / "2"
    out_clip.mkdir(parents=True)
    (out_clip / "dummy.txt").write_text("not empty")

    report = surgsync.decompose(
        dataset_root=PACKED_ROOT,
        out_root=out_root,
        clips=["online_data/2"],
        streams=("raw",),
        force=False,
        parallelism=1,
    )
    assert report.n_episodes_fail == 1
    assert "force=True" in (report.clips[0].error or "")


def test_resume_skip_via_sentinel(tmp_path: Path):
    """A second run with `force=False` should skip clips whose output
    carries the `.surgsync_unpacked.json` sentinel — no decode, no
    write, near-instant return. Critical for large sweeps where we
    expect to interrupt + resume.
    """
    out_root = tmp_path / "unpack"
    out_clip = out_root / "online_data" / "2"
    out_clip.mkdir(parents=True)
    # Drop a populated dir with the sentinel — the orchestrator should
    # treat this as "already done" without touching anything.
    (out_clip / "old.txt").write_text("leftover")
    (out_clip / ".surgsync_unpacked.json").write_text(
        json.dumps({"manually_stamped": True})
    )

    import time as _time
    started = _time.time()
    report = surgsync.decompose(
        dataset_root=PACKED_ROOT,
        out_root=out_root,
        clips=["online_data/2"],
        streams=("raw", "preprocess"),
        force=False,
        parallelism=1,
    )
    elapsed = _time.time() - started

    assert report.n_episodes_skipped == 1
    assert report.n_episodes_ok == 1
    assert report.n_episodes_fail == 0
    assert report.clips[0].skipped is True
    # Resume should be near-instant — well under the ~minutes a real
    # unpack would take. Generous bound to avoid flakes on busy boxes.
    assert elapsed < 10.0, f"resume took {elapsed:.1f}s — sentinel skip not working?"
    # And the pre-existing file should still be there (we didn't wipe).
    assert (out_clip / "old.txt").is_file()


def test_collision_detection_aborts_before_writing(tmp_path: Path):
    """Two episodes packed under the same `(dataset, clip_index)` across
    different task folders must abort the orchestrator before any
    decode/write happens — otherwise the second would silently
    overwrite the first in the unpacked tree.
    """
    # Build a synthetic dataset_root inline so this test stands alone
    # without depending on the SSD-mounted fixture.
    root = tmp_path / "fake_release"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "dataset.json").write_text(json.dumps({
        "name": "fake", "schema_version": "1.0.0",
    }))
    for task in ("taskA", "taskB"):
        clip_dir = root / "X" / "episodes" / task / "1"
        clip_dir.mkdir(parents=True)
        (clip_dir / ".surgsync_complete.json").write_text("{}")
        (clip_dir / "episode_meta.json").write_text(json.dumps({
            "schema_version": "1.0.0",
            "episode_id": f"fake_{task}_1",
            "task": task,
            "length_frames": 0,
            "master_t0_ns": 0,
            "recorder_variant": "online",
            "source_clip": "data/X/1/",
        }))

    out_root = tmp_path / "out"
    with pytest.raises(ValueError, match="appear under multiple task folders"):
        surgsync.decompose(
            dataset_root=root,
            out_root=out_root,
        )
    # Nothing should have been written before the abort.
    assert not (out_root / "X").exists()


