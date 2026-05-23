"""Per-clip orchestrator (`code_design.md` § 5.2).

`convert_clip` takes a RawClip + Hydra build cfg + dataset root and
produces one finalized episode directory. Atomic: either the final
episode dir exists with every artifact, or nothing changes at the
target location (failures leave staging dirs behind for inspection).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dvrk_data_processing.surgsync.align import align_clip, TolerancePolicy
from dvrk_data_processing.surgsync.encode.calibration import write_calibration
from dvrk_data_processing.surgsync.encode.episode_meta import (
    compute_episode_id, write_episode_meta,
)
from dvrk_data_processing.surgsync.encode.per_modality_parquet import write_all_per_modality
from dvrk_data_processing.surgsync.encode.preprocess import write_preprocess
from dvrk_data_processing.surgsync.encode.preview import write_preview
from dvrk_data_processing.surgsync.encode.time_sync_stat import write_time_sync_stat
from dvrk_data_processing.surgsync.encode.video_processed import write_processed_videos
from dvrk_data_processing.surgsync.encode.video_raw import write_raw_videos
from dvrk_data_processing.surgsync.ingest.annotations import load_annotations
from dvrk_data_processing.surgsync.ingest.calibrated_kinematic import (
    load_calibrated_arm, aligned_to_master,
)
from dvrk_data_processing.surgsync.ingest.calibration import load_calibration_bundle
from dvrk_data_processing.surgsync.ingest.clip import RawClip
from dvrk_data_processing.surgsync.ingest.kinematics import load_arm
from dvrk_data_processing.surgsync.ingest.meta import load_meta
from dvrk_data_processing.surgsync.ingest.timestamps import load_timestamps
from dvrk_data_processing.surgsync.pipeline.errors import (
    MissingPreprocessingOutputError, RecoverableExportError,
)
from dvrk_data_processing.surgsync.pipeline.staging import (
    episode_staging, episode_final_dir, finalize_episode, is_episode_complete,
)


log = logging.getLogger(__name__)


@dataclass
class ConvertedEpisode:
    """Result handed back by `convert_clip` for the structured log."""
    episode_id: str
    task: str
    dataset_name: str
    recorder_variant: str
    clip_index: str
    final_path: Path
    length_frames: int
    duration_s: float


def convert_clip(
    clip: RawClip,
    *,
    dataset_root: Path,
    task: str,
    align_cfg,
    encode_cfg,
    fps: float,
    include_video_processed: bool,
    include_video_raw: bool,
    include_preprocess: bool,
    include_preview: bool,
    force: bool,
    clean_staging: bool,
) -> ConvertedEpisode:
    """Pack one clip into one finalized episode.

    Raises:
        MissingPreprocessingOutputError — if `clip.intermediate_present` is False
            and `include_video_processed` is True (rectified video can't
            be encoded without the preprocessing intermediate output).
        RecoverableExportError — for any other per-clip failure that
            should be logged + skipped by the per-release sweep.
    """
    # 1. Pre-flight checks for preprocessing outputs we'll actually read.
    if include_video_processed and not clip.intermediate_present:
        raise MissingPreprocessingOutputError(
            f"Required preprocessing output missing for {clip.source_clip_str}: "
            f"{clip.intermediate_dir}/image/left/ does not exist. "
            "Run `python scripts/run_all_stages.py` (stage 1) first."
        )
    if include_preprocess and not any(clip.processed_present.values()):
        raise MissingPreprocessingOutputError(
            f"include_preprocess=True but no processed/ outputs for {clip.source_clip_str}. "
            "Run `python scripts/run_all_stages.py` first."
        )

    # 2. Ingest
    log.info("[%s] ingesting", clip.source_clip_str)
    ts = load_timestamps(clip.time_syn_dir)
    kin_ECM  = load_arm(clip.kinematic_dir, "ECM")
    kin_PSM1 = load_arm(clip.kinematic_dir, "PSM1")
    kin_PSM2 = load_arm(clip.kinematic_dir, "PSM2")
    ann = load_annotations(clip.annotation_dir)
    clip_meta = load_meta(clip.meta_path)
    # Calibration: always load from raw_dir. `intermediate_dir` is
    # only consulted for the optional rectify_params.json (preprocessing
    # stage 1 output) — when preprocessing hasn't run, the calibration
    # bundle ships raw-only, which is still complete for downstream
    # consumers.
    calib = load_calibration_bundle(
        clip.raw_dir,
        clip.intermediate_dir if clip.intermediate_present else None,
    )

    # 3. Align
    log.info("[%s] aligning", clip.source_clip_str)
    policy = TolerancePolicy.from_align_cfg(clip.recorder_variant, align_cfg)
    aligned = align_clip(
        ts,
        kin_ECM=kin_ECM, kin_PSM1=kin_PSM1, kin_PSM2=kin_PSM2,
        annotations=ann, policy=policy,
    )

    # 4. Episode ID + skip-existing check
    # The on-disk path uses the human-readable dataset/task/clip_index;
    # `episode_id` is the time-based composite stored inside
    # `episode_meta.json` for cross-reference. It encodes the
    # **absolute** t0 so two clips that both start at relative 0
    # still get distinct ids.
    episode_id = compute_episode_id(
        clip.dataset_name, clip.clip_index, aligned.master_t0_ns,
    )
    final_dir = episode_final_dir(dataset_root, clip.dataset_name, task, clip.clip_index)
    # Skip only if the dir is COMPLETE (sentinel present). An
    # incomplete dir (crashed prior run) won't short-circuit — we'd
    # rather repack than serve a half-finished episode.
    if is_episode_complete(final_dir) and not force:
        log.info("[%s] already finalized at %s — skip (force=false)",
                 clip.source_clip_str, final_dir)
        return ConvertedEpisode(
            episode_id=episode_id, task=task,
            dataset_name=clip.dataset_name,
            recorder_variant=clip.recorder_variant, clip_index=clip.clip_index,
            final_path=final_dir,
            length_frames=aligned.n_frames,
            # master_ns is rebased so its last value is duration in ns.
            duration_s=float(aligned.master_ns[-1] / 1e9),
        )
    if final_dir.exists() and force:
        import shutil as _sh
        log.warning("[%s] force=true — wiping existing %s", clip.source_clip_str, final_dir)
        _sh.rmtree(final_dir)

    # 5. Encode directly into the final dir; finalize stamps the
    # completion manifest on success. `episode_staging` writes a
    # `.surgsync_running.json` marker on entry (so an interrupted
    # pack is observably distinct from a never-started one), and on
    # exception writes `.surgsync_failed.json`. No intermediate
    # staging+rename — see `pipeline/staging.py` docstring for why.
    with episode_staging(
        dataset_root, clip.dataset_name, task, clip.clip_index,
        episode_id=episode_id,
        clean_existing=clean_staging,
    ) as staging:
        # Image size — pulled from the raw camera YAML, which is the
        # native capture resolution. `video_raw/*.mkv` encodes the raw
        # PNGs verbatim, so this is the right size for round-trip.
        # (The H.264 `video/*.mp4` lives at the rectified resolution;
        # consumers read that size from rectify_params.json when
        # present.)
        image_size = [calib.camera.image_width, calib.camera.image_height]

        # Processed videos (H.264, visually lossless). Gated on
        # `intermediate_present` because the source PNGs come from
        # the preprocessing rectify_resize stage — without it there's
        # nothing to encode here. (The earlier MissingPreprocessingOutputError
        # check would have already returned if
        # include_video_processed=True and intermediate is missing,
        # so the AND below is defense in depth.)
        if include_video_processed and clip.intermediate_present:
            log.info("[%s] encoding processed video", clip.source_clip_str)
            write_processed_videos(
                clip.intermediate_dir, staging / "video",
                fps=fps, crf=int(encode_cfg.h264.crf),
            )

        # Raw videos (FFV1, bit-exact — MANDATORY for decomposability)
        if include_video_raw:
            log.info("[%s] encoding raw video (FFV1)", clip.source_clip_str)
            write_raw_videos(
                clip.raw_dir, staging / "video_raw",
                fps=fps, side_dir_name=clip.side_dir_name,
            )

        # Preprocess outputs (FFV1 8-bit bit-exact from preprocessing visualization PNGs).
        # Goes under `<episode>/preprocess/` — matches the preprocessing source-side
        # folder name (`<raw_dir>/preprocess/`) so the verbiage is consistent
        # across the two halves of the pipeline.
        preprocess_written: dict = {}
        if include_preprocess and any(clip.processed_present.values()):
            log.info("[%s] encoding preprocess streams", clip.source_clip_str)
            preprocess_written = write_preprocess(
                clip.processed_dir, staging / "preprocess",
                fps=fps, n_frames_expected=aligned.n_frames,
            )

        # Calibration
        if calib is not None:
            log.info("[%s] writing calibration", clip.source_clip_str)
            write_calibration(calib, staging / "calibration")

        # Preview (Option C — stub for MVP)
        if include_preview:
            write_preview(clip.processed_dir, staging / "preview", fps=fps)

        # Calibrated kinematic (optional modality). The preprocessing
        # hand-eye stage emits per-frame JSON for PSM1/PSM2 in the
        # camera frame; we ingest and project onto the master timeline
        # so the per-PSM parquets can populate
        # `measured_cp_calibrated.*` and `setpoint_cp_calibrated.*`
        # columns. When the folder is absent (e.g. preprocessing's
        # calibrated_kinematic was not enabled, or the dVRK variant
        # was used), the lookup yields empty tables and the columns
        # stay NULL.
        cal_psm1 = load_calibrated_arm(clip.processed_dir, "PSM1")
        cal_psm2 = load_calibrated_arm(clip.processed_dir, "PSM2")
        cal_psm1_meas, cal_psm1_set = aligned_to_master(cal_psm1, aligned.source_frame_index)
        cal_psm2_meas, cal_psm2_set = aligned_to_master(cal_psm2, aligned.source_frame_index)
        has_calibrated_kinematic = any(s is not None for s in cal_psm1_meas) or \
                                   any(s is not None for s in cal_psm2_meas)

        # Per-modality parquets: timestamp + ECM + PSM1 + PSM2 + annotation.
        log.info("[%s] writing per-modality parquets%s",
                 clip.source_clip_str,
                 " (with calibrated kinematics)" if has_calibrated_kinematic else "")
        write_all_per_modality(
            aligned, staging,
            task=task,
            calibrated_psm1_measured=cal_psm1_meas,
            calibrated_psm1_setpoint=cal_psm1_set,
            calibrated_psm2_measured=cal_psm2_meas,
            calibrated_psm2_setpoint=cal_psm2_set,
        )

        # episode_meta.json
        log.info("[%s] writing episode_meta.json", clip.source_clip_str)
        write_episode_meta(
            clip_meta=clip_meta,
            aligned=aligned,
            task=task,
            source_clip=clip.source_clip_str,
            recorder_variant=clip.recorder_variant,
            dataset_name=clip.dataset_name,
            clip_index=clip.clip_index,
            image_size=image_size,
            has_preprocess=bool(preprocess_written),
            has_preview=bool(include_preview),
            has_video_raw=bool(include_video_raw),
            has_calibrated_kinematic=has_calibrated_kinematic,
            dst_path=staging / "episode_meta.json",
        )

        # time_sync_stat.json — per-topic latency detail (median, mean,
        # std, max in ms, plus the master-frame index of each topic's
        # worst-aligned frame and the count of present stamps).
        # Sibling file to episode_meta.json; consumers who only need
        # the cross-modal summary can stay with episode_meta.
        log.info("[%s] writing time_sync_stat.json", clip.source_clip_str)
        write_time_sync_stat(
            aligned=aligned,
            episode_id=episode_id,
            dst_path=staging / "time_sync_stat.json",
        )

        # modalities.json — per-episode manifest of every stream that
        # made it into the pack (presence + codec/format + frame
        # counts + on-disk sizes). Written last so it sees the final
        # filesystem state.
        from dvrk_data_processing.surgsync.encode.modalities import write_modalities_json
        log.info("[%s] writing modalities.json", clip.source_clip_str)
        write_modalities_json(staging, expected_frames=aligned.n_frames,
                              episode_id=episode_id)

        # 6. Mark complete — atomically stamps the
        # `.surgsync_complete.json` manifest and clears the running
        # marker. Single small-file rename, fast on every filesystem.
        # Visible to readers as soon as it returns; an in-flight clip
        # has only a running marker and is ignored by scanners.
        finalize_episode(
            staging, dataset_root, clip.dataset_name, task, clip.clip_index,
            episode_id=episode_id,
            length_frames=aligned.n_frames,
            duration_s=float(aligned.master_ns[-1] / 1e9),
            extra={
                "source_clip": clip.source_clip_str,
                "recorder_variant": clip.recorder_variant,
            },
        )

    return ConvertedEpisode(
        episode_id=episode_id,
        task=task,
        dataset_name=clip.dataset_name,
        recorder_variant=clip.recorder_variant,
        clip_index=clip.clip_index,
        final_path=final_dir,
        length_frames=aligned.n_frames,
        # master_ns is rebased so its last value is duration in ns.
        duration_s=float(aligned.master_ns[-1] / 1e9),
    )
