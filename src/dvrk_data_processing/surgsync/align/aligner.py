"""End-to-end align (`code_design.md` § 4.5).

`align_clip` runs the master clock derivation, per-topic delta
computation, contiguity detection, and assembles the `AlignedClip`
ready for the encoder.

Every synced modality (image streams + per-arm kinematic topics) is
driven from the central catalog at `align.topics.TIMESTAMP_TOPICS` —
adding a new topic touches only that file plus the timestamp schema.
The aligner is table-driven from the catalog; it never references a
topic name literally.

"No silent NULLs" rule: a delta value is NULL only when the **source
modality stamp itself is missing** (e.g. offline recorder's
`setpoint_cp_stamp`). Out-of-tolerance frames are NOT NULL'd — the
real signed delta is preserved; the tolerance assessment surfaces as
the `out_of_tol_counts` summary only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dvrk_data_processing.surgsync.align.master_clock import build_master_timeline
from dvrk_data_processing.surgsync.align.policy import TolerancePolicy
from dvrk_data_processing.surgsync.align.contiguity import detect_contiguity
from dvrk_data_processing.surgsync.align.topics import (
    KINEMATIC_TOPICS, TIMESTAMP_TOPICS,
)
from dvrk_data_processing.surgsync.ingest.timestamps import TimestampTable, NULL_TS
from dvrk_data_processing.surgsync.ingest.kinematics import ArmKinematics
from dvrk_data_processing.surgsync.ingest.annotations import AnnotationTable
from dvrk_data_processing.surgsync.serde.kinematic_io import KinematicSample
from dvrk_data_processing.surgsync.serde.annotation_io import AnnotationSample


# ---------------------------------------------------------------------------
# AlignedClip dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyncStats:
    """Sync quality statistics surfaced into episode_meta.json.

    Lightweight per-episode summary. The detailed per-topic latency
    breakdown (median / mean / std / max + max_delta_frame_idx +
    n_present for each topic) lives in `time_sync_stat.json`.

    `median_kin_delta_ms` / `max_kin_delta_ms` aggregate the
    kinematic-only topics (every per-arm topic in
    `align.topics.KINEMATIC_TOPICS`).
    `cross_modal_{median,mean,std,max}_delta_ms` aggregate over
    **every** synced topic — image streams + every kinematic topic.
    All four cross-modal stats are computed from the same pooled
    |delta| array, so they describe one population. `std` uses ddof=0
    (population standard deviation).

    Null semantics: a topic with zero present stamps contributes
    nothing to either aggregate. When every topic is empty, the
    summaries are `None`.

    `episode_length_s` mirrors `EpisodeMeta.duration_s` so callers
    reading sync_stats in isolation have the full picture.
    """
    episode_length_s: float
    median_kin_delta_ms: Optional[float]
    max_kin_delta_ms: Optional[float]
    cross_modal_median_delta_ms: Optional[float]
    cross_modal_mean_delta_ms: Optional[float]
    cross_modal_std_delta_ms: Optional[float]
    cross_modal_max_delta_ms: Optional[float]
    frames_dropped: int
    contiguity_ratio: float
    # Per-topic out-of-tolerance counters. Keys are the canonical topic
    # names from `align.topics.TIMESTAMP_TOPICS`.
    out_of_tol_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class AlignedClip:
    """All per-frame columns ready for the parquet writer.

    `n_frames` = N. Each list/array is length N. Per-arm state lists
    are `list[Optional[KinematicSample]]` so the encoder can emit
    Arrow NULLs cleanly.

    `master_ns` is rebased to clip start (`master_ns[0] == 0`); the
    original absolute t0 lives on `master_t0_ns`. Absolute time of
    row i is `master_t0_ns + master_ns[i]`.

    `topic_deltas[topic]` is the signed delta-to-master array for that
    topic (length N, NULL_DELTA where the source stamp is missing).
    Keys are exactly `align.topics.TOPIC_NAMES`. The encoder writes
    every entry into `timestamp.parquet` as `delta_to_master.<name>_ns`
    (int32). There are NO per-arm-parquet delta columns anymore —
    deltas are consolidated in timestamp.parquet.
    """
    n_frames: int
    master_ns: np.ndarray             # int64 (N,), rebased so master_ns[0] == 0
    master_t0_ns: int                 # absolute ns of master_ns[0] before rebase
    source_frame_index: np.ndarray    # int32 (N,)

    # Contiguity flags
    is_contiguous_to_prev: np.ndarray  # bool (N,)
    drop_count_since_prev: np.ndarray  # int8 (N,)

    # Per-topic deltas — keys = align.topics.TOPIC_NAMES. int64 arrays
    # with NULL_DELTA where the source stamp was missing.
    topic_deltas: dict[str, np.ndarray]

    # Per-arm aligned KinematicSample lists (length N, None where absent).
    ECM_aligned: list[Optional[KinematicSample]]
    PSM1_aligned: list[Optional[KinematicSample]]
    PSM2_aligned: list[Optional[KinematicSample]]

    # Aligned AnnotationSample list (length N, None where absent).
    annotations: list[Optional[AnnotationSample]]

    sync_stats: SyncStats
    # Per-topic latency detail destined for `time_sync_stat.json`.
    # Keys = TOPIC_NAMES. Each value is a dict with:
    #   median_delta_ms / mean_delta_ms / std_delta_ms / max_delta_ms
    #       Optional[float]; None when n_present == 0.
    #   max_delta_frame_idx: Optional[int] — master-timeline row
    #       where the max |delta| occurred. None when n_present == 0.
    #   n_present: int — count of master frames whose source stamp
    #       was present for this topic.
    per_topic_latency: dict[str, dict] = field(default_factory=dict)


# NULL sentinel for the per-topic delta arrays. Same value as
# `ingest.timestamps.NULL_TS`. Used only when the source stamp itself
# is missing (e.g. offline recorder's setpoint_cp_stamp); out-of-
# tolerance frames preserve the real signed delta.
NULL_DELTA = NULL_TS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delta_preserve_with_null_for_missing(
    deltas: np.ndarray, cand_ns: np.ndarray,
) -> np.ndarray:
    """Return the signed delta as int64, with NULL_DELTA at rows whose
    source stamp was NULL_TS. Caller computes `deltas` against the
    master timeline."""
    out = deltas.astype(np.int64, copy=True)
    out[cand_ns == NULL_TS] = NULL_DELTA
    return out


# ---------------------------------------------------------------------------
# align_clip — the orchestrator
# ---------------------------------------------------------------------------

def align_clip(
    ts: TimestampTable,
    *,
    kin_ECM: ArmKinematics,
    kin_PSM1: ArmKinematics,
    kin_PSM2: ArmKinematics,
    annotations: AnnotationTable,
    policy: TolerancePolicy,
) -> AlignedClip:
    """Compose master timeline + per-topic deltas + contiguity into an
    AlignedClip ready for the encoder.

    Topic catalog comes from `align.topics.TIMESTAMP_TOPICS` — this
    function iterates over the catalog; it never references a topic
    name literally. To add or remove a topic, edit `topics.py` (and
    nothing else inside this function).
    """
    # 1. Master timeline (absolute ns at this point).
    master = build_master_timeline(ts)
    N = master.master_ns.shape[0]

    # 1a. Capture the absolute first-frame timestamp before any rebase,
    # so the inverse (`absolute = master_t0_ns + relative`) stays exact.
    if N == 0:
        master_t0_ns_local = 0
    else:
        master_t0_ns_local = int(master.master_ns[0])

    # 2. Per-topic deltas. Driven by the catalog — every topic in
    # `TIMESTAMP_TOPICS` becomes a `topic_deltas[name]` entry (length
    # N, with NULL_DELTA at rows whose source stamp was missing).
    #
    # Tolerance: image_right uses `tol_image_right_ns`; image_side uses
    # `tol_image_side_ns`; every kinematic topic uses
    # `tol_kinematic_ns`. The policy table predates the catalog, so
    # this mapping is hardcoded here — if the tolerance scheme ever
    # gets richer (e.g. per-topic tolerance), surface it via the
    # catalog directly.
    def _tol_for(topic_name: str) -> int:
        if topic_name == "image_right":
            return int(policy.tol_image_right_ns)
        if topic_name == "image_side":
            return int(policy.tol_image_side_ns)
        # All other topics in the catalog are kinematic.
        return int(policy.tol_kinematic_ns)

    topic_signed_deltas: dict[str, np.ndarray] = {}
    out_of_tol_counts: dict[str, int] = {}
    for topic in TIMESTAMP_TOPICS:
        cand_ns = ts.topic_stamps.get(topic.name)
        if cand_ns is None:
            # Defensive: catalog declares a topic but the ingester
            # didn't produce its column. Materialize an all-NULL array
            # so the rest of the pipeline still works.
            cand_ns = np.full(N, NULL_TS, dtype=np.int64)
        is_null = cand_ns == NULL_TS
        raw_delta = cand_ns.astype(np.int64) - master.master_ns
        tol_ns = _tol_for(topic.name)
        oot = (np.abs(raw_delta) > tol_ns) & ~is_null
        topic_signed_deltas[topic.name] = _delta_preserve_with_null_for_missing(
            raw_delta, cand_ns,
        )
        out_of_tol_counts[topic.name] = int(oot.sum())

    # 3. Build per-arm aligned KinematicSample lists.
    def _sample_by_source(arm: ArmKinematics) -> dict[int, KinematicSample]:
        return {s.frame: s for s in arm.samples}

    ecm_by_src = _sample_by_source(kin_ECM)
    psm1_by_src = _sample_by_source(kin_PSM1)
    psm2_by_src = _sample_by_source(kin_PSM2)

    ECM_aligned: list[Optional[KinematicSample]] = []
    PSM1_aligned: list[Optional[KinematicSample]] = []
    PSM2_aligned: list[Optional[KinematicSample]] = []
    for i in range(N):
        src = int(master.source_frame_indices[i])
        ECM_aligned.append(ecm_by_src.get(src))
        PSM1_aligned.append(psm1_by_src.get(src))
        PSM2_aligned.append(psm2_by_src.get(src))

    # 4. Annotation alignment.
    annotations_aligned: list[Optional[AnnotationSample]] = [
        annotations.samples.get(int(master.source_frame_indices[i])) for i in range(N)
    ]

    # 5. Contiguity / drop detection on the master timeline.
    is_contig, drop_count = detect_contiguity(
        master.master_ns, period_multiplier=policy.contiguity_period_multiplier,
    )

    # 6. Per-topic stats + cross-modal aggregates.
    #
    # Per-topic stats use the present-stamp |delta| only. Topics with
    # n_present == 0 surface None for every metric (n_present stays
    # an explicit zero so consumers can distinguish "no data" from
    # "missing topic in the catalog").
    per_topic_latency: dict[str, dict] = {}
    topic_abs_deltas: dict[str, np.ndarray] = {}   # topic → |delta| of present rows
    for topic_name, signed in topic_signed_deltas.items():
        mask = signed != NULL_DELTA
        present_idx = np.where(mask)[0]
        abs_present = np.abs(signed[mask]).astype(np.int64)
        topic_abs_deltas[topic_name] = abs_present
        if abs_present.size:
            local_argmax = int(np.argmax(abs_present))
            per_topic_latency[topic_name] = {
                "median_delta_ms":     float(np.median(abs_present)) / 1e6,
                "mean_delta_ms":       float(np.mean(abs_present)) / 1e6,
                "std_delta_ms":        float(np.std(abs_present)) / 1e6,  # ddof=0
                "max_delta_ms":        float(np.max(abs_present)) / 1e6,
                "max_delta_frame_idx": int(present_idx[local_argmax]),
                "n_present":           int(abs_present.size),
            }
        else:
            per_topic_latency[topic_name] = {
                "median_delta_ms":     None,
                "mean_delta_ms":       None,
                "std_delta_ms":        None,
                "max_delta_ms":        None,
                "max_delta_frame_idx": None,
                "n_present":           0,
            }

    # Legacy kinematic-only summary — aggregates over every kinematic
    # topic in the catalog (per-arm measured/setpoint × js/cp/cv +
    # local_measured_cp + jaw{,measured/setpoint} for PSM).
    kin_pool = [topic_abs_deltas[name] for name in KINEMATIC_TOPICS
                if topic_abs_deltas.get(name) is not None and topic_abs_deltas[name].size]
    if kin_pool:
        kin_deltas_ns = np.concatenate(kin_pool)
        median_kin_ms: Optional[float] = float(np.median(kin_deltas_ns)) / 1e6
        max_kin_ms: Optional[float]    = float(np.max(kin_deltas_ns)) / 1e6
    else:
        median_kin_ms = None
        max_kin_ms    = None

    # Cross-modal — pool over EVERY topic in the catalog.
    cross_pool = [a for a in topic_abs_deltas.values() if a.size]
    if cross_pool:
        cross_modal_ns = np.concatenate(cross_pool)
        cross_modal_median_ms: Optional[float] = float(np.median(cross_modal_ns)) / 1e6
        cross_modal_mean_ms: Optional[float]   = float(np.mean(cross_modal_ns)) / 1e6
        cross_modal_std_ms: Optional[float]    = float(np.std(cross_modal_ns)) / 1e6
        cross_modal_max_ms: Optional[float]    = float(np.max(cross_modal_ns)) / 1e6
    else:
        cross_modal_median_ms = None
        cross_modal_mean_ms   = None
        cross_modal_std_ms    = None
        cross_modal_max_ms    = None

    # Episode length in seconds.
    if N >= 1:
        episode_length_s = float(master.master_ns[-1] - master.master_ns[0]) / 1e9
    else:
        episode_length_s = 0.0
    frames_dropped = int(drop_count.sum())
    contiguity_ratio = float(is_contig[1:].mean()) if N > 1 else 1.0

    stats = SyncStats(
        episode_length_s=episode_length_s,
        median_kin_delta_ms=median_kin_ms,
        max_kin_delta_ms=max_kin_ms,
        cross_modal_median_delta_ms=cross_modal_median_ms,
        cross_modal_mean_delta_ms=cross_modal_mean_ms,
        cross_modal_std_delta_ms=cross_modal_std_ms,
        cross_modal_max_delta_ms=cross_modal_max_ms,
        frames_dropped=frames_dropped,
        contiguity_ratio=contiguity_ratio,
        out_of_tol_counts=out_of_tol_counts,
    )

    # 7. Rebase master_ns so the first row is 0 ns. Absolute time of
    # any row is recoverable via `master_t0_ns + master_ns[i]`.
    master_ns_rebased = master.master_ns.astype(np.int64) - np.int64(master_t0_ns_local)

    return AlignedClip(
        n_frames=N,
        master_ns=master_ns_rebased,
        master_t0_ns=master_t0_ns_local,
        source_frame_index=master.source_frame_indices.astype(np.int32),
        is_contiguous_to_prev=is_contig,
        drop_count_since_prev=drop_count,
        topic_deltas=topic_signed_deltas,
        ECM_aligned=ECM_aligned,
        PSM1_aligned=PSM1_aligned,
        PSM2_aligned=PSM2_aligned,
        annotations=annotations_aligned,
        sync_stats=stats,
        per_topic_latency=per_topic_latency,
    )
