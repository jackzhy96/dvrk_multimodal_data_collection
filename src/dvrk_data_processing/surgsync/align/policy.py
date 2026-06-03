"""Tolerance policy.

The matcher needs a tolerance window per (modality, recorder_variant);
this module encodes that table without scattering magic numbers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TolerancePolicy:
    """Tolerance windows in nanoseconds, one per modality.

    Construct via `for_variant(recorder_variant, align_cfg)` so the
    millisecond numbers from the Hydra `align` config become the right
    ns ints in one place.
    """
    tol_image_right_ns: int
    tol_image_side_ns: int
    tol_kinematic_ns: int
    contiguity_period_multiplier: float

    @classmethod
    def for_variant(
        cls,
        recorder_variant: str,
        *,
        tol_ms_image_right: float,
        tol_ms_image_side: float,
        tol_ms_kinematic: float,
        contiguity_period_multiplier: float = 1.5,
    ) -> "TolerancePolicy":
        return cls(
            tol_image_right_ns=int(tol_ms_image_right * 1_000_000),
            tol_image_side_ns=int(tol_ms_image_side * 1_000_000),
            tol_kinematic_ns=int(tol_ms_kinematic * 1_000_000),
            contiguity_period_multiplier=contiguity_period_multiplier,
        )

    @classmethod
    def from_align_cfg(cls, recorder_variant: str, align_cfg) -> "TolerancePolicy":
        """Build a policy from a Hydra `AlignCfg`-shaped object.

        `align_cfg` is duck-typed — we just access the same field names
        as `surgsync.config.AlignCfg`.
        """
        if recorder_variant == "online":
            return cls.for_variant(
                "online",
                tol_ms_image_right=float(align_cfg.tol_ms_image_right_online),
                tol_ms_image_side=float(align_cfg.tol_ms_image_side_online),
                tol_ms_kinematic=float(align_cfg.tol_ms_kinematic_online),
                contiguity_period_multiplier=float(align_cfg.contiguity_period_multiplier),
            )
        if recorder_variant == "offline":
            return cls.for_variant(
                "offline",
                tol_ms_image_right=float(align_cfg.tol_ms_image_right_offline),
                tol_ms_image_side=float(align_cfg.tol_ms_image_side_offline),
                tol_ms_kinematic=float(align_cfg.tol_ms_kinematic_offline),
                contiguity_period_multiplier=float(align_cfg.contiguity_period_multiplier),
            )
        raise ValueError(f"Unknown recorder_variant: {recorder_variant!r}")
