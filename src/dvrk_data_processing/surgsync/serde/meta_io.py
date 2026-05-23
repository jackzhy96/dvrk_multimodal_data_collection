"""`meta_data.json` ↔ ClipMeta record, and the lossless mapping into
`episode_meta.json` for the unpack direction.

The raw `meta_data.json` schema (per `specs/raw_data_spec.md`):

    {
      "user_id": "2",
      "operator_skill_level": "Intermediate",
      "case_type": "Ex-vivo",
      "tool": {"PSM1": "...", "PSM2": "..."},
      "failure": [],
      "recovery": []
    }
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ClipMeta:
    """Strict typed image of one raw clip's `meta_data.json`.

    Unrecognized keys are stored on `extra` so the decomposer can
    re-emit them — important for invertibility on clips that carry
    site-specific extensions.
    """
    user_id: Optional[str] = None
    operator_skill_level: Optional[str] = None
    case_type: Optional[str] = None
    tool: dict = field(default_factory=lambda: {"PSM1": None, "PSM2": None})
    failure: list = field(default_factory=list)
    recovery: list = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


_KNOWN_KEYS = {"user_id", "operator_skill_level", "case_type", "tool", "failure", "recovery"}


def load_clip_meta(meta_path: Path) -> ClipMeta:
    """Parse `meta_data.json` into ClipMeta."""
    with open(meta_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"meta_data.json at {meta_path} must be a JSON object")

    extras = {k: v for k, v in payload.items() if k not in _KNOWN_KEYS}
    tool = payload.get("tool") or {}
    return ClipMeta(
        user_id=payload.get("user_id"),
        operator_skill_level=payload.get("operator_skill_level"),
        case_type=payload.get("case_type"),
        tool={"PSM1": tool.get("PSM1"), "PSM2": tool.get("PSM2")},
        failure=list(payload.get("failure", [])),
        recovery=list(payload.get("recovery", [])),
        extra=extras,
    )


def clip_meta_to_dict(cm: ClipMeta) -> dict:
    """Inverse of `load_clip_meta` — produce a dict for json.dump.

    Re-emits the `extra` keys so a forward+inverse round-trip preserves
    site-specific extensions verbatim.
    """
    out: dict = {
        "user_id": cm.user_id,
        "operator_skill_level": cm.operator_skill_level,
        "case_type": cm.case_type,
        "tool": cm.tool,
        "failure": cm.failure,
        "recovery": cm.recovery,
    }
    out.update(cm.extra)
    return out
