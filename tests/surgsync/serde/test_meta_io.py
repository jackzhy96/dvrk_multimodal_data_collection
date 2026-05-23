from __future__ import annotations
import json
from pathlib import Path

import pytest

from dvrk_data_processing.surgsync.serde.meta_io import (
    load_clip_meta,
    clip_meta_to_dict,
)


REPO = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(
    not (REPO / "data" / "online_data" / "2" / "meta_data.json").exists(),
    reason="sample data not present",
)
def test_load_online_sample_meta():
    cm = load_clip_meta(REPO / "data" / "online_data" / "2" / "meta_data.json")
    assert cm.operator_skill_level == "Intermediate"
    assert cm.case_type == "Ex-vivo"
    assert cm.tool["PSM1"] == "Large_Needle_Driver"
    assert cm.tool["PSM2"] == "Maryland_Bipolar_Forceps"
    assert cm.failure == []
    assert cm.recovery == []
    # No site-specific extra keys in the sample fixture.
    assert cm.extra == {}


def test_round_trip_preserves_extra_keys(tmp_path: Path):
    """Site-specific extension keys outside the known set survive a
    forward+inverse cycle bit-for-bit."""
    payload = {
        "user_id": "5",
        "operator_skill_level": "Novice",
        "case_type": "Phantom",
        "tool": {"PSM1": "Cadiere", "PSM2": "ProGrasp"},
        "failure": [[10, 20]],
        "recovery": [],
        "custom.site_id": "stanford-lab-04",  # extension key
        "custom.note": "operator hand-off mid-clip",
    }
    src = tmp_path / "meta_data.json"
    with open(src, "w") as f:
        json.dump(payload, f, sort_keys=True)

    cm = load_clip_meta(src)
    assert "custom.site_id" in cm.extra
    assert cm.extra["custom.site_id"] == "stanford-lab-04"

    out = clip_meta_to_dict(cm)
    # Sort keys to compare; the inverse may reorder.
    assert json.dumps(out, sort_keys=True) == json.dumps(payload, sort_keys=True)
