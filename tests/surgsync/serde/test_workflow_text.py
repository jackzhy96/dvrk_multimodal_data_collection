"""Workflow text-description lookups.

These tests pin the runtime behavior of the verbalize helpers in
`serde/workflow_text.py`. The single source of truth is the
hand-curated `workflow_description/workflow_description.json`; nothing
here re-derives that file from raw source documents.
"""
from __future__ import annotations

import json
from pathlib import Path

from dvrk_data_processing.surgsync.serde.workflow_text import (
    PHASE_DESCRIPTIONS,
    STEP_DESCRIPTIONS,
    SUTURING_GESTURE_DESCRIPTIONS,
    DISSECTION_GESTURE_DESCRIPTIONS,
    SUTURING_STEP_DESCRIPTIONS,
    SUTURING_TASKS,
    DISSECTION_TASKS,
    TASK_ROUTING,
    WORKFLOW_DOC,
    verbalize_phase,
    verbalize_step,
    verbalize_gesture,
)


def _spec() -> dict:
    """Load the committed JSON fresh — handy for tests that want to
    assert the runtime view matches what's on disk right now."""
    repo_root = Path(__file__).resolve().parents[3]
    return json.loads(
        (repo_root / "workflow_description" / "workflow_description.json").read_text()
    )


def test_workflow_doc_matches_disk():
    """WORKFLOW_DOC is the parsed consolidated JSON, byte-equivalent
    in structure to whatever is committed under `workflow_description/`.
    Catches accidental in-memory mutation of the doc."""
    assert WORKFLOW_DOC == _spec()


def test_phase_table_loads_known_ids():
    # The committed JSON carries phase ids 0..6.
    assert "0" in PHASE_DESCRIPTIONS
    assert "1" in PHASE_DESCRIPTIONS
    assert (
        "tissue" in PHASE_DESCRIPTIONS["0"].lower()
        or "manipulate" in PHASE_DESCRIPTIONS["0"].lower()
    )


def test_step_table_covers_suturing_and_tissue_manipulation():
    # Suturing high-level steps 11-15 live under phase 1; the
    # tissue-manipulation block 41-45 lives under phase 0 and phase 4.
    # The union surfaced as STEP_DESCRIPTIONS covers them all.
    for k in ("0", "11", "12", "13", "14", "15", "41", "42", "43", "44", "45"):
        assert k in STEP_DESCRIPTIONS, f"missing step id {k!r}"


def test_suturing_step_vocab_is_phase_1():
    """SUTURING_STEP_DESCRIPTIONS is exactly phase 1's step block."""
    assert SUTURING_STEP_DESCRIPTIONS == _spec()["phases"]["1"]["step"]


def test_suturing_gestures_loaded():
    # Phase 1 in the JSON carries 18 entries (1-18).
    assert len(SUTURING_GESTURE_DESCRIPTIONS) >= 17
    for gid in ("1", "2", "5", "13", "17"):
        assert gid in SUTURING_GESTURE_DESCRIPTIONS
    assert "Reach" in SUTURING_GESTURE_DESCRIPTIONS["1"]


def test_dissection_gestures_loaded():
    # Phase 2 carries the dissection vocab (PSM1-style 10..13, PSM2-style
    # 20..25, and the shared "0").
    for gid in ("0", "10", "11", "12", "13", "20", "21", "22", "23", "24", "25"):
        assert gid in DISSECTION_GESTURE_DESCRIPTIONS


def test_task_routing_derives_task_sets():
    """SUTURING_TASKS / DISSECTION_TASKS are derived from the JSON's
    `_task_routing` block; editing the JSON changes them automatically."""
    spec = _spec()
    routing = spec.get("_task_routing", {})

    # Whatever tasks the JSON routes to phase 1 should land in
    # SUTURING_TASKS; phases 2/5/6 into DISSECTION_TASKS.
    expected_suturing   = {t for t, pid in routing.items() if pid == "1"}
    expected_dissection = {t for t, pid in routing.items() if pid in {"2", "5", "6"}}
    assert SUTURING_TASKS == frozenset(expected_suturing)
    assert DISSECTION_TASKS == frozenset(expected_dissection)

    # Every entry under TASK_ROUTING points at a phase that exists in
    # the JSON — otherwise verbalize_* would silently pass through.
    for task, pid in TASK_ROUTING.items():
        assert pid in PHASE_DESCRIPTIONS, f"task {task!r} routes to unknown phase {pid!r}"


def test_verbalize_pass_through_for_unknown():
    """Unknown ids pass through unchanged so the column stays populated."""
    assert verbalize_phase("9999") == "9999"
    assert verbalize_step("9999") == "9999"
    assert verbalize_gesture("9999", "single_interrupted_stitch") == "9999"


def test_verbalize_none_stays_none():
    assert verbalize_phase(None) is None
    assert verbalize_step(None) is None
    assert verbalize_gesture(None, "single_interrupted_stitch") is None


def test_verbalize_gesture_is_task_aware():
    """The same gesture id resolves differently across tasks because
    `_task_routing` sends suturing to phase 1 and cold-cut dissection
    to phase 2."""
    suturing = verbalize_gesture("10", "single_interrupted_stitch")
    dissection = verbalize_gesture("10", "cold_cut_dissection")
    assert suturing != dissection
    assert "suture" in suturing.lower()
    assert "scissor" in dissection.lower() or "tissue" in dissection.lower()


def test_verbalize_gesture_unknown_task_passes_through():
    """Tasks whose routed phase has no gesture vocab (peg_transfer →
    phase 3) and unknown tasks both pass IDs through."""
    assert verbalize_gesture("1", "peg_transfer") == "1"
    assert verbalize_gesture("1", None) == "1"


def test_verbalize_phase_resolves_known_id():
    text = verbalize_phase("1")
    assert "suture" in text.lower() or "stitch" in text.lower()


def test_verbalize_step_routes_via_task():
    """`verbalize_step` looks up under the task's routed phase first."""
    # Suturing step "12" → phase 1 → "Needle Driving (...)".
    text = verbalize_step("12", task="single_interrupted_stitch")
    assert "Needle Driving" in text
    # Dissection step "23" → phase 2 → "cold_cut: perform cold-cut...".
    text = verbalize_step("23", task="cold_cut_dissection")
    assert "cold_cut" in text.lower()


def test_verbalize_step_fallback_when_task_none():
    """With no task, fall back to the first phase that carries the id."""
    text = verbalize_step("12")
    assert "needle" in text.lower()


def test_runtime_matches_json_for_every_phase():
    """Comprehensive sanity check: every phase/step/gesture entry in the
    committed JSON should be reachable via the matching verbalize_*
    call using the right task. Guards against future drift between the
    JSON layout and the lookup logic."""
    spec = _spec()
    routing = spec.get("_task_routing", {})
    # Pick a representative task for each phase id.
    phase_to_task: dict[str, str] = {}
    for task, pid in routing.items():
        phase_to_task.setdefault(pid, task)

    for pid, ph in spec["phases"].items():
        assert verbalize_phase(pid) == ph["description"], f"phase {pid}"
        task = phase_to_task.get(pid)
        for sid, expected in ph.get("step", {}).items():
            got = verbalize_step(sid, task=task)
            assert got == expected, f"phase {pid} step {sid}: {got!r} != {expected!r}"
        for gid, expected in ph.get("gesture", {}).items():
            if task is None:
                continue  # phase has no routed task → can't task-route
            got = verbalize_gesture(gid, task=task)
            assert got == expected, f"phase {pid} gesture {gid}: {got!r} != {expected!r}"
