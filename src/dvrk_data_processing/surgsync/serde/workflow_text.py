"""Workflow-annotation verbalization: numeric ID → text description.

Single source of truth: `workflow_description/workflow_description.json`,
a hand-curated, supplied-as-input file. The runtime reads it once at
import and serves every `verbalize_phase` / `verbalize_step` /
`verbalize_gesture` lookup out of it. There is no code in this module
that *generates* the JSON — it is treated as a fixed input.

Expected JSON layout:

    {
      "schema_version": ...,
      "phases": {
        "<phase_id>": {
          "description":  "<phase text>",
          "step":         {<step_id>: <step_text>, ...},
          "gesture":      {<gesture_id>: <gesture_text>, ...},
        },
        ...
      },
      "_task_routing": {<task_name>: <phase_id>, ...},
    }

Each task partition (`single_interrupted_stitch`, `cold_cut_dissection*`,
`peg_transfer`, `tissue_manipulation`, ...) maps to one phase id via
`_task_routing`. `verbalize_step` / `verbalize_gesture` use that mapping
to pick the right per-phase vocab; unknown ids fall through unchanged.

Tables load at import time and are read-only at runtime.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


def _workflow_root() -> Path:
    # This file is src/dvrk_data_processing/surgsync/serde/workflow_text.py
    # repo root is parents[4].
    return Path(__file__).resolve().parents[4] / "workflow_description"


_CONSOLIDATED_PATH = _workflow_root() / "workflow_description.json"


def _load_consolidated() -> dict:
    """Read the consolidated workflow JSON. Failures degrade gracefully —
    a missing or malformed file should not crash unrelated imports;
    instead, every lookup will pass IDs through unchanged."""
    try:
        return json.loads(_CONSOLIDATED_PATH.read_text())
    except FileNotFoundError:
        log.warning(
            "workflow_description.json not found at %s — "
            "verbalize_* lookups will pass IDs through unchanged.",
            _CONSOLIDATED_PATH,
        )
        return {"phases": {}, "_task_routing": {}}
    except json.JSONDecodeError as e:
        log.error(
            "workflow_description.json at %s is malformed (%s) — verbalize_* lookups will "
            "pass IDs through. Fix the file.",
            _CONSOLIDATED_PATH, e,
        )
        return {"phases": {}, "_task_routing": {}}


# Public read-only view of the consolidated JSON. Useful for callers that
# want to introspect the doc directly (validators, test fixtures).
WORKFLOW_DOC: dict = _load_consolidated()


def _phase_block(phase_id: Optional[str]) -> dict:
    """Return the `{description, step, gesture}` block for a phase id,
    or an empty dict when the id is unknown / None."""
    if phase_id is None:
        return {}
    return WORKFLOW_DOC.get("phases", {}).get(phase_id, {})


# Task partition name → phase id. Populated from the JSON's
# `_task_routing` block so adding a new task partition only requires
# editing the JSON.
TASK_ROUTING: dict[str, str] = dict(WORKFLOW_DOC.get("_task_routing", {}))


# Inverse routing — phase id → canonical task name. Used by the packer
# to auto-pick a task folder when no explicit override is supplied.
# Multiple tasks may route to the same phase (e.g. cold_cut_dissection*
# variants all sit under one phase). We honor TASK_ROUTING's insertion
# order from the JSON: first task wins per phase, which lets the JSON
# author pick the canonical task name by listing it first.
PHASE_TO_TASK: dict[str, str] = {}
for _task, _phase in TASK_ROUTING.items():
    PHASE_TO_TASK.setdefault(_phase, _task)


def phase_to_task(phase_id: Optional[str]) -> Optional[str]:
    """Phase id → canonical task name (inverse of `_task_routing`).

    Returns `None` for unmapped phases (e.g. phase "0" has no
    associated task in the routing table). Callers should fall back
    to an explicit override or raise — surgsync never invents a task
    name out of thin air.
    """
    if phase_id is None:
        return None
    return PHASE_TO_TASK.get(str(phase_id))


# ---------------------------------------------------------------------------
# Backward-compatible module-level dicts.
#
# Older callers (`validate/dataset.py`, tests) import these by name to
# build expected-text sets and to route tasks. They're now thin views
# over the consolidated JSON.
# ---------------------------------------------------------------------------

PHASE_DESCRIPTIONS: dict[str, str] = {
    pid: ph["description"]
    for pid, ph in WORKFLOW_DOC.get("phases", {}).items()
    if "description" in ph
}

# `STEP_DESCRIPTIONS`: the union of every phase's step dict, used by the
# validator (I-6) to assemble the set of permissible step texts. When the
# same step id appears under multiple phases with identical text, the
# union collapses cleanly; if texts ever diverge across phases, the last
# write wins — surface that in code review when curating the JSON.
STEP_DESCRIPTIONS: dict[str, str] = {}
for _pid, _ph in WORKFLOW_DOC.get("phases", {}).items():
    for _sid, _text in _ph.get("step", {}).items():
        STEP_DESCRIPTIONS[_sid] = _text

# Suturing phase 1 has its own step + gesture vocab; the dissection
# vocab lives under phase 2 in the consolidated JSON. Phases 5 / 6 are
# semantically dissection but their JSON gesture blocks are currently
# empty (vocab deferred).
SUTURING_STEP_DESCRIPTIONS: dict[str, str] = dict(_phase_block("1").get("step", {}))
SUTURING_GESTURE_DESCRIPTIONS: dict[str, str] = dict(_phase_block("1").get("gesture", {}))
DISSECTION_GESTURE_DESCRIPTIONS: dict[str, str] = dict(_phase_block("2").get("gesture", {}))

# Phase id ↔ task family. `_task_routing` in the JSON is authoritative,
# so derive these from it rather than from hardcoded constants. Anything
# routed to phase 1 is suturing; anything routed to phase 2 / 5 / 6 is
# dissection. Adding a new task partition to the JSON automatically
# extends these sets.
PHASES_SUTURING   = frozenset({"1"})
PHASES_DISSECTION = frozenset({"2", "5", "6"})

SUTURING_TASKS = frozenset(
    {task for task, pid in TASK_ROUTING.items() if pid in PHASES_SUTURING}
)
DISSECTION_TASKS = frozenset(
    {task for task, pid in TASK_ROUTING.items() if pid in PHASES_DISSECTION}
)


# ---------------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------------

def verbalize_phase(value: Optional[str]) -> Optional[str]:
    """Phase id → text. Task-agnostic — looked up against the JSON's
    `phases.<id>.description`. Unknown ids pass through unchanged."""
    if value is None:
        return None
    return PHASE_DESCRIPTIONS.get(value, value)


def verbalize_step(value: Optional[str], task: Optional[str] = None) -> Optional[str]:
    """Step id → text.

    Resolution order:
    1. If `task` is given and its phase (via `_task_routing`) has the id
       in its `step` dict, return that phase's step text.
    2. Otherwise scan every phase's step dict; return the first hit.
       Phase iteration order follows the JSON (insertion order).
    3. Unknown id → pass through unchanged.

    The fallback scan exists so callers that don't know the task (or
    pass `task=None`) still get a useful description for globally
    meaningful ids like `"0"` / `"10"`.
    """
    if value is None:
        return None
    pid = TASK_ROUTING.get(task) if task else None
    if pid is not None:
        steps = _phase_block(pid).get("step", {})
        if value in steps:
            return steps[value]
    # Fall back: walk every phase.
    for ph in WORKFLOW_DOC.get("phases", {}).values():
        if value in ph.get("step", {}):
            return ph["step"][value]
    return value


def task_vocab_rows() -> list[dict]:
    """Project the workflow JSON into one row per task for `meta/tasks.jsonl`.

    Iterates `_task_routing`; for each `(task → phase_id)` pair, looks
    up the phase block (`phases.<phase_id>.{description, step, gesture}`)
    and emits a row with the canonical task name, its routed phase id,
    the phase's description, and verbatim copies of the phase's step +
    gesture vocabs. Each row is fully self-contained — consumers can
    read the JSONL without joining back to the workflow JSON.

    Empty `step` / `gesture` vocabs are surfaced as empty dicts (e.g.
    peg_transfer has no gesture vocab). Tasks whose `_task_routing`
    phase id has no entry under `phases.<phase_id>` are skipped with
    a warning — that would only happen on a malformed workflow JSON.
    """
    out: list[dict] = []
    phases = WORKFLOW_DOC.get("phases", {})
    for task, phase_id in TASK_ROUTING.items():
        phase = phases.get(str(phase_id))
        if not phase:
            log.warning(
                "task_vocab_rows: task %r routes to phase %r which is not "
                "in workflow_description.phases — skipping",
                task, phase_id,
            )
            continue
        out.append({
            "task": task,
            "phase_id": str(phase_id),
            "phase_description": phase.get("description", ""),
            "step_vocab":    dict(phase.get("step", {})),
            "gesture_vocab": dict(phase.get("gesture", {})),
        })
    return out


def verbalize_gesture(value: Optional[str], task: Optional[str]) -> Optional[str]:
    """Gesture id → text, task-aware.

    The task name routes to a phase via `_task_routing`; that phase's
    `gesture` dict is the only lookup. Tasks whose phase has no gesture
    vocab (peg_transfer → phase 3, tissue_manipulation → phase 4,
    cold-cut variants whose phases 5/6 currently expose no gestures)
    pass IDs through unchanged — better than emitting NULL, since the
    raw id is still meaningful to a downstream consumer.
    """
    if value is None:
        return None
    pid = TASK_ROUTING.get(task) if task else None
    if pid is None:
        return value
    gestures = _phase_block(pid).get("gesture", {})
    return gestures.get(value, value)
