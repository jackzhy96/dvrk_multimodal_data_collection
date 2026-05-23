"""Shared types for the validators."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """One finding from a validator.

    `severity` ∈ {"ERROR", "WARNING", "INFO"}. The CLI counts ERRORs
    and exits non-zero if any are present.

    `code` is a short stable identifier (e.g. "raw_missing_subdir",
    "episode_schema_drift", "I-4") so the validator output is greppable.
    """
    severity: str
    code: str
    message: str
