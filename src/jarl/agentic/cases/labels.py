"""Spanish labels for agentic case outcomes in UI and progress logs."""

from __future__ import annotations

from typing import Literal

__all__ = [
    "agentic_case_exit_code_label",
    "agentic_case_status_label",
]

AgenticCaseStatusLabel = Literal["passed", "failed", "error"]

_STATUS_LABELS: dict[AgenticCaseStatusLabel, str] = {
    "passed": "Pasó",
    "failed": "Falló",
    "error": "Error",
}

_EXIT_CODE_LABELS: dict[int, str] = {
    0: "Pasó",
    1: "Falló",
    2: "Salida inválida",
    3: "Error",
}


def agentic_case_status_label(status: AgenticCaseStatusLabel) -> str:
    """Return a Spanish label for one persisted case status."""
    return _STATUS_LABELS[status]


def agentic_case_exit_code_label(exit_code: int) -> str:
    """Return a Spanish label for one agentic-case CLI or subprocess exit code."""
    return _EXIT_CODE_LABELS.get(exit_code, f"Código {exit_code}")
