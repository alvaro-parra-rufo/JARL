"""Tests for agentic case UI labels."""

from __future__ import annotations

from jarl.agentic.cases.labels import agentic_case_exit_code_label, agentic_case_status_label


def test_agentic_case_status_labels() -> None:
    assert agentic_case_status_label("passed") == "Pasó"
    assert agentic_case_status_label("failed") == "Falló"
    assert agentic_case_status_label("error") == "Error"


def test_agentic_case_exit_code_labels() -> None:
    assert agentic_case_exit_code_label(0) == "Pasó"
    assert agentic_case_exit_code_label(1) == "Falló"
    assert agentic_case_exit_code_label(2) == "Salida inválida"
    assert agentic_case_exit_code_label(3) == "Error"
