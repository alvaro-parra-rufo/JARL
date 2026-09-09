"""Tests for agentic progress feed."""

from __future__ import annotations

from pathlib import Path

from jarl.agentic.progress import (
    append_progress_event,
    load_progress_events,
    progress_path,
)


def test_append_and_load_progress_events(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    append_progress_event(exp_dir, "case_start", "Iniciando caso demo")
    append_progress_event(exp_dir, "tool_call", "Tool graph_extend (mutation)", data={"tool": "graph_extend"})

    events = load_progress_events(exp_dir)

    assert len(events) == 2
    assert events[0].kind == "case_start"
    assert events[1].data == {"tool": "graph_extend"}
    assert progress_path(exp_dir).is_file()
