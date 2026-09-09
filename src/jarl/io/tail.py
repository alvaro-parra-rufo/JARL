"""Incremental file tail readers without UI session state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jarl.experiments.io.metrics import JsonlMetricReader

__all__ = [
    "merge_lines_into_buffer",
    "parse_metrics_records",
    "read_jsonl_line_delta",
    "read_log_bytes_delta",
    "read_metrics_latest",
]


def read_log_bytes_delta(path: Path, *, byte_offset: int) -> tuple[list[str], int]:
    """Read new log lines from ``path`` starting at ``byte_offset``.

    Returns:
        Tuple of new lines and the next byte offset. If the file shrank, reading
        restarts from the beginning.
    """
    if not path.is_file():
        return [], byte_offset

    data = path.read_bytes()
    if len(data) < byte_offset:
        byte_offset = 0

    chunk = data[byte_offset:]
    next_offset = len(data)
    if not chunk:
        return [], next_offset

    text = chunk.decode("utf-8", errors="replace")
    return text.splitlines(), next_offset


def read_jsonl_line_delta(path: Path, *, line_offset: int) -> tuple[list[str], int]:
    """Read new raw JSONL lines from ``path`` starting at ``line_offset``."""
    if not path.is_file():
        return [], line_offset

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if line_offset > len(lines):
        line_offset = 0

    new_lines = lines[line_offset:]
    return new_lines, len(lines)


def read_metrics_latest(metrics_path: Path) -> dict[str, float]:
    """Return the latest scalar metrics from a node ``metrics.jsonl`` file."""
    if not metrics_path.is_file():
        return {}
    return JsonlMetricReader(metrics_path).latest()


def parse_metrics_records(lines: list[str]) -> list[dict[str, Any]]:
    """Parse metric JSONL lines, skipping blanks."""
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def merge_lines_into_buffer(existing: list[str], new_lines: list[str], *, max_lines: int) -> list[str]:
    """Append log lines and cap the buffer size."""
    if not new_lines:
        return existing
    buffer = [*existing, *new_lines]
    if len(buffer) > max_lines:
        return buffer[-max_lines:]
    return buffer
