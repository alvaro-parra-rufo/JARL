"""Shared IO utilities for jarl."""

from jarl.io.tail import (
    merge_lines_into_buffer,
    parse_metrics_records,
    read_jsonl_line_delta,
    read_log_bytes_delta,
    read_metrics_latest,
)

__all__ = [
    "merge_lines_into_buffer",
    "parse_metrics_records",
    "read_jsonl_line_delta",
    "read_log_bytes_delta",
    "read_metrics_latest",
]
