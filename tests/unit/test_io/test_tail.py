"""Tests for ``jarl.io.tail``."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.io.tail import (
    merge_lines_into_buffer,
    parse_metrics_records,
    read_jsonl_line_delta,
    read_log_bytes_delta,
    read_metrics_latest,
)


class TestReadLogBytesDelta:
    """Tests for incremental log reads."""

    def test_returns_new_lines_and_offset(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runner_lab.log"
        log_path.write_text("line1\n", encoding="utf-8")

        lines, offset = read_log_bytes_delta(log_path, byte_offset=0)

        assert lines == ["line1"]
        assert offset == len("line1\n")

    def test_returns_only_appended_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runner_lab.log"
        log_path.write_text("line1\n", encoding="utf-8")
        _, offset = read_log_bytes_delta(log_path, byte_offset=0)
        log_path.write_text("line1\nline2\n", encoding="utf-8")

        lines, next_offset = read_log_bytes_delta(log_path, byte_offset=offset)

        assert lines == ["line2"]
        assert next_offset == len("line1\nline2\n")

    def test_restarts_when_file_shrank(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runner_lab.log"
        log_path.write_text("old content\n", encoding="utf-8")
        log_path.write_text("fresh\n", encoding="utf-8")

        lines, offset = read_log_bytes_delta(log_path, byte_offset=10_000)

        assert lines == ["fresh"]
        assert offset == len("fresh\n")


class TestMergeLinesIntoBuffer:
    """Tests for terminal buffer merging."""

    def test_appends_lines(self) -> None:
        merged = merge_lines_into_buffer(["a"], ["b", "c"], max_lines=10)

        assert merged == ["a", "b", "c"]

    def test_caps_buffer_size(self) -> None:
        merged = merge_lines_into_buffer(["1", "2", "3"], ["4"], max_lines=3)

        assert merged == ["2", "3", "4"]


class TestReadJsonlLineDelta:
    """Tests for metrics JSONL incremental reads."""

    def test_reads_only_new_lines(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text(
            '{"step": 1, "name": "rollout/episode_return", "value": 0.1}\n',
            encoding="utf-8",
        )
        _, line_offset = read_jsonl_line_delta(metrics_path, line_offset=0)
        metrics_path.write_text(
            '{"step": 1, "name": "rollout/episode_return", "value": 0.1}\n'
            '{"step": 2, "name": "eval/episode_return", "value": 0.9}\n',
            encoding="utf-8",
        )

        new_lines, next_offset = read_jsonl_line_delta(metrics_path, line_offset=line_offset)

        assert len(new_lines) == 1
        assert json.loads(new_lines[0])["name"] == "eval/episode_return"
        assert next_offset == 2


class TestParseMetricsRecords:
    """Tests for JSONL metric parsing."""

    def test_skips_blank_lines(self) -> None:
        records = parse_metrics_records(
            [
                "",
                '{"step": 1, "name": "loss/policy_gradient_loss", "value": 0.2}',
            ]
        )

        assert len(records) == 1
        assert records[0]["name"] == "loss/policy_gradient_loss"


class TestReadMetricsLatest:
    """Tests for latest metric lookup."""

    def test_reads_latest_values(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text(
            '{"step": 1, "name": "rollout/episode_return", "value": 0.1}\n'
            '{"step": 2, "name": "rollout/episode_return", "value": 0.5}\n'
            '{"step": 2, "name": "eval/episode_return", "value": 0.8}\n',
            encoding="utf-8",
        )

        latest = read_metrics_latest(metrics_path)

        assert latest["rollout/episode_return"] == 0.5
        assert latest["eval/episode_return"] == 0.8
