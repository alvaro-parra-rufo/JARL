"""Tests for metric snapshot loading."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.io.metrics import JsonlMetricReader, JsonlMetricWriter
from jarl.experiments.metrics import MetricsSnapshot, build_metrics_snapshot


def _write_sample_metrics(path: Path) -> None:
    with JsonlMetricWriter(path) as writer:
        writer.log_scalar(1, "rollout/episode_return", 0.1)
        writer.log_scalar(1, "loss/policy_gradient_loss", 0.5)
        writer.log_scalar(2, "rollout/episode_return", 0.2)
        writer.log_scalar(2, "eval/episode_return", 0.9)


class TestBuildMetricsSnapshot:
    """Single-pass metrics loading."""

    def test_empty_when_file_missing(self, tmp_path: Path) -> None:
        snapshot = build_metrics_snapshot(tmp_path / "missing.jsonl")

        assert snapshot.record_count == 0
        assert snapshot.latest == {}
        assert snapshot.names == ()

    def test_matches_reader_latest_and_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        _write_sample_metrics(path)

        snapshot = build_metrics_snapshot(path)

        assert snapshot.record_count == 4
        assert snapshot.step_count == 2
        assert snapshot.latest == JsonlMetricReader(path).latest()
        assert set(snapshot.names) == {
            "rollout/episode_return",
            "loss/policy_gradient_loss",
            "eval/episode_return",
        }


def build_metrics_snapshot_from_steps(by_step: dict[int, dict[str, float]]) -> MetricsSnapshot:
    latest: dict[str, tuple[int, float]] = {}
    names: set[str] = set()
    record_count = 0
    for step, metrics in by_step.items():
        for name, value in metrics.items():
            record_count += 1
            names.add(name)
            current = latest.get(name)
            if current is None or step >= current[0]:
                latest[name] = (step, value)
    return MetricsSnapshot(
        latest={name: value for name, (_step, value) in latest.items()},
        names=tuple(sorted(names)),
        record_count=record_count,
        by_step=by_step,
    )
