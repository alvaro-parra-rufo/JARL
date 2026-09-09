"""Single-pass metric snapshot loading."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from jarl.experiments.io.metrics import JsonlMetricReader, MetricRecord

__all__ = [
    "MetricsSnapshot",
    "available_metric_names",
    "build_metrics_snapshot",
    "load_metric_records",
]


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Single-pass summary of a node ``metrics.jsonl`` file."""

    latest: dict[str, float]
    names: tuple[str, ...]
    record_count: int
    by_step: dict[int, dict[str, float]]

    @classmethod
    def empty(cls) -> MetricsSnapshot:
        """Return an empty snapshot for missing metric files."""
        return cls(latest={}, names=(), record_count=0, by_step={})

    @property
    def step_count(self) -> int:
        """Number of distinct training steps in the snapshot."""
        return len(self.by_step)


def build_metrics_snapshot(metrics_path: Path) -> MetricsSnapshot:
    """Load latest values and step-indexed metrics in a single file pass."""
    if not metrics_path.is_file():
        return MetricsSnapshot.empty()

    best: dict[str, tuple[int, float]] = {}
    names: set[str] = set()
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    record_count = 0

    for record in JsonlMetricReader(metrics_path).iter_records():
        record_count += 1
        names.add(record.name)
        by_step[record.step][record.name] = record.value
        current = best.get(record.name)
        if current is None or record.step > current[0] or record.step == current[0]:
            best[record.name] = (record.step, record.value)

    latest = {name: value for name, (_step, value) in best.items()}
    return MetricsSnapshot(
        latest=latest,
        names=tuple(sorted(names)),
        record_count=record_count,
        by_step=dict(by_step),
    )


def load_metric_records(metrics_path: Path) -> list[MetricRecord]:
    """Load all metric records from a node ``metrics.jsonl`` file."""
    if not metrics_path.is_file():
        return []
    return list(JsonlMetricReader(metrics_path).iter_records())


def available_metric_names(records: list[MetricRecord]) -> list[str]:
    """Return sorted unique metric names."""
    return sorted({record.name for record in records})
