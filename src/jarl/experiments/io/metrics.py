"""Incremental JSONL metrics IO for node workspaces."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

__all__ = [
    "JsonlMetricReader",
    "JsonlMetricWriter",
    "MetricRecord",
]


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Single scalar metric entry persisted to JSONL.

    Args:
        step: Training step associated with the measurement.
        name: Metric identifier.
        value: Scalar metric value.
    """

    step: int
    name: str
    value: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert the record to a JSON-serializable dictionary."""
        return {"step": self.step, "name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MetricRecord:
        """Build a record from a parsed JSON object.

        Args:
            data: Parsed JSON object from one JSONL line.

        Returns:
            Parsed `MetricRecord`.

        Raises:
            KeyError: If required fields are missing.
            TypeError: If field types are invalid.
        """
        return cls(
            step=int(data["step"]),  # type: ignore[arg-type]
            name=str(data["name"]),
            value=float(data["value"]),  # type: ignore[arg-type]
        )


class JsonlMetricWriter:
    """Append scalar metrics to a JSONL file.

    Args:
        path: Destination JSONL file path.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the writer."""
        self._path = Path(path)
        self._file: TextIO | None = None

    @property
    def path(self) -> Path:
        """Destination JSONL file path."""
        return self._path

    def open(self) -> None:
        """Open the JSONL file for append and materialize parent directories."""
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")

    def write(self, record: MetricRecord) -> None:
        """Append one metric record to the JSONL file.

        Args:
            record: Metric entry to persist.
        """
        self.open()
        if self._file is None:
            msg = "Metric writer failed to open output file."
            raise RuntimeError(msg)
        self._file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        self.flush()

    def log_scalar(self, step: int, name: str, value: float) -> None:
        """Append one scalar metric at the given step.

        Args:
            step: Training step associated with the measurement.
            name: Metric identifier.
            value: Scalar metric value.
        """
        self.write(MetricRecord(step=step, name=name, value=value))

    def flush(self) -> None:
        """Flush buffered writes to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlMetricWriter:
        """Open the writer as a context manager."""
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the writer when leaving the context."""
        self.close()


class JsonlMetricReader:
    """Read scalar metrics from a JSONL file.

    Args:
        path: Source JSONL file path.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the reader."""
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Source JSONL file path."""
        return self._path

    def iter_records(self) -> Iterator[MetricRecord]:
        """Yield metric records in file order.

        Yields:
            Parsed `MetricRecord` entries.

        Raises:
            FileNotFoundError: If the JSONL file does not exist.
        """
        if not self._path.exists():
            msg = f"Metrics file not found: {self._path}"
            raise FileNotFoundError(msg)
        with self._path.open(encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                yield MetricRecord.from_dict(json.loads(stripped))

    def read_records(self) -> list[MetricRecord]:
        """Load all metric records into memory.

        Returns:
            Parsed records in file order.
        """
        return list(self.iter_records())

    def latest(self) -> dict[str, float]:
        """Return the most recent value recorded for each metric name.

        When a metric appears at multiple steps, the value at the highest
        step wins. Ties within the same step keep the last written value.

        Returns:
            Mapping from metric name to latest scalar value.
        """
        best: dict[str, tuple[int, float]] = {}
        if not self._path.exists():
            return {}
        for record in self.iter_records():
            current = best.get(record.name)
            if current is None or record.step > current[0] or record.step == current[0]:
                best[record.name] = (record.step, record.value)
        return {name: value for name, (_step, value) in best.items()}

    def at_step(self, step: int) -> dict[str, float]:
        """Return all metric values recorded at a specific step.

        Args:
            step: Training step to query.

        Returns:
            Mapping from metric name to value at that step.
        """
        result: dict[str, float] = {}
        if not self._path.exists():
            return result
        for record in self.iter_records():
            if record.step == step:
                result[record.name] = record.value
        return result

    def series(self, name: str) -> list[tuple[int, float]]:
        """Return the time series for one metric name.

        Args:
            name: Metric identifier.

        Returns:
            `(step, value)` pairs sorted by step.
        """
        points = [(record.step, record.value) for record in self.iter_records() if record.name == name]
        return sorted(points, key=lambda item: item[0])
