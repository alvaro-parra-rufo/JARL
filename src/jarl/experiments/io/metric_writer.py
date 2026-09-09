"""Unified local metric writer for node workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jarl.experiments.io.metrics import JsonlMetricWriter, MetricRecord

if TYPE_CHECKING:
    from flax.metrics.tensorboard import SummaryWriter

__all__ = [
    "NodeMetricWriter",
]


class NodeMetricWriter:
    """Persist scalar metrics to JSONL and optional TensorBoard.

    Mirrors the practical behavior of ``LocalMetricWriter`` from the reference
    benchmark while using the Flax TensorBoard stack already present in ``jarl``.

    Args:
        jsonl_path: Destination JSONL metrics file.
        tensorboard_dir: Directory for TensorBoard event files.
        enable_tensorboard: Whether to write TensorBoard events.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tensorboard_dir: str | Path,
        *,
        enable_tensorboard: bool = False,
    ) -> None:
        """Initialize the writer."""
        self._jsonl_path = Path(jsonl_path)
        self._tensorboard_dir = Path(tensorboard_dir)
        self._enable_tensorboard = enable_tensorboard
        self._jsonl_writer = JsonlMetricWriter(self._jsonl_path)
        self._summary_writer: SummaryWriter | None = None

    @property
    def jsonl_path(self) -> Path:
        """Destination JSONL metrics file."""
        return self._jsonl_path

    @property
    def tensorboard_dir(self) -> Path:
        """Directory for TensorBoard event files."""
        return self._tensorboard_dir

    @property
    def summary_writer(self) -> SummaryWriter | None:
        """Flax TensorBoard writer when tracking is enabled."""
        return self._summary_writer

    @property
    def enable_tensorboard(self) -> bool:
        """Whether TensorBoard event logging is active."""
        return self._enable_tensorboard and self._summary_writer is not None

    def open(self) -> None:
        """Open JSONL output and optionally initialize TensorBoard."""
        self._jsonl_writer.open()
        if self._enable_tensorboard and self._summary_writer is None:
            from flax.metrics.tensorboard import SummaryWriter

            self._tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self._summary_writer = SummaryWriter(log_dir=str(self._tensorboard_dir))

    def write(self, record: MetricRecord) -> None:
        """Persist one metric record to all enabled outputs.

        Args:
            record: Metric entry to write.
        """
        self._jsonl_writer.write(record)
        if self._summary_writer is not None:
            self._summary_writer.scalar(record.name, record.value, record.step)

    def log_scalar(self, step: int, name: str, value: float) -> None:
        """Write one scalar metric to JSONL and optional TensorBoard.

        Args:
            step: Training step associated with the measurement.
            name: Metric identifier.
            value: Scalar metric value.
        """
        self.write(MetricRecord(step=step, name=name, value=value))

    def flush(self) -> None:
        """Flush buffered metric outputs."""
        self._jsonl_writer.flush()
        if self._summary_writer is not None:
            self._summary_writer.flush()

    def close(self) -> None:
        """Close JSONL and TensorBoard outputs."""
        self._jsonl_writer.close()
        if self._summary_writer is not None:
            self._summary_writer.flush()
            self._summary_writer.close()
            self._summary_writer = None

    def __enter__(self) -> NodeMetricWriter:
        """Open the writer as a context manager."""
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the writer when leaving the context."""
        self.close()
