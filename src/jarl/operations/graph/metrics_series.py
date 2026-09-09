"""Read downsampled metric series for sub-agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.metrics import build_metrics_snapshot, validate_requested_metric_keys
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.contracts.constants import METRICS_SERIES_DEFAULT_KEYS, METRICS_SERIES_MAX_POINTS
from jarl.operations.contracts.sampling import downsample_indices
from jarl.training.config import RLRunConfig

__all__ = ["MetricPoint", "MetricsSeriesRequest", "MetricsSeriesResponse", "metrics_series"]


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """One downsampled metric observation."""

    step: int
    value: float


@dataclass(frozen=True, slots=True)
class MetricsSeriesRequest:
    """Inputs for reading metric series."""

    node_id: str
    metric_keys: list[str] | None = None
    max_points: int = METRICS_SERIES_MAX_POINTS


@dataclass(frozen=True, slots=True)
class MetricsSeriesResponse:
    """Outcome of ``metrics_series``."""

    node_id: str
    series: dict[str, list[MetricPoint]]

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def metrics_series(
    graph: ExperimentGraph[RLRunConfig],
    request: MetricsSeriesRequest,
) -> MetricsSeriesResponse:
    """Return downsampled metric series for the requested keys."""
    workspace = graph.get_node(request.node_id)
    keys = request.metric_keys or sorted(METRICS_SERIES_DEFAULT_KEYS)
    if request.metric_keys is not None:
        validate_requested_metric_keys(
            build_metrics_snapshot(workspace.metrics_jsonl_path).names,
            request.metric_keys,
        )
    reader_path = workspace.metrics_jsonl_path
    if not reader_path.is_file():
        return MetricsSeriesResponse(node_id=request.node_id, series={key: [] for key in keys})

    reader = JsonlMetricReader(reader_path)
    series: dict[str, list[MetricPoint]] = {}
    for key in keys:
        points = reader.series(key)
        indices = downsample_indices(len(points), request.max_points)
        series[key] = [MetricPoint(step=points[index][0], value=points[index][1]) for index in indices]
    return MetricsSeriesResponse(node_id=request.node_id, series=series)
