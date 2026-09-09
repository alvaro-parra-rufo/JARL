"""Prepare objective metric features for qualitative subagent analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.graph.metrics_series import MetricsSeriesRequest, metrics_series
from jarl.operations.subagent.metrics_preprocess import MetricsFeaturePayload, build_metrics_features
from jarl.training.config import RLRunConfig

__all__ = [
    "MetricsAnalysisRequest",
    "MetricsAnalysisResponse",
    "metrics_analysis",
]


@dataclass(frozen=True, slots=True)
class MetricsAnalysisRequest:
    """Inputs for building the metrics-analysis feature payload."""

    node_id: str
    metric_keys: list[str] | None = None
    focus: str = ""


@dataclass(frozen=True, slots=True)
class MetricsAnalysisResponse:
    """Objective feature payload consumed by the metrics-analysis subagent."""

    node_id: str
    features: dict[str, float | int]
    feature_ids: tuple[str, ...]
    focus: str = ""

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)

    def to_feature_payload(self) -> MetricsFeaturePayload:
        """Return the typed feature payload used by validators."""
        return MetricsFeaturePayload(
            node_id=self.node_id,
            features=dict(self.features),
            feature_ids=self.feature_ids,
            focus=self.focus,
        )


def metrics_analysis(
    graph: ExperimentGraph[RLRunConfig],
    request: MetricsAnalysisRequest,
) -> MetricsAnalysisResponse:
    """Load metric series and build objective features for LLM interpretation."""
    series_response = metrics_series(
        graph,
        MetricsSeriesRequest(node_id=request.node_id, metric_keys=request.metric_keys),
    )
    series_points: dict[str, list[tuple[int, float]]] = {
        key: [(point.step, point.value) for point in points] for key, points in series_response.series.items()
    }
    payload = build_metrics_features(
        request.node_id,
        series_points,
        focus=request.focus,
    )
    return MetricsAnalysisResponse(
        node_id=payload.node_id,
        features=payload.features,
        feature_ids=payload.feature_ids,
        focus=payload.focus,
    )
