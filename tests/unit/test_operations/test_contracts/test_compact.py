"""Tests for operation compact serialization contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.contracts.compact import compact_value, dataclass_to_compact_dict
from jarl.operations.graph.metrics_series import MetricPoint, MetricsSeriesResponse
from jarl.operations.graph.summary import SummaryRequest, summary
from jarl.training.config import RLRunConfig


class _Color(Enum):
    RED = "red"


@dataclass(frozen=True, slots=True)
class _Nested:
    step: int
    value: float


@dataclass(frozen=True, slots=True)
class _Sample:
    name: str
    path: Path
    nested: _Nested
    color: _Color
    optional: str | None = None


class TestCompactValue:
    def test_serializes_path_enum_and_nested_dataclass(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        payload = compact_value(
            _Sample(
                name="node",
                path=exp_dir,
                nested=_Nested(step=1, value=2.5),
                color=_Color.RED,
            )
        )

        assert payload == {
            "name": "node",
            "path": exp_dir.as_posix(),
            "nested": {"step": 1, "value": 2.5},
            "color": "red",
        }

    def test_dataclass_to_compact_dict_omits_none_fields(self, tmp_path: Path) -> None:
        compact = dataclass_to_compact_dict(
            _Sample(
                name="node",
                path=tmp_path / "exp",
                nested=_Nested(step=0, value=0.0),
                color=_Color.RED,
                optional=None,
            )
        )

        assert "optional" not in compact


class TestResponseCompactDict:
    def test_metrics_series_to_compact_dict_is_json_serializable(self) -> None:
        response = MetricsSeriesResponse(
            node_id="main_baseline_ab12cd34",
            series={
                "rollout/episode_return": [
                    MetricPoint(step=0, value=1.0),
                    MetricPoint(step=1, value=2.0),
                ]
            },
        )

        payload = response.to_compact_dict()

        json.dumps(payload)
        assert payload["series"]["rollout/episode_return"] == [
            {"step": 0, "value": 1.0},
            {"step": 1, "value": 2.0},
        ]

    def test_summary_to_compact_dict_is_json_serializable(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        payload = summary(prepared_root, SummaryRequest()).to_compact_dict()

        json.dumps(payload)
        assert isinstance(payload["experiment"]["exp_dir"], str)
