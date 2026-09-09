"""Tests for JSONL metric records, writers and readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.io.metrics import JsonlMetricReader, JsonlMetricWriter, MetricRecord


class TestMetricRecord:
    """Tests for the MetricRecord contract."""

    @pytest.mark.parametrize(
        ("step", "name", "value"),
        [
            pytest.param(0, "loss", 0.5, id="step_zero"),
            pytest.param(42, "accuracy", 0.91, id="mid_training"),
        ],
    )
    def test_round_trip_dict(self, step: int, name: str, value: float) -> None:
        record = MetricRecord(step=step, name=name, value=value)

        restored = MetricRecord.from_dict(record.to_dict())

        assert restored == record


class TestJsonlMetricWriter:
    """Tests for incremental JSONL metric writes."""

    def test_log_scalar_appends_jsonl_line(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"

        with JsonlMetricWriter(path) as writer:
            writer.log_scalar(1, "loss", 0.5)

        line = path.read_text(encoding="utf-8").strip()
        assert json.loads(line) == {"name": "loss", "step": 1, "value": 0.5}

    def test_write_flushes_before_close(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlMetricWriter(path)

        writer.open()
        writer.log_scalar(1, "loss", 0.5)

        assert JsonlMetricReader(path).latest() == {"loss": 0.5}

        writer.close()

    def test_sequential_writes_track_history(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlMetricWriter(path)

        writer.log_scalar(1, "loss", 0.5)
        writer.log_scalar(2, "loss", 0.3)
        writer.flush()

        reader = JsonlMetricReader(path)

        assert reader.series("loss") == [(1, 0.5), (2, 0.3)]


class TestJsonlMetricReader:
    """Tests for JSONL metric reads and summaries."""

    def test_latest_returns_most_recent_value_per_metric(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"step": 1, "name": "loss", "value": 0.5}),
                    json.dumps({"step": 2, "name": "loss", "value": 0.3}),
                    json.dumps({"step": 2, "name": "accuracy", "value": 0.9}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        latest = JsonlMetricReader(path).latest()

        assert latest == {"loss": 0.3, "accuracy": 0.9}

    def test_at_step_returns_metrics_for_step(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        path.write_text(
            json.dumps({"step": 10, "name": "loss", "value": 0.2}) + "\n",
            encoding="utf-8",
        )

        assert JsonlMetricReader(path).at_step(10) == {"loss": 0.2}
        assert JsonlMetricReader(path).at_step(11) == {}

    def test_latest_on_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert JsonlMetricReader(tmp_path / "missing.jsonl").latest() == {}


class TestNodeWorkspaceJsonlMetrics:
    """Tests for NodeWorkspace JSONL integration."""

    def test_log_scalars_writes_jsonl(self, tmp_path: Path) -> None:
        from jarl.experiments.node import NodeMetadata, NodeWorkspace

        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34", step=5),
        )

        workspace.log_scalars(5, loss=0.4, accuracy=0.8)
        workspace.metric_writer.flush()

        reader = JsonlMetricReader(workspace.metrics_jsonl_path)

        assert reader.at_step(5) == {"loss": 0.4, "accuracy": 0.8}
        assert workspace.latest_metrics() == {"loss": 0.4, "accuracy": 0.8}

    def test_save_metrics_redirects_to_jsonl(self, tmp_path: Path) -> None:
        from jarl.experiments.node import NodeMetadata, NodeWorkspace

        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34", step=3),
        )

        path = workspace.save_metrics({"loss": 0.25})

        assert path == workspace.metrics_jsonl_path
        assert path.exists()
        assert workspace.latest_metrics() == {"loss": 0.25}


class TestGraphMetricsAlongLineage:
    """Tests for lineage metric collection from JSONL."""

    def test_lineage_reads_persisted_jsonl(self, tmp_path: Path) -> None:
        from jarl.config import BaseConfig
        from jarl.experiments.graph import ExperimentGraph

        class SampleConfig(BaseConfig):
            learning_rate: float = 0.1

        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=SampleConfig(), branch="main")
        root.log_scalars(0, loss=0.5)
        root.metric_writer.flush()
        step2 = graph.extend("main")
        step2.log_scalars(1, loss=0.3)
        step2.metric_writer.flush()

        metrics = graph.get_metrics_along_lineage(step2)

        assert metrics == [{"loss": 0.5}, {"loss": 0.3}]
