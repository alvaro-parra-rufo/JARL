"""Tests for unified NodeMetricWriter with optional TensorBoard."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.io.metric_writer import NodeMetricWriter
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace
from jarl.experiments.run_config import TrackingConfig


class TestNodeMetricWriter:
    """Tests for JSONL and TensorBoard combined writes."""

    def test_disabled_tensorboard_writes_jsonl_only(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "metrics.jsonl"
        tensorboard_dir = tmp_path / "tensorboard"

        with NodeMetricWriter(jsonl_path, tensorboard_dir, enable_tensorboard=False) as writer:
            writer.log_scalar(1, "loss", 0.5)

        assert JsonlMetricReader(jsonl_path).latest() == {"loss": 0.5}
        assert not tensorboard_dir.exists()

    def test_enabled_tensorboard_writes_jsonl_and_events(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "metrics.jsonl"
        tensorboard_dir = tmp_path / "tensorboard"

        with NodeMetricWriter(jsonl_path, tensorboard_dir, enable_tensorboard=True) as writer:
            writer.log_scalar(2, "accuracy", 0.9)

        assert JsonlMetricReader(jsonl_path).latest() == {"accuracy": 0.9}
        assert tensorboard_dir.is_dir()
        assert list(tensorboard_dir.glob("events.out.tfevents.*"))


class TestNodeWorkspaceMetricWriterLifecycle:
    """Tests for run-scoped metric writer integration."""

    @pytest.fixture()
    def workspace(self, tmp_path: Path) -> NodeWorkspace:
        """Workspace with configurable tracking defaults."""
        return NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )

    def test_training_context_closes_run_writer(self, workspace: NodeWorkspace) -> None:
        with workspace:
            workspace.log_scalar(0, "loss", 0.4)

        assert workspace.metrics_jsonl_path.exists()
        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": 0.4}
        assert workspace.status == NodeStatus.COMPLETED

    @pytest.mark.parametrize(
        ("track_tensorboard", "expect_events"),
        [
            pytest.param(False, False, id="tensorboard_off"),
            pytest.param(True, True, id="tensorboard_on"),
        ],
    )
    def test_training_context_respects_tracking_config(
        self,
        tmp_path: Path,
        track_tensorboard: bool,
        expect_events: bool,
    ) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
            tracking=TrackingConfig(track_tensorboard=track_tensorboard),
        )

        with workspace:
            workspace.log_scalar(1, "loss", 0.2)

        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": 0.2}
        has_events = (
            bool(list(workspace.tensorboard_dir.glob("events.out.tfevents.*")))
            if workspace.tensorboard_dir.exists()
            else False
        )
        assert has_events is expect_events

    def test_log_scalar_outside_context_still_writes_jsonl(self, workspace: NodeWorkspace) -> None:
        workspace.log_scalar(3, "loss", 0.1)
        workspace.metric_writer.flush()

        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": 0.1}
        assert not workspace.tensorboard_dir.exists()
