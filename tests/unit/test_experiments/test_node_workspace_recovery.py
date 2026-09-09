"""Tests for node recovery, deferred preparation, and execution attempts."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.execution_attempts import ExecutionAttemptStatus
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import (
    NodeMetadata,
    NodeStatus,
    NodeWorkspace,
    validate_prepare_transition,
    validate_training_entry,
)
from jarl.experiments.run_config import TrackingConfig


class SampleConfig(BaseConfig):
    """Minimal config for recovery lifecycle tests."""

    learning_rate: float = 0.001


class TestLifecycleTransitions:
    """Parametrized lifecycle transition guards."""

    @pytest.mark.parametrize(
        ("status", "should_fail"),
        [
            pytest.param(NodeStatus.CREATED, False, id="created"),
            pytest.param(NodeStatus.PREPARED, False, id="prepared"),
            pytest.param(NodeStatus.FAILED, False, id="failed"),
            pytest.param(NodeStatus.INTERRUPTED, False, id="interrupted"),
            pytest.param(NodeStatus.COMPLETED, True, id="completed"),
            pytest.param(NodeStatus.TRAINING, True, id="training"),
        ],
    )
    def test_validate_training_entry(self, status: NodeStatus, should_fail: bool) -> None:
        if should_fail:
            with pytest.raises(RuntimeError):
                validate_training_entry(status)
        else:
            validate_training_entry(status)

    @pytest.mark.parametrize(
        ("status", "should_fail"),
        [
            pytest.param(NodeStatus.CREATED, False, id="created"),
            pytest.param(NodeStatus.PREPARED, True, id="prepared"),
            pytest.param(NodeStatus.FAILED, True, id="failed"),
        ],
    )
    def test_validate_prepare_transition(self, status: NodeStatus, should_fail: bool) -> None:
        if should_fail:
            with pytest.raises(RuntimeError):
                validate_prepare_transition(status)
        else:
            validate_prepare_transition(status)


class TestExecutionAttempts:
    """Tests for execution attempt registration across training runs."""

    def test_failed_run_records_attempt_and_resume_links(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_run_ab12cd34", branch="main"),
            tracking=TrackingConfig(track_tensorboard=False, track_wandb=False),
        )
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)

        with pytest.raises(RuntimeError, match="boom"), workspace:
            workspace.save_checkpoint(1, {"value": 1})
            raise RuntimeError("boom")

        assert workspace.status == NodeStatus.FAILED
        attempts = workspace.list_execution_attempts()
        assert len(attempts) == 1
        assert attempts[0].status == ExecutionAttemptStatus.FAILED
        assert attempts[0].checkpoint_step_at_end == 1
        assert attempts[0].error_type == "RuntimeError"

        with workspace:
            restored = workspace.load_resume_checkpoint()
            assert restored == {"value": 1}

        attempts = workspace.list_execution_attempts()
        assert len(attempts) == 2
        assert attempts[1].status == ExecutionAttemptStatus.COMPLETED
        assert attempts[1].resume_of_attempt_id == attempts[0].attempt_id
        assert attempts[1].checkpoint_step_at_start == 1

    def test_keyboard_interrupt_marks_interrupted(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_run_ab12cd34", branch="main"),
        )

        with pytest.raises(KeyboardInterrupt), workspace:
            raise KeyboardInterrupt

        assert workspace.status == NodeStatus.INTERRUPTED
        attempt = workspace.latest_execution_attempt()
        assert attempt is not None
        assert attempt.status == ExecutionAttemptStatus.INTERRUPTED


class TestDeferredPreparation:
    """Tests for prepare/start without implicit training."""

    def test_prepare_root_does_not_write_run_metrics(self, tmp_path: Path) -> None:
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "exp")
        workspace = graph.create_root(
            config=SampleConfig(),
            branch="main",
            label="baseline",
            prepare=True,
        )

        assert workspace.status == NodeStatus.PREPARED
        assert workspace.metadata_path.exists()
        assert workspace.config_path.exists()
        assert not workspace.metrics_jsonl_path.exists()
        assert not workspace.log_path.exists()

    def test_prepared_node_starts_training_explicitly(self, tmp_path: Path) -> None:
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "exp")
        workspace = graph.create_root(
            config=SampleConfig(),
            branch="main",
            prepare=True,
        )

        with workspace:
            workspace.log_scalar(1, "loss", 0.5)

        assert workspace.status == NodeStatus.COMPLETED
        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": pytest.approx(0.5)}

    def test_prepare_extend_round_trip(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        graph.create_root(config=SampleConfig(), branch="main", label="root")
        child = graph.extend("main", label="child", prepare=True)
        graph.save()

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)
        reloaded = restored.get_node(child.id)

        assert reloaded.status == NodeStatus.PREPARED


class TestResumeWithoutNewNode:
    """Tests for resuming the same node after failure."""

    def test_failed_node_resumes_from_latest_checkpoint(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_run_ab12cd34", branch="main"),
        )
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)

        with pytest.raises(ValueError, match="boom"), workspace:
            workspace.save_checkpoint(3, {"step": 3})
            raise ValueError("boom")

        assert workspace.status == NodeStatus.FAILED
        restored = workspace.load_resume_checkpoint()

        assert restored == {"step": 3}

    def test_resolve_resume_skips_unrestorable_latest_checkpoint(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_run_ab12cd34", branch="main"),
        )
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(1, {"step": 1}, force=True)
        workspace.save_checkpoint(2, {"step": 2}, force=True)

        import shutil

        shutil.rmtree(workspace.checkpoint_dir / "2")

        record = workspace.resolve_resume_checkpoint_record()

        assert record is not None
        assert record.checkpoint_step == 1
        assert workspace.load_resume_checkpoint() == {"step": 1}

    def test_failed_node_without_checkpoint_retries_without_resume_link(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(
                id="main_child_ab12cd34",
                branch="main",
                parent_id="main_parent_ab12cd34",
                parent_checkpoint_step=5,
            ),
        )

        with pytest.raises(RuntimeError, match="boom"), workspace:
            raise RuntimeError("boom")

        assert workspace.status == NodeStatus.FAILED

        with workspace:
            pass

        attempts = workspace.list_execution_attempts()
        assert len(attempts) == 2
        assert attempts[1].resume_of_attempt_id is None
        assert attempts[1].checkpoint_step_at_start is None

    def test_resume_requires_restorable_checkpoint_when_any_exist(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_run_ab12cd34", branch="main"),
        )
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)

        with pytest.raises(RuntimeError, match="boom"), workspace:
            workspace.save_checkpoint(1, {"step": 1})
            raise RuntimeError("boom")

        import shutil

        shutil.rmtree(workspace.checkpoint_dir)

        with pytest.raises(RuntimeError, match="No restorable checkpoint available"), workspace:
            pass

    def test_completed_node_still_requires_extend(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=SampleConfig(), branch="main", label="root")

        with root:
            pass

        with pytest.raises(RuntimeError, match="Cannot re-enter a completed node"), root:
            pass

        child = graph.extend("main", label="child")
        assert child.node_metadata.parent_id == root.id
