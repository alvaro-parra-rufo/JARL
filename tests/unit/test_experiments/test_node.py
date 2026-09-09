"""Tests for NodeMetadata and NodeWorkspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace
from jarl.experiments.run_config import TrackingConfig
from jarl.metadata import now_iso

# ---- Fixtures ---- #


@pytest.fixture()
def sample_metadata() -> NodeMetadata:
    """A minimal node metadata instance."""
    return NodeMetadata(
        id="main_baseline_a1b2c3d4",
        parent_id=None,
        branch="main",
        label="baseline",
        description="Root node",
        created_at=now_iso(),
        updated_at=now_iso(),
    )


@pytest.fixture()
def workspace(tmp_path: Path, sample_metadata: NodeMetadata) -> NodeWorkspace:
    """A fresh node workspace in a temp directory."""
    return NodeWorkspace(
        node_dir=tmp_path / "nodes" / sample_metadata.id,
        node_metadata=sample_metadata,
    )


# ---- NodeMetadata serialization ---- #


class TestNodeMetadata:
    """Tests for metadata save/load round-trip."""

    def test_save_creates_file(self, tmp_path: Path, sample_metadata: NodeMetadata) -> None:
        path = sample_metadata.save(tmp_path / "node.json")

        assert path.exists()

    def test_round_trip_preserves_fields(self, tmp_path: Path, sample_metadata: NodeMetadata) -> None:
        sample_metadata.save(tmp_path / "node.json")

        loaded = NodeMetadata.load(tmp_path / "node.json")

        assert loaded.id == sample_metadata.id
        assert loaded.branch == "main"
        assert loaded.label == "baseline"
        assert loaded.parent_id is None
        assert loaded.description == "Root node"

    def test_round_trip_preserves_custom_metadata(self, tmp_path: Path) -> None:
        meta = NodeMetadata(
            id="test_node",
            metadata={"reason": "testing higher lr", "author": "llm"},
        )
        meta.save(tmp_path / "node.json")

        loaded = NodeMetadata.load(tmp_path / "node.json")

        assert loaded.metadata == {"reason": "testing higher lr", "author": "llm"}


# ---- NodeWorkspace directory structure ---- #


class TestNodeWorkspaceDirectories:
    """Tests for workspace path materialization."""

    def test_base_dir_created(self, workspace: NodeWorkspace) -> None:
        assert workspace.path.exists()

    def test_checkpoint_path_has_no_side_effects(self, workspace: NodeWorkspace) -> None:
        assert not workspace.checkpoint_dir.exists()

        assert workspace.checkpoint_dir == workspace.path / "checkpoint"

    def test_init_checkpoint_manager_materializes_checkpoint_dir(
        self,
        workspace: NodeWorkspace,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch("jarl.experiments.node.ocp.CheckpointManager")

        workspace.init_checkpoint_manager()

        assert workspace.checkpoint_dir.is_dir()

    def test_tensorboard_path_has_no_side_effects(self, workspace: NodeWorkspace) -> None:
        assert not workspace.tensorboard_dir.exists()
        assert workspace.tensorboard_dir == workspace.path / "tensorboard"

    def test_tb_writer_materializes_tensorboard_dir(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "nodes" / "main_test_abc12345"
        workspace = NodeWorkspace(
            node_dir=node_dir,
            node_metadata=NodeMetadata(id="main_test_abc12345"),
            tracking=TrackingConfig(track_tensorboard=True),
        )

        with workspace:
            _ = workspace.tb_writer

        assert workspace.tensorboard_dir.is_dir()

    def test_log_path(self, workspace: NodeWorkspace) -> None:
        assert workspace.log_path == workspace.path / "train.log"
        assert not workspace.log_path.exists()


# ---- NodeWorkspace context manager ---- #


class TestNodeWorkspaceContextManager:
    """Tests for __enter__ / __exit__ lifecycle."""

    def test_enter_sets_training_status(self, workspace: NodeWorkspace) -> None:
        with workspace as ws:
            assert ws.status == NodeStatus.TRAINING

    def test_exit_sets_completed(self, workspace: NodeWorkspace) -> None:
        with workspace:
            pass

        assert workspace.status == NodeStatus.COMPLETED

    def test_exit_sets_failed_on_exception(self, workspace: NodeWorkspace) -> None:
        with pytest.raises(ValueError, match="boom"), workspace:
            raise ValueError("boom")

        assert workspace.status == NodeStatus.FAILED

    def test_exit_sets_interrupted_on_keyboard_interrupt(self, workspace: NodeWorkspace) -> None:
        with pytest.raises(KeyboardInterrupt), workspace:
            raise KeyboardInterrupt

        assert workspace.status == NodeStatus.INTERRUPTED

    def test_metadata_persisted_on_exit(self, workspace: NodeWorkspace) -> None:
        with workspace:
            pass

        assert workspace.metadata_path.exists()
        loaded = NodeMetadata.load(workspace.metadata_path)
        assert loaded.status == NodeStatus.COMPLETED

    def test_completed_node_reentry_raises(self, workspace: NodeWorkspace) -> None:
        with workspace:
            pass

        assert workspace.status == NodeStatus.COMPLETED
        with pytest.raises(RuntimeError, match="Cannot re-enter a completed node"), workspace:
            pass

    def test_failed_node_reentry_allowed(self, workspace: NodeWorkspace) -> None:
        with pytest.raises(ValueError, match="boom"), workspace:
            raise ValueError("boom")

        assert workspace.status == NodeStatus.FAILED
        with workspace as ws:
            assert ws.status == NodeStatus.TRAINING

    def test_log_handler_attached_and_detached(self, workspace: NodeWorkspace) -> None:
        import logging

        logger = logging.getLogger("jarl")
        handlers_before = len(logger.handlers)

        with workspace:
            assert len(logger.handlers) == handlers_before + 1

        assert len(logger.handlers) == handlers_before


# ---- NodeWorkspace identity ---- #


class TestNodeWorkspaceIdentity:
    """Tests for identity properties."""

    def test_id(self, workspace: NodeWorkspace) -> None:
        assert workspace.id == "main_baseline_a1b2c3d4"

    def test_branch(self, workspace: NodeWorkspace) -> None:
        assert workspace.branch == "main"

    def test_repr(self, workspace: NodeWorkspace) -> None:
        result = repr(workspace)

        assert "main_baseline_a1b2c3d4" in result
        assert "main" in result


# ---- NodeWorkspace metrics ---- #


class TestNodeWorkspaceMetrics:
    """Tests for per-node metrics persistence."""

    def test_save_metrics_writes_jsonl(self, workspace: NodeWorkspace) -> None:
        path = workspace.save_metrics({"loss": 0.5, "accuracy": 0.9})

        assert path == workspace.metrics_jsonl_path
        assert path.exists()

    def test_save_metrics_appends_multiple_steps(self, workspace: NodeWorkspace) -> None:
        workspace.node_metadata.step = 1
        workspace.save_metrics({"loss": 0.5})
        workspace.node_metadata.step = 2
        workspace.save_metrics({"loss": 0.3})

        from jarl.experiments.io import JsonlMetricReader

        assert JsonlMetricReader(workspace.metrics_jsonl_path).series("loss") == [(1, 0.5), (2, 0.3)]


# ---- NodeWorkspace checkpoint paths ---- #


class TestNodeWorkspaceCheckpointPaths:
    """Tests for Orbax checkpoint directory resolution."""

    def test_init_checkpoint_manager_uses_absolute_path(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        sample_metadata: NodeMetadata,
    ) -> None:
        import os

        relative_root = Path("rel_exp")
        (tmp_path / relative_root / "nodes" / sample_metadata.id).mkdir(parents=True)
        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            workspace = NodeWorkspace(
                node_dir=relative_root / "nodes" / sample_metadata.id,
                node_metadata=sample_metadata,
            )
            mock_manager = mocker.patch("jarl.experiments.node.ocp.CheckpointManager")

            workspace.init_checkpoint_manager()

            directory = mock_manager.call_args.kwargs["directory"]
            assert directory.is_absolute()
            assert not workspace.checkpoint_dir.is_absolute()
            assert directory == workspace.checkpoint_dir.resolve()
        finally:
            os.chdir(original_cwd)

    def test_load_parent_checkpoint_uses_absolute_path(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        sample_metadata: NodeMetadata,
    ) -> None:
        import os

        relative_root = Path("rel_exp")
        parent_ckpt = relative_root / "nodes" / "parent" / "checkpoint"
        (tmp_path / parent_ckpt).mkdir(parents=True)
        (tmp_path / relative_root / "nodes" / sample_metadata.id).mkdir(parents=True)
        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            workspace = NodeWorkspace(
                node_dir=relative_root / "nodes" / sample_metadata.id,
                node_metadata=sample_metadata,
                parent_checkpoint_path=parent_ckpt,
            )
            mock_manager = mocker.patch("jarl.experiments.node.ocp.CheckpointManager")
            mock_manager.return_value.latest_step.return_value = 0
            mock_manager.return_value.restore.return_value = {"state": 1}

            workspace.load_parent_checkpoint()

            directory = mock_manager.call_args.kwargs["directory"]
            assert directory.is_absolute()
            assert directory == parent_ckpt.resolve()
        finally:
            os.chdir(original_cwd)


# ---- NodeWorkspace config overrides ---- #


class TestNodeWorkspaceConfigOverrides:
    """Tests for config override persistence."""

    def test_save_and_load_overrides(self, workspace: NodeWorkspace) -> None:
        workspace.save_config_overrides({"learning_rate": 0.01})

        loaded = workspace.load_config_overrides()

        assert loaded == {"learning_rate": 0.01}

    def test_load_empty_when_no_file(self, workspace: NodeWorkspace) -> None:
        result = workspace.load_config_overrides()

        assert result == {}
