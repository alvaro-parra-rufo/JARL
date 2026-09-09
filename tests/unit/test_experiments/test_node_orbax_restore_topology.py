"""Tests for NodeWorkspace Orbax checkpoint restore behavior."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from pytest_mock import MockerFixture

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.node import NodeMetadata, NodeWorkspace
from tests.helpers.checkpoint_restore import standard_restore_args_from_call


class SampleConfig(BaseConfig):
    """Minimal config for checkpoint restore flow tests."""

    learning_rate: float = 0.1


class TestNodeWorkspaceCheckpointRoundTrip:
    """Save and restore checkpoints through the public NodeWorkspace API."""

    def test_load_checkpoint_restores_saved_payload(self, tmp_path: Path) -> None:
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_baseline_ab12cd34", branch="main"),
        )
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        payload = {"weights": jnp.array([1.0, 2.0], dtype=jnp.float32)}

        workspace.save_checkpoint(1, payload, force=True)
        restored = workspace.load_checkpoint(1)

        assert restored == {"weights": pytest.approx(jnp.array([1.0, 2.0], dtype=jnp.float32))}

    def test_child_load_checkpoint_restores_parent_fork_step(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=SampleConfig(), branch="main", label="baseline")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        root.save_checkpoint(1, {"value": 1}, force=True)
        root.save_checkpoint(2, {"value": 2}, force=True)

        child = graph.fork(
            "alt",
            from_node=root,
            from_checkpoint=CheckpointRef(node_id=root.id, checkpoint_step=1),
        )

        restored = child.load_checkpoint()

        assert child.node_metadata.parent_checkpoint_step == 1
        assert restored == {"value": 1}


class TestNodeWorkspaceOrbaxRestoreArgs:
    """NodeWorkspace passes fallback sharding into Orbax restore."""

    @pytest.fixture()
    def workspace(self, tmp_path: Path) -> NodeWorkspace:
        return NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_child_ab12cd34", branch="main"),
            parent_checkpoint_path=tmp_path / "parent" / "checkpoint",
        )

    def test_load_parent_checkpoint_restore_uses_fallback_sharding(
        self,
        mocker: MockerFixture,
        workspace: NodeWorkspace,
    ) -> None:
        parent_dir = workspace._parent_ckpt_path
        assert parent_dir is not None
        parent_dir.mkdir(parents=True)
        mock_manager = mocker.patch("jarl.experiments.node.ocp.CheckpointManager")
        mock_manager.return_value.latest_step.return_value = 1
        mock_manager.return_value.restore.return_value = {"value": 1}

        workspace.load_parent_checkpoint(1)

        restore_args = standard_restore_args_from_call(mock_manager.return_value.restore.call_args)
        assert isinstance(restore_args, ocp.args.StandardRestore)
        assert restore_args.fallback_sharding == jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def test_load_checkpoint_restore_uses_fallback_sharding(
        self,
        mocker: MockerFixture,
        workspace: NodeWorkspace,
    ) -> None:
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(1, {"value": 1}, force=True)
        manager = workspace._ensure_local_checkpoint_manager()
        assert manager is not None
        spy_restore = mocker.spy(manager, "restore")

        workspace.load_checkpoint(1)

        restore_args = standard_restore_args_from_call(spy_restore.call_args)
        assert isinstance(restore_args, ocp.args.StandardRestore)
        assert restore_args.fallback_sharding == jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def test_validate_saved_checkpoint_restore_uses_fallback_sharding(
        self,
        mocker: MockerFixture,
        workspace: NodeWorkspace,
    ) -> None:
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(1, {"value": 1}, force=True)
        manager = workspace._ensure_local_checkpoint_manager()
        assert manager is not None
        spy_restore = mocker.spy(manager, "restore")

        workspace.validate_saved_checkpoint(1)

        restore_args = standard_restore_args_from_call(spy_restore.call_args)
        assert isinstance(restore_args, ocp.args.StandardRestore)
        assert restore_args.fallback_sharding == jax.sharding.SingleDeviceSharding(jax.devices()[0])


class TestParentCheckpointRestoreIntegration:
    """Parent checkpoints restore through ``load_parent_checkpoint`` without local steps."""

    def test_load_parent_checkpoint_restores_saved_parent_payload(
        self,
        tmp_path: Path,
    ) -> None:
        parent_dir = tmp_path / "parent"
        child_dir = tmp_path / "child"
        payload = {"weights": jnp.array([1.0, 2.0], dtype=jnp.float32)}

        parent_ws = NodeWorkspace(
            node_dir=parent_dir,
            node_metadata=NodeMetadata(id="main_parent_ab12cd34", branch="main"),
        )
        parent_ws.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        parent_ws.save_checkpoint(1, payload, force=True)

        child_ws = NodeWorkspace(
            node_dir=child_dir,
            node_metadata=NodeMetadata(
                id="main_child_ab12cd34",
                branch="main",
                parent_id=parent_ws.id,
                parent_checkpoint_step=1,
            ),
            parent_checkpoint_path=parent_ws.checkpoint_dir,
        )

        restored = child_ws.load_parent_checkpoint(1)

        assert restored == {"weights": pytest.approx(jnp.array([1.0, 2.0], dtype=jnp.float32))}
