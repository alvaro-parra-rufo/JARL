"""Tests for NodeWorkspace checkpoint and model archive APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_FINAL,
    CHECKPOINT_ALIAS_LATEST,
    CheckpointRef,
)
from jarl.experiments.io.model_archive import MODEL_ALIAS_LATEST, canonical_model_filename
from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace
from jarl.experiments.run_config import CheckpointConfig


class SampleConfig(BaseConfig):
    """Concrete config for checkpoint tests."""

    learning_rate: float = 0.1


@pytest.fixture()
def workspace(tmp_path: Path) -> NodeWorkspace:
    """Node workspace with checkpoint manager initialized."""
    ws = NodeWorkspace(
        node_dir=tmp_path / "node",
        node_metadata=NodeMetadata(id="main_baseline_ab12cd34", branch="main"),
    )
    ws.init_checkpoint_manager(max_to_keep=None, save_interval_steps=100)
    return ws


class TestNodeWorkspaceCheckpoints:
    """Tests for checkpoint persistence and aliases."""

    def test_save_checkpoint_writes_checkpoints_json(self, workspace: NodeWorkspace) -> None:
        workspace.log_scalar(5, "loss", 0.2)

        workspace.save_checkpoint(5, {"value": 5}, metrics={"loss": 0.2})

        payload = json.loads(workspace.checkpoints_registry_path.read_text(encoding="utf-8"))
        record = payload["checkpoints"][0]

        assert record["node_step"] == 5
        assert record["checkpoint_step"] == 5
        assert record["status"] == "saved"
        assert record["metrics"]["loss"] == pytest.approx(0.2)
        assert payload["aliases"][CHECKPOINT_ALIAS_LATEST] == 5

    def test_save_checkpoint_if_due_respects_step_policy(self, workspace: NodeWorkspace) -> None:
        policy = CheckpointConfig(save_interval_steps=5)

        assert not workspace.save_checkpoint_if_due(4, {"value": 4}, policy=policy)
        assert workspace.save_checkpoint_if_due(5, {"value": 5}, policy=policy)

        latest = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)

        assert latest is not None
        assert latest.checkpoint_step == 5

    def test_save_checkpoint_direct_always_saves(self, workspace: NodeWorkspace) -> None:
        workspace.save_checkpoint(7, {"value": 7}, force=True)

        assert workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST) is not None

    def test_save_training_checkpoint_if_absent_skips_existing_step(self, workspace: NodeWorkspace) -> None:
        workspace.save_checkpoint(10, {"value": 10}, force=True)

        assert workspace.has_saved_checkpoint(10)
        assert not workspace.save_training_checkpoint_if_absent(10, {"value": 11})

        latest = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)

        assert latest is not None
        assert latest.checkpoint_step == 10

    def test_promote_checkpoint_best(self, workspace: NodeWorkspace) -> None:
        workspace.save_checkpoint(3, {"value": 3}, force=True)
        workspace.save_checkpoint(6, {"value": 6}, force=True)

        workspace.promote_checkpoint_best(
            3,
            metric_name="loss",
            metric_value=0.1,
            reason="manual",
        )

        best = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_BEST)

        assert best is not None
        assert best.checkpoint_step == 3

    def test_context_manager_sets_final_alias(self, workspace: NodeWorkspace) -> None:
        with workspace:
            workspace.save_checkpoint(1, {"value": 1}, force=True)

        final = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_FINAL)

        assert final is not None
        assert final.checkpoint_step == 1
        assert workspace.status == NodeStatus.COMPLETED


class TestNodeWorkspaceModelArchives:
    """Tests for model archive registration."""

    def test_save_and_load_model_archive(self, workspace: NodeWorkspace) -> None:
        policy = {"weights": [1.0]}
        critic = {"weights": [2.0]}

        workspace.save_model_archive("weights", 5, policy=policy, critic=critic, alias=MODEL_ALIAS_LATEST)
        restored = workspace.load_model_archive("weights", 5)
        alias_record = workspace.resolve_model_alias(MODEL_ALIAS_LATEST)

        assert restored["policy"] == policy
        assert restored["critic"] == critic
        assert alias_record is not None
        assert alias_record.name == "weights"
        assert alias_record.step == 5
        assert (workspace.models_dir / canonical_model_filename("weights", 5)).exists()


class TestForkFromCheckpoint:
    """Tests for fork/extend from a concrete parent checkpoint."""

    def test_fork_from_checkpoint_loads_parent_step(self, tmp_path: Path) -> None:
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

        pinned = json.loads(root.checkpoints_registry_path.read_text(encoding="utf-8"))["pinned_checkpoint_steps"]

        assert 1 in pinned

    def test_from_directory_restores_parent_checkpoint_step(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=SampleConfig(), branch="main", label="baseline")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        root.save_checkpoint(1, {"value": 1}, force=True)
        graph.fork(
            "alt",
            from_node=root,
            from_checkpoint=CheckpointRef(node_id=root.id, checkpoint_step=1),
        )
        graph.save()

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)
        child = restored.head("alt")

        assert child.node_metadata.parent_checkpoint_step == 1
