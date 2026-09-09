"""Tests for checkpoint and model archive semantics on NodeWorkspace."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CHECKPOINT_ALIAS_FINAL, CHECKPOINT_ALIAS_LATEST, CheckpointRef
from jarl.experiments.io.model_archive import MODEL_ALIAS_LATEST, canonical_model_filename, load_model_archive
from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace


class SampleConfig(BaseConfig):
    """Concrete config for checkpoint and model semantics tests."""

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


class TestModelArchiveImmutability:
    """Tests for canonical model archives and alias pointers."""

    def test_multiple_steps_keep_distinct_files(self, workspace: NodeWorkspace) -> None:
        policy_a = {"weights": [1.0]}
        policy_b = {"weights": [2.0]}
        critic = {"weights": [0.0]}

        path_a = workspace.save_model_archive("weights", 5, policy=policy_a, critic=critic, alias=MODEL_ALIAS_LATEST)
        path_b = workspace.save_model_archive("weights", 10, policy=policy_b, critic=critic, alias=MODEL_ALIAS_LATEST)

        assert path_a.name == canonical_model_filename("weights", 5)
        assert path_b.name == canonical_model_filename("weights", 10)
        assert path_a.exists()
        assert path_b.exists()
        assert workspace.get_artifact("model", "weights", 5) is not None
        assert workspace.get_artifact("model", "weights", 10) is not None
        latest = workspace.resolve_model_alias(MODEL_ALIAS_LATEST)
        assert latest is not None
        assert latest.name == "weights"
        assert latest.step == 10

    def test_unsafe_model_archive_rejected(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "evil.model"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "../manifest.json",
                json.dumps({"format_version": 1, "components": ["policy"], "name": "x", "step": 1}),
            )

        with pytest.raises(ValueError, match="Unsafe path"):
            load_model_archive(archive_path)


class TestCheckpointSemantics:
    """Tests for save/load, final alias, and validation behavior."""

    def test_save_checkpoint_saves_without_explicit_force(self, workspace: NodeWorkspace) -> None:
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=10_000)

        saved = workspace.save_checkpoint(3, {"value": 3})

        assert saved
        assert workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST) is not None

    def test_failed_run_keeps_latest_without_final(self, workspace: NodeWorkspace) -> None:
        with pytest.raises(RuntimeError, match="boom"), workspace:
            workspace.save_checkpoint(2, {"value": 2})
            raise RuntimeError("boom")

        assert workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST) is not None
        assert workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_FINAL) is None
        assert workspace.status == NodeStatus.FAILED

    def test_promote_missing_checkpoint_raises(self, workspace: NodeWorkspace) -> None:
        with pytest.raises(KeyError, match="No checkpoint record"):
            workspace.promote_checkpoint_best(99, metric_name="loss", metric_value=0.1, reason="missing")

    def test_pinned_checkpoint_survives_max_to_keep(self, tmp_path: Path) -> None:
        ws = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_baseline_ab12cd34", branch="main"),
        )
        ws.init_checkpoint_manager(max_to_keep=1, save_interval_steps=1)
        ws.save_checkpoint(1, {"value": 1})
        ws.pin_checkpoint(1)
        ws.save_checkpoint(2, {"value": 2})
        ws.save_checkpoint(3, {"value": 3})

        restored = ws.load_checkpoint(1)

        assert restored == {"value": 1}

    def test_load_checkpoint_lazy_inits_from_disk(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=SampleConfig(), branch="main", label="baseline")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=100)
        root.save_checkpoint(4, {"value": 4})
        graph.save()

        restored_graph = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)
        reloaded = restored_graph.head("main")

        assert reloaded.load_checkpoint() == {"value": 4}


class TestForkValidation:
    """Tests for strict CheckpointRef validation before graph mutation."""

    def test_fork_rejects_missing_checkpoint_ref(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=SampleConfig(), branch="main", label="baseline")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        root.save_checkpoint(1, {"value": 1})

        with pytest.raises(ValueError, match="No checkpoint record registered for step 99"):
            graph.fork(
                "alt",
                from_node=root,
                from_checkpoint=CheckpointRef(node_id=root.id, checkpoint_step=99),
            )

        assert "alt" not in graph._branch_heads
