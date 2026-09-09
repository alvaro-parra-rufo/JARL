"""Regression tests for transfer-spine fork/extend graph operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.app.lib.graph_ops import extend_branch, fork_branch
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.training.config import RLRunConfig


def _create_root(exp_dir: Path) -> str:
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    root = graph.create_root(RLRunConfig(), branch="main", label="baseline", prepare=True)
    graph.save()
    return root.id


class TestTransferSpineForkExtend:
    """Showcase transfer spine must fork once, then extend the same branch."""

    def test_second_fork_on_existing_branch_raises(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        root_id = _create_root(exp_dir)
        fork_branch(
            exp_dir,
            from_node=root_id,
            branch="transfer_spine",
            label="transfer_empty6",
            prepare=True,
        )

        with pytest.raises(ValueError, match="already exists"):
            fork_branch(
                exp_dir,
                from_node=root_id,
                branch="transfer_spine",
                label="transfer_empty8",
                prepare=True,
            )

    def test_extend_continues_existing_transfer_spine(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        root_id = _create_root(exp_dir)
        first_id = fork_branch(
            exp_dir,
            from_node=root_id,
            branch="transfer_spine",
            label="transfer_empty6",
            prepare=True,
        )
        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        first_ws = graph.get_node(first_id)
        first_ws.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        first_ws.save_checkpoint(1, {"value": 1}, force=True)
        graph.save()

        second_id = extend_branch(
            exp_dir,
            from_node=first_id,
            branch="transfer_spine",
            label="transfer_empty8",
            from_checkpoint=CheckpointRef(node_id=first_id, checkpoint_step=1),
            config_overrides={"environment.env_id": "Navix-Empty-8x8-v0"},
            prepare=True,
        )

        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        assert graph.head("transfer_spine").id == second_id
        assert graph.get_node(second_id).node_metadata.parent_id == first_id

    def test_extend_rejects_parent_that_is_not_branch_head(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        root_id = _create_root(exp_dir)
        first_id = fork_branch(
            exp_dir,
            from_node=root_id,
            branch="transfer_spine",
            label="transfer_empty6",
            prepare=True,
        )
        second_id = extend_branch(
            exp_dir,
            from_node=first_id,
            branch="transfer_spine",
            label="transfer_empty8",
            prepare=True,
        )

        with pytest.raises(ValueError, match="not the head"):
            extend_branch(
                exp_dir,
                from_node=first_id,
                branch="transfer_spine",
                label="transfer_empty8_retry",
                prepare=True,
            )

        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        assert graph.head("transfer_spine").id == second_id

    def test_extend_rejects_non_head_when_branch_implicit(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        root_id = _create_root(exp_dir)
        first_id = fork_branch(
            exp_dir,
            from_node=root_id,
            branch="transfer_spine",
            label="transfer_empty6",
            prepare=True,
        )
        second_id = extend_branch(
            exp_dir,
            from_node=first_id,
            branch="transfer_spine",
            label="transfer_empty8",
            prepare=True,
        )

        with pytest.raises(ValueError, match="not the head"):
            extend_branch(
                exp_dir,
                from_node=first_id,
                label="transfer_empty8_retry",
                prepare=True,
            )

        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        assert graph.head("transfer_spine").id == second_id
