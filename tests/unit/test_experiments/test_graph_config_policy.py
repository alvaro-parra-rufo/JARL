"""Tests for immutable overrides and Navix transfer validation in ExperimentGraph."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.envs.navix.catalog import NavixMapContract, register_map
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.training.config import EnvironmentConfig, RLRunConfig


@pytest.fixture()
def rl_graph(tmp_path: Path) -> ExperimentGraph[RLRunConfig]:
    """Experiment graph with a Navix root config."""
    config = RLRunConfig(environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"))
    graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(tmp_path / "exp", base_config=config)
    graph.create_root(config=config, branch="main", label="baseline")
    return graph


class TestImmutableOverrides:
    """Tests for IMMUTABLE_PATHS enforcement on fork/extend."""

    def test_fork_allows_mutable_algorithm_override(self, rl_graph: ExperimentGraph[RLRunConfig]) -> None:
        root = rl_graph.head("main")
        child_config = rl_graph.resolve_config(root).apply_overrides({"algorithm.learning_rate": 1e-5})

        child = rl_graph.fork("lr_fork", from_node=root, config=child_config)

        assert rl_graph.resolve_config(child).algorithm.learning_rate == 1e-5

    def test_fork_rejects_immutable_algorithm_name(self, rl_graph: ExperimentGraph[RLRunConfig]) -> None:
        root = rl_graph.head("main")
        child_config = rl_graph.resolve_config(root).apply_overrides({"algorithm.name": "other.agent"})

        with pytest.raises(ValueError, match="immutable paths"):
            rl_graph.fork("bad_fork", from_node=root, config=child_config)

    def test_extend_rejects_immutable_gru_hidden_dim(self, rl_graph: ExperimentGraph[RLRunConfig]) -> None:
        child_config = rl_graph.resolve_config(rl_graph.head("main")).apply_overrides({"algorithm.gru_hidden_dim": 128})

        with pytest.raises(ValueError, match="immutable paths"):
            rl_graph.extend("main", config=child_config)


class TestTransferGroup:
    """Tests for experiment transfer group pinning and map compatibility."""

    def test_create_root_persists_transfer_group(self, tmp_path: Path) -> None:
        config = RLRunConfig(environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"))
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(tmp_path / "exp", base_config=config)
        graph.create_root(config=config, branch="main")

        metadata = json.loads((tmp_path / "exp" / "run_metadata.json").read_text(encoding="utf-8"))

        assert metadata["transfer_group"] == "navix_symbolic_fp_147_7"

    def test_fork_allows_compatible_env_change(self, rl_graph: ExperimentGraph[RLRunConfig]) -> None:
        root = rl_graph.head("main")
        child_config = rl_graph.resolve_config(root).apply_overrides({"environment.env_id": "Navix-DoorKey-5x5-v0"})

        child = rl_graph.fork("map_fork", from_node=root, config=child_config)

        assert rl_graph.resolve_config(child).environment.env_id == "Navix-DoorKey-5x5-v0"

    def test_fork_rejects_incompatible_env_change(self, rl_graph: ExperimentGraph[RLRunConfig]) -> None:
        register_map(
            NavixMapContract(
                env_id="Navix-Incompatible-Graph-v0",
                processed_obs_shape=(99,),
                action_names=("a", "b"),
                transfer_group="other_group",
            )
        )
        root = rl_graph.head("main")
        child_config = rl_graph.resolve_config(root).apply_overrides(
            {"environment.env_id": "Navix-Incompatible-Graph-v0"}
        )

        with pytest.raises(ValueError, match="transfer groups"):
            rl_graph.fork("bad_map", from_node=root, config=child_config)

    def test_fork_with_checkpoint_validates_compatible_env(
        self,
        rl_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        root = rl_graph.head("main")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        root.save_checkpoint(1, {"step": 1}, force=True)
        child_config = rl_graph.resolve_config(root).apply_overrides({"environment.env_id": "Navix-DoorKey-5x5-v0"})
        checkpoint_ref = CheckpointRef(node_id=root.id, checkpoint_step=1)

        child = rl_graph.fork(
            "resume_map",
            from_node=root,
            config=child_config,
            from_checkpoint=checkpoint_ref,
        )

        assert child.node_metadata.parent_checkpoint_step == 1
