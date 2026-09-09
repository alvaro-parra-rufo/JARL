"""Tests for parent environment lookup during fork checkpoint restore."""

from __future__ import annotations

from pathlib import Path

from jarl.agents.ppo.checkpoint_restore import load_parent_environment_config
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig
from tests.helpers.minimal_navix_configs import minimal_ppo_run_config


def test_load_parent_environment_config_reads_sibling_node_config(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    config = minimal_ppo_run_config().apply_overrides(
        {
            "environment.env_id": "Navix-Empty-5x5-v0",
            "environment.nr_envs": 8,
            "environment.seed": 21,
        }
    )
    graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(exp_dir, base_config=config)
    root = graph.create_root(config, label="parent")
    child = graph.fork("alt", from_node=root, label="child")
    graph.save()

    parent_env = load_parent_environment_config(child)

    assert parent_env is not None
    assert parent_env.env_id == "Navix-Empty-5x5-v0"
    assert parent_env.nr_envs == 8
    assert parent_env.seed == 21
