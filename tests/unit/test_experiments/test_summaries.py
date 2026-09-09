"""Tests for ``jarl.experiments.summaries``."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.summaries import config_highlights, load_experiment_summary
from jarl.training.config import RLRunConfig


def test_load_experiment_summary_counts_nodes(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    graph.save()

    summary = load_experiment_summary(exp_dir)

    assert summary.node_count == 1
    assert summary.active_node_id == summary.nodes[0].node_id
    assert summary.branch_heads == {"main": summary.nodes[0].node_id}


def test_config_highlights_includes_algorithm_and_env() -> None:
    config = RLRunConfig()

    highlights = config_highlights(config)

    assert highlights["algorithm"] == config.algorithm.name
    assert highlights["env"] == config.environment.env_id
    assert highlights["nr_envs"] == config.environment.nr_envs
    assert "reward" not in highlights
    assert "scenario_reward_id" not in highlights
    assert "scenario_reward_version" not in highlights
