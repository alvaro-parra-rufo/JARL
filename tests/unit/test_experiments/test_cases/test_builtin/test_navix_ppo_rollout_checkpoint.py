"""Tests for the built-in rollout-ready PPO root case."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.agents.ppo.checkpoint_state import CHECKPOINT_KIND_PPO, PPOCheckpoint
from jarl.experiments.cases.builtin._navix_rollout_checkpoint import ROLLOUT_CHECKPOINT_STEP
from jarl.experiments.cases.builtin.navix_ppo_rollout_checkpoint import (
    CASE,
    ROLLOUT_ENV_ID,
    ROLLOUT_MAX_STEPS,
)
from jarl.experiments.node import NodeStatus

# Compiles a Navix env; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow


class TestNavixPpoRolloutCheckpointCase:
    """Materializes a prepared root with one typed PPO checkpoint."""

    def test_materializes_prepared_root_with_restorable_checkpoint(self, tmp_path: Path) -> None:
        context = CASE.materialize(tmp_path / "rollout")

        graph = context.reload_graph()
        root = graph.get_node(context.node_id("root"))
        config = graph.resolve_config(root)
        checkpoint = PPOCheckpoint.from_pytree(root.load_checkpoint(ROLLOUT_CHECKPOINT_STEP))

        assert graph.as_networkx().number_of_nodes() == 1
        assert root.status is NodeStatus.PREPARED
        assert root.branch == "main"
        assert root.has_saved_checkpoint(ROLLOUT_CHECKPOINT_STEP)
        assert config.environment.env_id == ROLLOUT_ENV_ID
        assert config.environment.max_episode_steps == ROLLOUT_MAX_STEPS
        assert checkpoint.kind == CHECKPOINT_KIND_PPO
