"""Tests for the trained Empty-5x5 rollout-ready PPO root case."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.agents.ppo.checkpoint_state import CHECKPOINT_KIND_PPO, PPOCheckpoint
from jarl.experiments.cases.builtin.navix_ppo_rollout_trained_checkpoint import (
    CASE,
    ROLLOUT_CHECKPOINT_STEP,
    ROLLOUT_ENV_ID,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SAVED_CHECKPOINT_COUNT,
    ROLLOUT_TRAINING_TIMESTEPS,
)
from jarl.experiments.io.checkpoints import CheckpointStatus
from jarl.experiments.node import NodeStatus
from tests.helpers.jax_training_subprocess import materialize_case_in_training_subprocess


@pytest.mark.slow
class TestNavixPpoRolloutTrainedCheckpointCase:
    """Materializes a completed root with periodic typed PPO checkpoints."""

    def test_materializes_completed_root_with_restorable_checkpoint(self, tmp_path: Path) -> None:
        context = materialize_case_in_training_subprocess(CASE, tmp_path / "rollout_trained")

        graph = context.reload_graph()
        root = graph.get_node(context.node_id("root"))
        config = graph.resolve_config(root)
        checkpoint = PPOCheckpoint.from_pytree(root.load_checkpoint(ROLLOUT_CHECKPOINT_STEP))

        saved = [record for record in root.list_checkpoints() if record.status is CheckpointStatus.SAVED]
        eval_returns = {
            float(record.metrics["eval/episode_return"]) for record in saved if "eval/episode_return" in record.metrics
        }
        eval_lengths = {
            float(record.metrics["eval/episode_length"]) for record in saved if "eval/episode_length" in record.metrics
        }

        assert graph.as_networkx().number_of_nodes() == 1
        assert root.status is NodeStatus.COMPLETED
        assert root.branch == "main"
        assert len(saved) == ROLLOUT_SAVED_CHECKPOINT_COUNT
        assert len(eval_returns) >= 2
        assert len(eval_lengths) >= 2
        assert root.has_saved_checkpoint(ROLLOUT_CHECKPOINT_STEP)
        assert config.environment.env_id == ROLLOUT_ENV_ID
        assert config.environment.max_episode_steps == ROLLOUT_MAX_STEPS
        assert config.algorithm.total_timesteps == ROLLOUT_TRAINING_TIMESTEPS
        assert checkpoint.kind == CHECKPOINT_KIND_PPO
