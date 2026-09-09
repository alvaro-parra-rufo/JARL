"""CPU integration tests for checkpoint rollout on real PPO checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.cases.builtin._navix_rollout_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SEED,
)
from jarl.experiments.cases.builtin.navix_ppo_gru_rollout_checkpoint import (
    CASE as PPO_GRU_CASE,
)
from jarl.experiments.cases.builtin.navix_ppo_rollout_trained_checkpoint import (
    CASE as PPO_TRAINED_CASE,
)
from jarl.experiments.cases.builtin.navix_ppo_rollout_trained_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP as TRAINED_ROLLOUT_CHECKPOINT_STEP,
)
from jarl.experiments.cases.case import ExperimentCase
from jarl.operations.graph.checkpoint_rollout import (
    CheckpointRolloutRequest,
    checkpoint_rollout,
)
from jarl.training.config import RLRunConfig
from tests.helpers.jax_training_subprocess import materialize_case_in_training_subprocess


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "algorithm_name", "checkpoint_step"),
    [
        pytest.param(PPO_TRAINED_CASE, "ppo.full_jax.navix", TRAINED_ROLLOUT_CHECKPOINT_STEP, id="ppo_trained"),
        pytest.param(PPO_GRU_CASE, "ppo_gru.full_jax.navix", ROLLOUT_CHECKPOINT_STEP, id="ppo_gru"),
    ],
)
def test_checkpoint_rollout_cpu(
    case: ExperimentCase[RLRunConfig],
    algorithm_name: str,
    checkpoint_step: int,
    tmp_path: Path,
) -> None:
    destination = tmp_path / case.spec.id
    if case.spec.id == PPO_TRAINED_CASE.spec.id:
        context = materialize_case_in_training_subprocess(case, destination)
    else:
        context = case.materialize(destination)
    graph = context.reload_graph()

    first = checkpoint_rollout(
        graph,
        CheckpointRolloutRequest(
            checkpoint_step=checkpoint_step,
            seed=ROLLOUT_SEED,
            max_steps=ROLLOUT_MAX_STEPS,
        ),
    )
    second = checkpoint_rollout(
        graph,
        CheckpointRolloutRequest(
            checkpoint_step=checkpoint_step,
            seed=ROLLOUT_SEED,
            max_steps=ROLLOUT_MAX_STEPS,
        ),
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.rollout_id == second.rollout_id
    assert first.algorithm_name == algorithm_name
    assert first.seed == ROLLOUT_SEED
    assert first.summary.outcome.length >= 0
    assert first.summary.artifact_paths
    compact = first.to_compact_dict()
    if algorithm_name == "ppo.full_jax.navix":
        checkpoint_eval = compact.get("checkpoint_eval")
        assert isinstance(checkpoint_eval, dict)
        assert "episode_return" in checkpoint_eval
        assert "episode_length" in checkpoint_eval
    workspace = graph.get_node(context.node_id("root"))
    for relative_path in first.summary.artifact_paths:
        assert (workspace.path / relative_path).is_file()
