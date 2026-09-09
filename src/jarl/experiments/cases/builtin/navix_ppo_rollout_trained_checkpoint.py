"""Completed Navix Empty-5x5 root after a short PPO training run."""

from __future__ import annotations

from jarl.experiments.cases.builtin._navix_rollout_checkpoint import ROLLOUT_SEED
from jarl.experiments.cases.builtin._rollout_trained_checkpoint import (
    ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY,
    ROLLOUT_TRAINED_ENV_ID,
    ROLLOUT_TRAINED_MAX_STEPS,
    ROLLOUT_TRAINED_TRAINING_TIMESTEPS,
    materialize_rollout_trained_case,
    rollout_trained_checkpoint_step,
    saved_checkpoint_count,
)
from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.training.config import RLRunConfig

__all__ = [
    "CASE",
    "ROLLOUT_CHECKPOINT_EVAL_FREQUENCY",
    "ROLLOUT_CHECKPOINT_STEP",
    "ROLLOUT_ENV_ID",
    "ROLLOUT_MAX_STEPS",
    "ROLLOUT_SAVED_CHECKPOINT_COUNT",
    "ROLLOUT_SEED",
    "ROLLOUT_TRAINING_TIMESTEPS",
    "NavixPpoRolloutTrainedCheckpointCase",
]

ROLLOUT_ENV_ID = ROLLOUT_TRAINED_ENV_ID
ROLLOUT_MAX_STEPS = ROLLOUT_TRAINED_MAX_STEPS
ROLLOUT_TRAINING_TIMESTEPS = ROLLOUT_TRAINED_TRAINING_TIMESTEPS
ROLLOUT_CHECKPOINT_EVAL_FREQUENCY = ROLLOUT_TRAINED_CHECKPOINT_EVAL_FREQUENCY
ROLLOUT_SAVED_CHECKPOINT_COUNT = saved_checkpoint_count(
    evaluation_and_save_frequency=ROLLOUT_CHECKPOINT_EVAL_FREQUENCY,
)
ROLLOUT_CHECKPOINT_STEP = rollout_trained_checkpoint_step(
    evaluation_and_save_frequency=ROLLOUT_CHECKPOINT_EVAL_FREQUENCY,
)


class NavixPpoRolloutTrainedCheckpointCase(ExperimentCase[RLRunConfig]):
    """Materialize a completed root with periodic feed-forward PPO checkpoints."""

    def __init__(self) -> None:
        """Initialize the trained rollout-ready PPO root case."""
        super().__init__(
            CaseSpec(
                id="navix_ppo_rollout_trained_checkpoint",
                title="Navix PPO trained checkpoint for rollout analysis",
                description=(
                    "Completed Navix Empty-5x5 root after a PPO training run "
                    f"({ROLLOUT_TRAINING_TIMESTEPS} timesteps) with checkpoints every "
                    f"{ROLLOUT_CHECKPOINT_EVAL_FREQUENCY} steps and varied eval metrics."
                ),
                tags=frozenset({"navix", "trained", "root", "checkpoints", "rollout", "ppo"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root, train briefly, and persist periodic typed checkpoints."""
        materialize_rollout_trained_case(
            context,
            evaluation_and_save_frequency=ROLLOUT_CHECKPOINT_EVAL_FREQUENCY,
            label="rollout_ppo_trained",
        )


CASE = NavixPpoRolloutTrainedCheckpointCase()
"""Default trained rollout-ready PPO root case instance."""
