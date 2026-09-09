"""Prepared Navix root with one initialized PPO checkpoint for rollout analysis."""

from __future__ import annotations

from jarl.experiments.cases.builtin._navix_rollout_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP,
    ROLLOUT_ENV_ID,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SEED,
    materialize_untrained_rollout_case,
    minimal_rollout_run_config,
)
from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.training.config import RLRunConfig

__all__ = [
    "CASE",
    "ROLLOUT_CHECKPOINT_STEP",
    "ROLLOUT_ENV_ID",
    "ROLLOUT_MAX_STEPS",
    "ROLLOUT_SEED",
    "NavixPpoRolloutCheckpointCase",
]


class NavixPpoRolloutCheckpointCase(ExperimentCase[RLRunConfig]):
    """Materialize a prepared root with one feed-forward PPO checkpoint."""

    def __init__(self) -> None:
        """Initialize the rollout-ready PPO root case."""
        super().__init__(
            CaseSpec(
                id="navix_ppo_rollout_checkpoint",
                title="Navix PPO checkpoint for rollout analysis",
                description=(
                    "Prepared Navix Empty root with one initialized PPO checkpoint "
                    "suitable for greedy rollout inference without full training."
                ),
                tags=frozenset({"navix", "prepared", "root", "checkpoints", "rollout", "ppo"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root and persist one typed PPO checkpoint."""
        materialize_untrained_rollout_case(
            context,
            config=minimal_rollout_run_config(algorithm_name="ppo.full_jax.navix"),
            algorithm_name="ppo.full_jax.navix",
            label="rollout_ppo",
        )


CASE = NavixPpoRolloutCheckpointCase()
"""Default rollout-ready PPO root case instance."""
