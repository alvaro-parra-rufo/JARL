"""Prepared Navix root with one initialized PPO-GRU checkpoint for rollout analysis."""

from __future__ import annotations

from jarl.agents.ppo.checkpoint_state import CHECKPOINT_KIND_PPO_GRU
from jarl.experiments.cases.builtin._navix_rollout_checkpoint import (
    materialize_untrained_rollout_case,
    minimal_rollout_run_config,
)
from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.training.config import RLRunConfig

__all__ = ["CASE", "NavixPpoGruRolloutCheckpointCase"]


class NavixPpoGruRolloutCheckpointCase(ExperimentCase[RLRunConfig]):
    """Materialize a prepared root with one recurrent PPO checkpoint."""

    def __init__(self) -> None:
        """Initialize the rollout-ready PPO-GRU root case."""
        super().__init__(
            CaseSpec(
                id="navix_ppo_gru_rollout_checkpoint",
                title="Navix PPO-GRU checkpoint for rollout analysis",
                description=(
                    "Prepared Navix Empty root with one initialized PPO-GRU checkpoint "
                    "suitable for greedy rollout inference without full training."
                ),
                tags=frozenset({"navix", "prepared", "root", "checkpoints", "rollout", "ppo_gru"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root and persist one typed PPO-GRU checkpoint."""
        materialize_untrained_rollout_case(
            context,
            config=minimal_rollout_run_config(algorithm_name=CHECKPOINT_KIND_PPO_GRU),
            algorithm_name=CHECKPOINT_KIND_PPO_GRU,
            label="rollout_ppo_gru",
        )


CASE = NavixPpoGruRolloutCheckpointCase()
"""Default rollout-ready PPO-GRU root case instance."""
