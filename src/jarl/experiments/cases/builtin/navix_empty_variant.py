"""Built-in EmptyVariant root with the default task mix and overlay snapshot."""

from __future__ import annotations

from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.metadata import ExperimentCaseMetadata
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig

__all__ = ["CASE", "NavixEmptyVariantCase"]


class NavixEmptyVariantCase(ExperimentCase[RLRunConfig]):
    """Materialize one prepared EmptyVariant root on ``main``."""

    def __init__(self) -> None:
        """Initialize the built-in EmptyVariant root case."""
        super().__init__(
            CaseSpec(
                id="navix_empty_variant",
                title="Prepared EmptyVariant root",
                description=(
                    "One prepared Navix EmptyVariant root on the main branch "
                    "with the default task mix and native horizon."
                ),
                tags=frozenset({"navix", "prepared"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the prepared root and persist the case snapshot."""
        config = RLRunConfig(
            environment=EnvironmentConfig(
                env_id=EMPTY_VARIANT_ENV_ID,
                max_episode_steps=None,
                reward=RewardWeightsConfig(goal_reached=1.0),
                scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
                scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
            ),
            algorithm=AlgorithmConfig(gamma=0.99),
        )
        graph = ExperimentGraph(context.experiment_dir, base_config=config)
        response = create_root(
            graph,
            CreateRootRequest(
                label="root",
                branch="main",
                config=config,
            ),
        )
        context.register_alias("root", response.node_id)
        ExperimentCaseMetadata(
            case_id=self.spec.id,
            env_id=EMPTY_VARIANT_ENV_ID,
            scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
            scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
        ).save(graph.layout.case_metadata_path)


CASE = NavixEmptyVariantCase()
"""Default EmptyVariant root case instance."""
