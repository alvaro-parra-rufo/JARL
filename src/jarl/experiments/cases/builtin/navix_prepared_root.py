"""Built-in prepared Navix root case."""

from __future__ import annotations

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.training.config import EnvironmentConfig, RLRunConfig

__all__ = ["CASE", "NavixPreparedRootCase"]


class NavixPreparedRootCase(ExperimentCase[RLRunConfig]):
    """Materialize one prepared Navix Empty root on ``main``."""

    def __init__(self) -> None:
        """Initialize the built-in prepared root case."""
        super().__init__(
            CaseSpec(
                id="navix_prepared_root",
                title="Prepared Navix root",
                description="One prepared Navix Empty root node on the main branch.",
                tags=frozenset({"navix", "prepared", "root"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create and register the prepared baseline root."""
        config = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"),
        )
        graph = ExperimentGraph(context.experiment_dir, base_config=config)
        response = create_root(
            graph,
            CreateRootRequest(
                label="baseline",
                branch="main",
                config=config,
            ),
        )
        context.register_alias("root", response.node_id)


CASE = NavixPreparedRootCase()
"""Default prepared Navix root case instance."""
