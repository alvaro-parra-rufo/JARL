"""Prepared Navix root with a clearly improving eval return curve."""

from __future__ import annotations

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.training.config import EnvironmentConfig, RLRunConfig

__all__ = ["CASE", "NavixRootImprovingEvalCase"]


class NavixRootImprovingEvalCase(ExperimentCase[RLRunConfig]):
    """Materialize a prepared root with monotonically increasing eval return."""

    def __init__(self) -> None:
        """Initialize the improving-eval prepared-root case."""
        super().__init__(
            CaseSpec(
                id="navix_root_improving_eval",
                title="Navix root with improving eval return",
                description="Prepared Navix Empty root with an increasing eval/episode_return series.",
                tags=frozenset({"navix", "prepared", "root", "metrics", "eval"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root and log an improving eval series."""
        config = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"),
        )
        graph = ExperimentGraph(context.experiment_dir, base_config=config)
        response = create_root(
            graph,
            CreateRootRequest(
                label="improving",
                branch="main",
                config=config,
            ),
        )
        workspace = graph.get_node(response.node_id)
        for step in range(9):
            workspace.log_scalar(step, "eval/episode_return", float(step))
        graph.save()
        context.register_alias("root", response.node_id)


CASE = NavixRootImprovingEvalCase()
"""Default improving-eval Navix root case."""
