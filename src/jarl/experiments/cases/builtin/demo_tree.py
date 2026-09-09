"""Built-in three-node experiment tree case."""

from __future__ import annotations

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.operations.graph.extend import ExtendRequest, extend
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.training.config import EnvironmentConfig, RLRunConfig

__all__ = ["CASE", "DemoTreeCase"]


class DemoTreeCase(ExperimentCase[RLRunConfig]):
    """Materialize a prepared root and a two-node experimental branch."""

    def __init__(self) -> None:
        """Initialize the built-in demo tree case."""
        super().__init__(
            CaseSpec(
                id="demo_tree",
                title="Demo experiment tree",
                description="Prepared main root plus two prepared nodes on the exp branch.",
                tags=frozenset({"navix", "prepared", "tree"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root, fork, and branch extension."""
        config = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"),
        )
        graph = ExperimentGraph(context.experiment_dir, base_config=config)
        root = create_root(
            graph,
            CreateRootRequest(
                label="root",
                branch="main",
                config=config,
            ),
        )
        first = fork(
            graph,
            ForkRequest(
                branch="exp",
                from_node=root.node_id,
                label="first",
                prepare=True,
            ),
        )
        second = extend(
            graph,
            ExtendRequest(
                branch="exp",
                label="second",
                prepare=True,
            ),
        )
        context.register_alias("root", root.node_id)
        context.register_alias("first", first.node_id)
        context.register_alias("second", second.node_id)


CASE = DemoTreeCase()
"""Default demo tree case instance."""
