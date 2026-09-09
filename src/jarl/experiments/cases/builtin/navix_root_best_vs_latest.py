"""Prepared Navix root whose best checkpoint is not the latest."""

from __future__ import annotations

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_LATEST,
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
)
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.operations.graph.promote import PromoteRequest, promote
from jarl.training.config import EnvironmentConfig, RLRunConfig

__all__ = [
    "BEST_CHECKPOINT_STEP",
    "CASE",
    "LATEST_CHECKPOINT_STEP",
    "NavixRootBestVsLatestCase",
]

BEST_CHECKPOINT_STEP = 20
"""Checkpoint step with the best evaluation in this scenario."""

LATEST_CHECKPOINT_STEP = 30
"""Later checkpoint that remains ``latest`` with a worse evaluation."""


class NavixRootBestVsLatestCase(ExperimentCase[RLRunConfig]):
    """Materialize a prepared root where ``best`` and ``latest`` diverge."""

    def __init__(self) -> None:
        """Initialize the diverging-checkpoint prepared-root case."""
        super().__init__(
            CaseSpec(
                id="navix_root_best_vs_latest",
                title="Navix root with best vs latest checkpoints",
                description=("Prepared Navix Empty root whose best eval checkpoint is older than latest."),
                tags=frozenset({"navix", "prepared", "root", "checkpoints", "best"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root and register diverging best/latest checkpoints."""
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
        workspace = graph.get_node(response.node_id)
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(
            BEST_CHECKPOINT_STEP,
            {"value": BEST_CHECKPOINT_STEP},
            metrics={
                CHECKPOINT_BEST_RETURN_METRIC: 0.9,
                CHECKPOINT_BEST_LENGTH_METRIC: 12.0,
            },
            force=True,
        )
        workspace.save_checkpoint(
            LATEST_CHECKPOINT_STEP,
            {"value": LATEST_CHECKPOINT_STEP},
            metrics={
                CHECKPOINT_BEST_RETURN_METRIC: 0.4,
                CHECKPOINT_BEST_LENGTH_METRIC: 40.0,
            },
            force=True,
        )
        promote(graph, PromoteRequest(node_id=workspace.id))
        latest = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST)
        best = workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_BEST)
        if latest is None or latest.checkpoint_step != LATEST_CHECKPOINT_STEP:
            msg = "Expected latest alias on the worse later checkpoint."
            raise RuntimeError(msg)
        if best is None or best.checkpoint_step != BEST_CHECKPOINT_STEP:
            msg = "Expected best alias on the earlier higher-return checkpoint."
            raise RuntimeError(msg)
        context.register_alias("root", response.node_id)


CASE = NavixRootBestVsLatestCase()
"""Default diverging best/latest prepared-root case."""
