"""Prepared Navix root that failed mid-training with a restorable checkpoint."""

from __future__ import annotations

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.training.config import EnvironmentConfig, RLRunConfig

__all__ = [
    "CASE",
    "FAILED_ERROR_MESSAGE",
    "RESUMABLE_CHECKPOINT_STEP",
    "NavixRootFailedResumableCase",
]

RESUMABLE_CHECKPOINT_STEP = 10
"""Checkpoint step available for resume after the simulated failure."""

FAILED_ERROR_MESSAGE = "simulated training failure"
"""Error message recorded on the failed execution attempt."""


class NavixRootFailedResumableCase(ExperimentCase[RLRunConfig]):
    """Materialize a failed Navix root with one restorable checkpoint."""

    def __init__(self) -> None:
        """Initialize the failed-resumable prepared-root case."""
        super().__init__(
            CaseSpec(
                id="navix_root_failed_resumable",
                title="Navix root failed with resumable checkpoint",
                description=("Failed Navix Empty root that still has a restorable checkpoint for resume."),
                tags=frozenset({"navix", "failed", "resume", "checkpoints"}),
                requirements=frozenset({"navix"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Create the root, save a checkpoint, and leave the node failed."""
        config = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"),
        )
        graph = ExperimentGraph(context.experiment_dir, base_config=config)
        response = create_root(
            graph,
            CreateRootRequest(
                label="failed_baseline",
                branch="main",
                config=config,
            ),
        )
        workspace = graph.get_node(response.node_id)
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        try:
            with workspace:
                workspace.save_checkpoint(
                    RESUMABLE_CHECKPOINT_STEP,
                    {"value": RESUMABLE_CHECKPOINT_STEP},
                    metrics={"eval/episode_return": 0.5},
                    force=True,
                )
                raise RuntimeError(FAILED_ERROR_MESSAGE)
        except RuntimeError:
            pass
        if workspace.status != NodeStatus.FAILED:
            msg = f"Expected failed node status, got {workspace.status!r}."
            raise RuntimeError(msg)
        if workspace.resolve_resume_checkpoint_record() is None:
            msg = "Expected a restorable checkpoint after the simulated failure."
            raise RuntimeError(msg)
        graph.save()
        context.register_alias("root", response.node_id)


CASE = NavixRootFailedResumableCase()
"""Default failed-resumable Navix root case."""
