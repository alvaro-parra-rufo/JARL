"""Global RL training runner orchestrating experiment graph and trainer.

Orchestrates ``ExperimentGraph``, ``NodeWorkspace``, schedule derivation, and
an injected ``Trainer``. Does not implement PPO or environment logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarl.env_setup import setup_jax
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.layout import ExperimentLayout
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.schedule import (
    TrainingSchedule,
    apply_training_schedule,
    compute_training_schedule,
    format_training_schedule_message,
)
from jarl.training.trainer import EnvFactory, ScheduleBuilder, Trainer

__all__ = [
    "RunTrainingResult",
    "resume_training",
    "run_training",
]


@dataclass(frozen=True, slots=True)
class RunTrainingResult:
    """Outcome of a single ``run_training`` invocation.

    Args:
        graph: Experiment graph mutated by the run (saved to disk before return).
        workspace: Node workspace that executed training.
        config: Fully resolved config including derived schedule fields.
        schedule: Derived schedule counters passed to the trainer.
    """

    graph: ExperimentGraph[RLRunConfig]
    workspace: NodeWorkspace
    config: RLRunConfig
    schedule: TrainingSchedule


def run_training(
    *,
    trainer: Trainer,
    experiment_dir: str | Path,
    graph: ExperimentGraph[RLRunConfig] | None = None,
    node: str | NodeWorkspace | None = None,
    config: RLRunConfig | None = None,
    config_overrides: dict[str, Any] | None = None,
    schedule_builder: ScheduleBuilder = apply_training_schedule,
    env_factory: EnvFactory | None = None,
    create_root: bool = False,
    branch: str = "main",
    label: str = "",
) -> RunTrainingResult:
    """Run training on an experiment node through the graph + workspace lifecycle.

    Model mental: experiment graph → target node → resolved config → schedule →
    trainer. Tracking is propagated from ``config.tracking`` when nodes are
    created by the graph; the runner does not remap tracking fields.

    Args:
        trainer: Callable that performs training inside an active workspace.
        experiment_dir: Root directory for the experiment artifacts.
        graph: Optional in-memory graph. Loaded from ``experiment_dir`` when
            omitted.
        node: Target node id or workspace. Defaults to the graph current node.
        config: Base config used when ``create_root=True``.
        config_overrides: Sparse overrides merged before schedule derivation.
        schedule_builder: Callable that populates derived schedule fields.
        env_factory: Optional environment factory forwarded to ``trainer``.
        create_root: When ``True``, create a root node if the experiment is empty.
        branch: Branch name used when creating a root node.
        label: Label used when creating a root node.

    Returns:
        Run result containing the saved graph, workspace, config, and schedule.

    Raises:
        RuntimeError: If the experiment has no nodes and ``create_root`` is false.
        ValueError: If ``create_root`` is requested but ``config`` is missing.
    """
    exp_dir = Path(experiment_dir)
    active_graph = graph or _load_or_empty_graph(exp_dir)

    if active_graph.as_networkx().number_of_nodes() == 0:
        if not create_root:
            msg = "Experiment has no nodes. Pass create_root=True with a base config."
            raise RuntimeError(msg)
        if config is None:
            msg = "config is required when create_root=True."
            raise ValueError(msg)
        active_graph = ExperimentGraph(exp_dir, base_config=config)
        active_graph.create_root(
            config=config,
            branch=branch,
            label=label,
        )

    workspace = _resolve_workspace(active_graph, node)
    resolved = active_graph.resolve_config(workspace)
    if config_overrides:
        resolved = resolved.apply_overrides(config_overrides)

    schedule = compute_training_schedule(resolved.environment, resolved.algorithm)
    resolved_with_schedule = schedule_builder(resolved)
    workspace.save_resolved_config(resolved_with_schedule)
    workspace.bind_wandb_run_context(resolved_with_schedule, schedule)

    logger = logging.getLogger(resolved_with_schedule.logging.logger_name)
    setup_jax(resolved_with_schedule.jax)
    try:
        with workspace:
            logger.info(format_training_schedule_message(schedule))
            trainer(
                workspace,
                resolved_with_schedule,
                schedule,
                env_factory=env_factory,
            )
    finally:
        active_graph.save()
    return RunTrainingResult(
        graph=active_graph,
        workspace=workspace,
        config=resolved_with_schedule,
        schedule=schedule,
    )


def _load_or_empty_graph(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    layout = ExperimentLayout(exp_dir)
    if layout.manifest_path.is_file():
        return ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentGraph(exp_dir, base_config=None)


def _resolve_workspace(
    graph: ExperimentGraph[RLRunConfig],
    node: str | NodeWorkspace | None,
) -> NodeWorkspace:
    if node is None:
        return graph.current_node
    if isinstance(node, NodeWorkspace):
        return node
    return graph.get_node(node)


def resume_training(
    *,
    trainer: Trainer,
    experiment_dir: str | Path,
    graph: ExperimentGraph[RLRunConfig] | None = None,
    node: str | NodeWorkspace,
    config_overrides: dict[str, Any] | None = None,
    schedule_builder: ScheduleBuilder = apply_training_schedule,
    env_factory: EnvFactory | None = None,
) -> RunTrainingResult:
    """Resume training on a failed or interrupted node.

    Args:
        trainer: Callable that performs training inside an active workspace.
        experiment_dir: Root directory for the experiment artifacts.
        graph: Optional in-memory graph. Loaded from ``experiment_dir`` when
            omitted.
        node: Failed or interrupted node to resume.
        config_overrides: Optional sparse overrides applied before schedule derivation.
        schedule_builder: Callable that populates derived schedule fields.
        env_factory: Optional environment factory forwarded to ``trainer``.

    Returns:
        Run result after the resumed attempt completes.

    Raises:
        RuntimeError: If the node is not in a resumable status.
    """
    active_graph = graph or ExperimentGraph.from_directory(experiment_dir, config_cls=RLRunConfig)
    workspace = _resolve_workspace(active_graph, node)
    if workspace.status not in {NodeStatus.FAILED, NodeStatus.INTERRUPTED}:
        msg = f"Node {workspace.id!r} is not resumable from status {workspace.status!r}."
        raise RuntimeError(msg)
    return run_training(
        trainer=trainer,
        experiment_dir=experiment_dir,
        graph=active_graph,
        node=workspace,
        config_overrides=config_overrides,
        schedule_builder=schedule_builder,
        env_factory=env_factory,
    )
