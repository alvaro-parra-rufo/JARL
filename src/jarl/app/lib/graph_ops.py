"""Experiment graph operations for the Runner Lab."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.viz import plot_dag
from jarl.operations.graph.checkout import CheckoutRequest, checkout
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.operations.graph.extend import ExtendRequest, extend
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.operations.graph.pin import PinRequest, pin
from jarl.operations.graph.promote import PromoteRequest, promote
from jarl.training.config import RLRunConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _load_graph(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    return ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)


def render_dag_figure(
    graph: ExperimentGraph[RLRunConfig],
    *,
    metric_key: str = "rollout/episode_return",
) -> Figure:
    """Render the experiment DAG to a matplotlib figure."""
    return plot_dag(graph, metric_key=metric_key)


def render_dag_png(exp_dir: Path, *, metric_key: str = "rollout/episode_return", revision: int = 0) -> bytes:
    """Render the experiment DAG to PNG bytes.

    Deprecated for Runner Lab UI; prefer the live tree explorer. Kept for exports.
    """
    del revision
    graph = _load_graph(exp_dir)
    figure = render_dag_figure(graph, metric_key=metric_key)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return buffer.getvalue()


def checkout_node(
    exp_dir: Path,
    node_id: str,
    *,
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> None:
    """Move the current node pointer without forking."""
    active_graph = graph or _load_graph(exp_dir)
    checkout(active_graph, CheckoutRequest(node_id=node_id))


def create_prepared_root(
    exp_dir: Path,
    *,
    config: RLRunConfig,
    label: str,
    branch: str = "main",
) -> str:
    """Create a root node in ``prepared`` status without training."""
    graph = ExperimentGraph(exp_dir, base_config=config)
    response = create_root(
        graph,
        CreateRootRequest(label=label, branch=branch, config=config, preset="custom"),
    )
    return response.node_id


def extend_branch(
    exp_dir: Path,
    *,
    from_node: str | None = None,
    branch: str | None = None,
    label: str = "",
    config_overrides: dict[str, Any] | None = None,
    from_checkpoint: CheckpointRef | None = None,
    prepare: bool = False,
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> str:
    """Extend a branch from its head.

    When ``from_node`` is set, it must match the head of ``target_branch``
    (derived from ``branch`` or the node's branch). Prefer passing only
    ``branch``; ``from_node`` exists for legacy callers and validation only.
    """
    active_graph = graph or _load_graph(exp_dir)
    if from_node is not None:
        parent = active_graph.get_node(from_node)
        target_branch = branch or parent.branch
        active_graph.require_branch_head(from_node, branch=target_branch)
    else:
        target_branch = branch or active_graph.current_node.branch

    response = extend(
        active_graph,
        ExtendRequest(
            label=label,
            branch=target_branch,
            config_overrides=config_overrides,
            from_checkpoint=from_checkpoint,
            prepare=prepare,
        ),
    )
    return response.node_id


def fork_branch(
    exp_dir: Path,
    *,
    from_node: str | None = None,
    branch: str,
    label: str = "",
    config_overrides: dict[str, Any] | None = None,
    from_checkpoint: CheckpointRef | None = None,
    prepare: bool = False,
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> str:
    """Fork a new branch from a parent node."""
    active_graph = graph or _load_graph(exp_dir)
    response = fork(
        active_graph,
        ForkRequest(
            branch=branch,
            label=label,
            from_node=from_node,
            config_overrides=config_overrides,
            from_checkpoint=from_checkpoint,
            prepare=prepare,
        ),
    )
    return response.node_id


def fork_with_learning_rate(
    exp_dir: Path,
    *,
    learning_rate: float,
    branch: str,
    from_node: str | None = None,
    label: str = "lr_sweep",
    from_checkpoint: CheckpointRef | None = None,
    prepare: bool = False,
) -> str:
    """Fork from a node with a new learning rate and optional checkpoint."""
    return fork_branch(
        exp_dir,
        from_node=from_node,
        branch=branch,
        label=label,
        config_overrides={"algorithm.learning_rate": learning_rate},
        from_checkpoint=from_checkpoint,
        prepare=prepare,
    )


def fork_from_checkpoint(
    exp_dir: Path,
    *,
    from_node: str,
    checkpoint_step: int,
    branch: str,
    label: str,
    config_overrides: dict[str, Any] | None = None,
    prepare: bool = False,
) -> str:
    """Fork a child node restoring parent weights from a concrete checkpoint."""
    checkpoint = CheckpointRef(node_id=from_node, checkpoint_step=checkpoint_step)
    return fork_branch(
        exp_dir,
        from_node=from_node,
        branch=branch,
        label=label,
        config_overrides=config_overrides,
        from_checkpoint=checkpoint,
        prepare=prepare,
    )


def extend_from_checkpoint(
    exp_dir: Path,
    *,
    from_node: str,
    checkpoint_step: int,
    label: str,
    branch: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    prepare: bool = False,
) -> str:
    """Extend a branch restoring parent weights from a concrete checkpoint."""
    checkpoint = CheckpointRef(node_id=from_node, checkpoint_step=checkpoint_step)
    return extend_branch(
        exp_dir,
        from_node=from_node,
        branch=branch,
        label=label,
        config_overrides=config_overrides,
        from_checkpoint=checkpoint,
        prepare=prepare,
    )


def promote_checkpoint_best(
    exp_dir: Path,
    *,
    node_id: str,
    metric: str = "eval/episode_return",
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> int | None:
    """Promote the best saved checkpoint for a metric."""
    active_graph = graph or _load_graph(exp_dir)
    response = promote(active_graph, PromoteRequest(node_id=node_id, metric=metric))
    return response.checkpoint_step


def pin_checkpoint(
    exp_dir: Path,
    *,
    node_id: str,
    checkpoint_step: int,
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> None:
    """Pin a checkpoint step on a node."""
    active_graph = graph or _load_graph(exp_dir)
    pin(active_graph, PinRequest(node_id=node_id, checkpoint_step=checkpoint_step))
