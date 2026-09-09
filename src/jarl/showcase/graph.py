"""Experiment graph helpers for the showcase pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.training.config import RLRunConfig

__all__ = ["extend_branch", "fork_branch"]


def _apply_overrides(
    graph: ExperimentGraph[RLRunConfig],
    parent_id: str,
    config_overrides: dict[str, Any] | None,
) -> RLRunConfig | None:
    if not config_overrides:
        return None
    parent_config = graph.resolve_config(graph.get_node(parent_id))
    return parent_config.apply_overrides(config_overrides)


def extend_branch(
    exp_dir: Path,
    *,
    from_node: str | None = None,
    branch: str | None = None,
    label: str = "",
    config_overrides: dict[str, Any] | None = None,
    from_checkpoint: CheckpointRef | None = None,
    prepare: bool = False,
) -> str:
    """Extend a branch from its head or the given node."""
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    parent = graph.get_node(from_node) if from_node is not None else graph.current_node
    target_branch = branch or parent.branch
    if from_node is not None and branch is not None:
        head_id = graph.head(branch).id
        if head_id != parent.id:
            msg = f"Cannot extend branch {branch!r} from node {parent.id!r}: branch head is {head_id!r}."
            raise ValueError(msg)
    target_config = _apply_overrides(graph, parent.id, config_overrides)
    child = graph.extend(
        target_branch,
        config=target_config,
        label=label,
        from_checkpoint=from_checkpoint,
        prepare=prepare,
    )
    graph.save()
    return child.id


def fork_branch(
    exp_dir: Path,
    *,
    from_node: str | None = None,
    branch: str,
    label: str = "",
    config_overrides: dict[str, Any] | None = None,
    from_checkpoint: CheckpointRef | None = None,
    prepare: bool = False,
) -> str:
    """Fork a new branch from a parent node."""
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    parent = graph.get_node(from_node) if from_node is not None else graph.current_node
    target_config = _apply_overrides(graph, parent.id, config_overrides)
    child = graph.fork(
        branch,
        from_node=parent,
        config=target_config,
        label=label,
        from_checkpoint=from_checkpoint,
        prepare=prepare,
    )
    graph.save()
    return child.id
