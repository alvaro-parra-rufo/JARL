"""Experiment and node summaries for CLI, agents, and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig

__all__ = [
    "ExperimentSummary",
    "NodeSummary",
    "build_experiment_summary",
    "config_highlights",
    "latest_metric",
    "load_experiment_summary",
]


@dataclass(frozen=True, slots=True)
class NodeSummary:
    """Compact node state for tables and feed payloads."""

    node_id: str
    status: str
    branch: str
    label: str
    parent_id: str | None
    step: int
    updated_at: str
    metrics_count: int
    checkpoint_count: int
    artifact_count: int
    latest_return: float | None
    latest_eval_return: float | None
    has_tensorboard: bool
    has_wandb: bool
    has_videos: bool


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Compact experiment state for status tools and dashboards."""

    exp_dir: Path
    active_node_id: str
    active_status: str
    node_count: int
    branch_count: int
    branch_heads: dict[str, str]
    status_counts: dict[str, int]
    nodes: list[NodeSummary]


def load_experiment_summary(exp_dir: Path) -> ExperimentSummary:
    """Load graph state and build structured node summaries."""
    graph: ExperimentGraph[RLRunConfig] = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    return build_experiment_summary(graph)


def build_experiment_summary(graph: ExperimentGraph[RLRunConfig]) -> ExperimentSummary:
    """Build structured node summaries from an in-memory experiment graph."""
    nodes = [_summarize_node(workspace) for workspace in _ordered_nodes(graph)]
    status_counts: dict[str, int] = {}
    for node in nodes:
        status_counts[node.status] = status_counts.get(node.status, 0) + 1
    return ExperimentSummary(
        exp_dir=graph.layout.root,
        active_node_id=graph.current_node.id,
        active_status=graph.current_node.status.value,
        node_count=len(nodes),
        branch_count=len(graph.branch_heads),
        branch_heads={branch: workspace.id for branch, workspace in graph.branch_heads.items()},
        status_counts=status_counts,
        nodes=nodes,
    )


def latest_metric(workspace: NodeWorkspace, name: str) -> float | None:
    """Return one latest scalar metric from a workspace."""
    value = workspace.latest_metrics().get(name)
    return float(value) if value is not None else None


def config_highlights(config: RLRunConfig) -> dict[str, str | int | float | bool]:
    """Return operator-facing config fields for compact preview."""
    return {
        "algorithm": config.algorithm.name,
        "env": config.environment.env_id,
        "seed": config.environment.seed,
        "nr_envs": config.environment.nr_envs,
        "total_timesteps": config.algorithm.total_timesteps,
        "nr_steps": config.algorithm.nr_steps,
        "minibatch": config.algorithm.minibatch_size,
        "learning_rate": config.algorithm.learning_rate,
        "eval_frequency": config.algorithm.evaluation_and_save_frequency,
        "wandb": config.tracking.track_wandb,
        "tensorboard": config.tracking.track_tensorboard,
        "video": config.video.record_video,
    }


def _ordered_nodes(graph: ExperimentGraph[RLRunConfig]) -> list[NodeWorkspace]:
    network = graph.as_networkx()
    ordered_ids = list(network.nodes)
    return [graph.get_node(node_id) for node_id in ordered_ids]


def _summarize_node(workspace: NodeWorkspace) -> NodeSummary:
    latest = workspace.latest_metrics()
    return NodeSummary(
        node_id=workspace.id,
        status=workspace.status.value,
        branch=workspace.branch,
        label=workspace.node_metadata.label,
        parent_id=workspace.node_metadata.parent_id,
        step=workspace.node_metadata.step,
        updated_at=workspace.node_metadata.updated_at,
        metrics_count=_metrics_record_count(workspace),
        checkpoint_count=len(workspace.list_checkpoints()),
        artifact_count=len(workspace.list_artifacts()),
        latest_return=_optional_float(latest.get("rollout/episode_return")),
        latest_eval_return=_optional_float(latest.get("eval/episode_return")),
        has_tensorboard=workspace.tensorboard_dir.is_dir() and any(workspace.tensorboard_dir.iterdir()),
        has_wandb=workspace.wandb_dir.is_dir() and any(workspace.wandb_dir.iterdir()),
        has_videos=workspace.videos_dir.is_dir() and any(workspace.videos_dir.iterdir()),
    )


def _metrics_record_count(workspace: NodeWorkspace) -> int:
    metrics_path = workspace.metrics_jsonl_path
    if not metrics_path.is_file():
        return 0
    return len(JsonlMetricReader(metrics_path).read_records())


def _optional_float(value: float | None) -> float | None:
    return float(value) if value is not None else None
