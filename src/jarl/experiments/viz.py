"""Experiment tree visualization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize

from jarl.experiments.git_tree_svg import build_git_tree_png, build_git_tree_svg
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "build_git_tree_png",
    "build_git_tree_svg",
    "plot_dag",
]


def _node_metric(ws: NodeWorkspace, metric_key: str) -> float | None:
    """Return one latest scalar metric for plotting, preferring ``metrics.jsonl``."""
    value = ws.latest_metrics().get(metric_key)
    return float(value) if value is not None else None


@overload
def plot_dag(
    graph: ExperimentGraph,
    *,
    metric_key: str = "test_accuracy",
    ax: Axes,
    cmap: str = "viridis",
) -> Axes: ...
@overload
def plot_dag(
    graph: ExperimentGraph,
    *,
    metric_key: str = "test_accuracy",
    figsize: tuple[float, float] = (10.0, 8.0),
    cmap: str = "viridis",
) -> Figure: ...


def plot_dag(
    graph: ExperimentGraph,
    *,
    metric_key: str = "test_accuracy",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    cmap: str = "viridis",
) -> Figure | Axes:
    """Draw the experiment tree with nodes colored and sized by a metric.

    Args:
        graph: Experiment graph to visualize.
        metric_key: Metric used for node color and size.
        ax: Existing matplotlib axes. Creates a new figure when omitted.
        figsize: Figure size when creating a new figure.
        cmap: Matplotlib colormap name for metric values.

    Returns:
        The matplotlib `Figure` or `Axes` used for the plot.
    """
    tree = graph.as_networkx()
    if tree.number_of_nodes() == 0:
        msg = "Experiment graph has no nodes to plot."
        raise ValueError(msg)

    metric_values = {ws.id: _node_metric(ws, metric_key) for ws in graph.all_nodes.values()}
    present_values = [value for value in metric_values.values() if value is not None]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_axes = True
    else:
        created_axes = False

    pos = nx.spring_layout(tree, seed=0)
    node_ids = list(tree.nodes())

    if present_values:
        norm = Normalize(vmin=min(present_values), vmax=max(present_values))
        colors = [
            plt.get_cmap(cmap)(norm(metric_values[node_id])) if metric_values[node_id] is not None else "#cccccc"
            for node_id in node_ids
        ]
        sizes = [
            800.0 if metric_values[node_id] is None else 400.0 + 1600.0 * norm(metric_values[node_id])
            for node_id in node_ids
        ]
    else:
        colors = ["#cccccc"] * len(node_ids)
        sizes = [800.0] * len(node_ids)

    nx.draw_networkx_nodes(tree, pos, nodelist=node_ids, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(tree, pos, arrows=True, arrowsize=15, ax=ax)
    nx.draw_networkx_labels(tree, pos, labels={node_id: node_id for node_id in node_ids}, font_size=8, ax=ax)

    ax.set_title(f"Experiment tree ({metric_key})")
    ax.axis("off")

    if present_values:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=metric_key)

    figure = ax.figure
    if figure is not None:
        figure.tight_layout()
    if created_axes and figure is not None:
        return figure
    return ax
