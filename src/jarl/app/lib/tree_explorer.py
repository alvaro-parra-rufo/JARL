"""Runner Lab live tree explorer helpers."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Literal

from jarl.experiments.git_tree_svg import build_git_tree_exports, build_git_tree_png, build_git_tree_svg
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig

DEFAULT_TREE_REFRESH_SECONDS = 5.0
ACTIVE_RUN_TREE_REFRESH_SECONDS = 2.5
TRAINING_TAB_SESSION_KEY = "training_module_tab"
TreeExportStyle = Literal["vertical", "horizontal", "dag"]
TreeExportFormat = Literal["svg", "png"]

__all__ = [
    "ACTIVE_RUN_TREE_REFRESH_SECONDS",
    "DEFAULT_TREE_REFRESH_SECONDS",
    "TRAINING_TAB_SESSION_KEY",
    "TreeExportFormat",
    "TreeExportStyle",
    "build_tree_export_png",
    "build_tree_export_svg",
    "build_tree_exports",
    "tree_explorer_session_token",
]


def tree_explorer_session_token(exp_dir: Path) -> str:
    """Return a stable session key suffix for one experiment directory."""
    return hashlib.sha256(str(exp_dir.resolve()).encode("utf-8")).hexdigest()[:16]


def build_tree_exports(
    exp_dir: Path,
    *,
    metric_key: str,
    style: TreeExportStyle = "vertical",
) -> tuple[bytes, bytes]:
    """Render the full experiment tree as SVG and PNG bytes.

    Args:
        exp_dir: Experiment root directory.
        metric_key: Metric used for node coloring.
        style: Git-style ``vertical`` / ``horizontal`` explorer layout, or the
            matplotlib spring ``dag``.

    Returns:
        Pair of SVG bytes and PNG bytes.

    Raises:
        ValueError: If ``style`` is not a supported export layout.
    """
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    if style in ("vertical", "horizontal"):
        svg, png = build_git_tree_exports(graph, metric_key=metric_key, orientation=style)
        return svg.encode("utf-8"), png
    if style == "dag":
        return _dag_exports(graph, metric_key=metric_key)
    msg = f"Unknown tree export style: {style!r}"
    raise ValueError(msg)


def build_tree_export_svg(
    exp_dir: Path,
    *,
    metric_key: str,
    style: TreeExportStyle = "vertical",
) -> bytes:
    """Render the full experiment tree as SVG bytes for download.

    Args:
        exp_dir: Experiment root directory.
        metric_key: Metric used for node coloring.
        style: Git-style ``vertical`` / ``horizontal`` explorer layout, or the
            matplotlib spring ``dag``.

    Returns:
        SVG document bytes.

    Raises:
        ValueError: If ``style`` is not a supported export layout.
    """
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    if style in ("vertical", "horizontal"):
        return build_git_tree_svg(graph, metric_key=metric_key, orientation=style).encode("utf-8")
    if style == "dag":
        return _dag_bytes(graph, metric_key=metric_key, fmt="svg")
    msg = f"Unknown tree export style: {style!r}"
    raise ValueError(msg)


def build_tree_export_png(
    exp_dir: Path,
    *,
    metric_key: str,
    style: TreeExportStyle = "vertical",
) -> bytes:
    """Render the full experiment tree as PNG bytes for download.

    Args:
        exp_dir: Experiment root directory.
        metric_key: Metric used for node coloring.
        style: Git-style ``vertical`` / ``horizontal`` explorer layout, or the
            matplotlib spring ``dag``.

    Returns:
        PNG document bytes.

    Raises:
        ValueError: If ``style`` is not a supported export layout.
    """
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    if style in ("vertical", "horizontal"):
        return build_git_tree_png(graph, metric_key=metric_key, orientation=style)
    if style == "dag":
        return _dag_bytes(graph, metric_key=metric_key, fmt="png")
    msg = f"Unknown tree export style: {style!r}"
    raise ValueError(msg)


def _dag_exports(graph: ExperimentGraph[RLRunConfig], *, metric_key: str) -> tuple[bytes, bytes]:
    """Render the matplotlib spring-layout DAG as SVG and PNG bytes."""
    import matplotlib.pyplot as plt

    from jarl.app.lib.graph_ops import render_dag_figure

    figure = render_dag_figure(graph, metric_key=metric_key)
    svg = _savefig_bytes(figure, fmt="svg")
    png = _savefig_bytes(figure, fmt="png")
    plt.close(figure)
    return svg, png


def _dag_bytes(graph: ExperimentGraph[RLRunConfig], *, metric_key: str, fmt: TreeExportFormat) -> bytes:
    """Render the matplotlib spring-layout DAG as image bytes."""
    import matplotlib.pyplot as plt

    from jarl.app.lib.graph_ops import render_dag_figure

    figure = render_dag_figure(graph, metric_key=metric_key)
    payload = _savefig_bytes(figure, fmt=fmt)
    plt.close(figure)
    return payload


def _savefig_bytes(figure: object, *, fmt: TreeExportFormat) -> bytes:
    """Serialize a matplotlib figure to SVG or PNG bytes."""
    from matplotlib.figure import Figure

    if not isinstance(figure, Figure):
        msg = "DAG export requires a matplotlib figure."
        raise TypeError(msg)
    buffer = io.BytesIO()
    if fmt == "png":
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    else:
        figure.savefig(buffer, format="svg", bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()
