"""Shared layout helpers for Runner Lab navigation pages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from jarl.app.lib.session import experiment_dir, list_node_summaries, load_graph, workspace_root
from jarl.app.lib.ui import empty_state, format_float, render_status_badge
from jarl.experiments.io.metrics import JsonlMetricReader

if TYPE_CHECKING:
    from jarl.experiments.graph import ExperimentGraph
    from jarl.experiments.node import NodeWorkspace
    from jarl.training.config import RLRunConfig

APP_PAGES = {
    "inicio": "views/1_Inicio.py",
    "entrenar": "views/3_Entrenar.py",
    "metricas": "views/4_Metricas.py",
    "arbol": "views/5_Arbol.py",
    "inferencias": "views/8_Inferencias.py",
    "testing": "views/9_Testing.py",
}

__all__ = [
    "APP_PAGES",
    "format_node_label",
    "latest_scalar_metrics",
    "load_experiment_graph",
    "node_status_map",
    "render_context_bar",
    "render_module_tab_selector",
    "require_experiment_dir",
    "select_node_workspace",
]


def format_node_label(node_id: str, status: str) -> str:
    """Format a node id and lifecycle status for select boxes."""
    return f"{node_id} · {status}"


def render_module_tab_selector(
    options: tuple[str, ...],
    *,
    session_key: str,
    default: str,
) -> str:
    """Render a horizontal tab selector that runs one body per rerun.

    Unlike ``st.tabs``, only the selected section should render widgets on each
    rerun. That avoids conflicting writes to shared ``st.session_state`` keys.
    """
    if session_key not in st.session_state:
        st.session_state[session_key] = default
    selected = st.radio(
        "Vista",
        options=options,
        horizontal=True,
        key=session_key,
        label_visibility="collapsed",
    )
    return str(selected)


def node_status_map(
    exp_dir: Path,
    graph: ExperimentGraph[RLRunConfig] | None = None,
) -> dict[str, str]:
    """Return node id to status labels for an experiment directory."""
    return dict(list_node_summaries(exp_dir, graph=graph))


def require_experiment_dir(*, empty_title: str = "Sin experimento activo") -> Path | None:
    """Return the active experiment directory or render an empty state."""
    exp_dir = experiment_dir()
    if exp_dir is not None:
        return exp_dir
    empty_state(empty_title, f"Workspace actual: {workspace_root()}")
    from jarl.app.lib.navigation import nav_page_path

    st.page_link(nav_page_path("entrenar"), label="Crear experimento", icon=":material/add_circle:")
    return None


def load_experiment_graph(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    """Load the experiment graph for ``exp_dir``."""
    return load_graph(exp_dir)


def latest_scalar_metrics(workspace: NodeWorkspace) -> dict[str, float]:
    """Return latest scalar metrics for a node workspace."""
    if not workspace.metrics_jsonl_path.is_file():
        return {}
    return JsonlMetricReader(workspace.metrics_jsonl_path).latest()


def select_node_workspace(
    graph: ExperimentGraph[RLRunConfig],
    exp_dir: Path,
    *,
    session_key: str,
    label: str = "Nodo",
    default_node_id: str | None = None,
) -> NodeWorkspace:
    """Render a node picker bound to ``session_key`` and return the selected workspace."""
    summaries = list_node_summaries(exp_dir, graph=graph)
    node_ids = [node_id for node_id, _ in summaries]
    status_by_node = dict(summaries)
    default = default_node_id or graph.current_node.id
    if default not in node_ids and node_ids:
        default = node_ids[0]
    if session_key not in st.session_state and default in node_ids:
        st.session_state[session_key] = default
    selected = st.selectbox(
        label,
        options=node_ids,
        format_func=lambda node_id: format_node_label(node_id, status_by_node[node_id]),
        key=session_key,
    )
    return graph.get_node(selected)


def render_context_bar(
    workspace: NodeWorkspace,
    *,
    metrics: dict[str, float] | None = None,
) -> None:
    """Render a compact context strip for the active node."""
    scalars = metrics if metrics is not None else latest_scalar_metrics(workspace)
    train_return = scalars.get("rollout/episode_return")
    eval_return = scalars.get("eval/episode_return")
    global_step = scalars.get("global_step")

    st.markdown(
        """
        <style>
        .jarl-context-bar {
          border: 1px solid #d9dee7;
          border-radius: 8px;
          background: #ffffff;
          padding: 0.65rem 0.9rem;
          margin: 0.2rem 0 1rem;
        }
        .jarl-context-label {
          color: #647084;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: .06em;
          text-transform: uppercase;
        }
        .jarl-context-value {
          font-size: 0.95rem;
          font-weight: 600;
          margin-top: 0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    branch = workspace.branch
    parent = workspace.node_metadata.parent_id or "—"
    cols = st.columns((1.35, 0.9, 0.8, 0.9, 0.9, 0.9), gap="small")
    with cols[0]:
        st.markdown('<p class="jarl-context-label">Nodo activo</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="jarl-context-value"><code>{workspace.id}</code></p>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<p class="jarl-context-label">Estado</p>', unsafe_allow_html=True)
        render_status_badge(workspace.status.value)
    with cols[2]:
        st.markdown('<p class="jarl-context-label">Rama</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="jarl-context-value">{branch}</p>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<p class="jarl-context-label">Padre</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="jarl-context-value">{parent}</p>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown('<p class="jarl-context-label">Train return</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="jarl-context-value">{format_float(train_return)}</p>', unsafe_allow_html=True)
    with cols[5]:
        st.markdown('<p class="jarl-context-label">Eval return</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="jarl-context-value">{format_float(eval_return)}</p>', unsafe_allow_html=True)

    if global_step is not None:
        st.caption(f"Último global_step en metrics.jsonl: {format_float(global_step)}")
