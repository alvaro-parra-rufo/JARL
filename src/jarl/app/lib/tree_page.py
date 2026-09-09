"""Runner Lab tree module page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from jarl.app.components.tree_explorer import tree_explorer
from jarl.app.lib.graph_feed import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_VISIBLE,
    MAX_TREE_DEPTH,
    build_visible_subtree,
    combined_graph_revision,
)
from jarl.app.lib.layout import format_node_label, load_experiment_graph, render_context_bar
from jarl.app.lib.navigation import nav_page_path
from jarl.app.lib.session import (
    active_run,
    checkout_experiment_node,
    graph_revision,
    list_node_summaries,
    load_graph,
    poll_active_run,
)
from jarl.app.lib.summaries import node_table
from jarl.app.lib.tree_explorer import (
    ACTIVE_RUN_TREE_REFRESH_SECONDS,
    DEFAULT_TREE_REFRESH_SECONDS,
    TRAINING_TAB_SESSION_KEY,
    TreeExportStyle,
    build_tree_exports,
    tree_explorer_session_token,
)
from jarl.app.lib.ui import key_value, path_block, render_status_badge
from jarl.experiments.feed import (
    experiment_tree_root_id,
    node_ids_within_depth,
    payload_to_json_dict,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.summaries import load_experiment_summary
from jarl.training.config import RLRunConfig

__all__ = ["render_tree_page"]


def render_tree_page(exp_dir: Path) -> None:
    """Render the dedicated experiment tree explorer module."""
    exp_path = exp_dir
    graph = load_experiment_graph(exp_path)
    summary = load_experiment_summary(exp_path)
    path_block(exp_path)
    render_context_bar(graph.current_node)

    token = tree_explorer_session_token(exp_path)
    metric_key = st.selectbox(
        "Métrica para colorear nodos",
        options=("rollout/episode_return", "eval/episode_return", "loss/policy_gradient_loss"),
        key="arbol_metric_key",
    )

    control_left, control_right = st.columns(2)
    with control_left:
        max_depth = st.slider(
            "Profundidad máxima",
            min_value=1,
            max_value=MAX_TREE_DEPTH,
            value=int(st.session_state.get(f"jarl_tree_depth__{token}", DEFAULT_MAX_DEPTH)),
            key=f"jarl_tree_depth__{token}",
        )
    with control_right:
        max_visible = st.number_input(
            "Límite de nodos visibles",
            min_value=8,
            max_value=1024,
            value=int(st.session_state.get(f"jarl_tree_max_visible__{token}", DEFAULT_MAX_VISIBLE)),
            step=8,
            key=f"jarl_tree_max_visible__{token}",
        )
    search_filter = st.text_input(
        "Incluir nodos por id",
        key=f"jarl_tree_search__{token}",
        placeholder="Substring del id; fuerza ancestros visibles",
    )

    refresh_seconds = ACTIVE_RUN_TREE_REFRESH_SECONDS if active_run() is not None else DEFAULT_TREE_REFRESH_SECONDS
    _render_live_tree(
        exp_path,
        metric_key=metric_key,
        max_depth=int(max_depth),
        max_visible=int(max_visible),
        search_filter=str(search_filter),
        refresh_seconds=refresh_seconds,
        token=token,
    )

    st.divider()
    _render_node_actions(exp_path, graph=load_graph(exp_path))
    st.divider()

    export_style: TreeExportStyle = st.radio(
        "Formato de exportación",
        options=("vertical", "horizontal", "dag"),
        format_func=lambda value: {
            "vertical": "Árbol vertical (como la app)",
            "horizontal": "Árbol horizontal (como la app)",
            "dag": "Grafo spring (matplotlib)",
        }[value],
        horizontal=True,
        key=f"jarl_tree_export_style__{token}",
    )
    try:
        svg_bytes, png_bytes = build_tree_exports(exp_path, metric_key=metric_key, style=export_style)
        stem = {
            "vertical": "experiment_tree",
            "horizontal": "experiment_tree_horizontal",
            "dag": "experiment_tree_dag",
        }[export_style]
        col_svg, col_png = st.columns(2)
        with col_svg:
            st.download_button(
                "Exportar SVG",
                data=svg_bytes,
                file_name=f"{stem}.svg",
                mime="image/svg+xml",
                icon=":material/download:",
                key=f"jarl_tree_export_svg__{token}",
            )
        with col_png:
            st.download_button(
                "Exportar PNG",
                data=png_bytes,
                file_name=f"{stem}.png",
                mime="image/png",
                icon=":material/image:",
                key=f"jarl_tree_export_png__{token}",
            )
        st.caption(
            "El árbol git coincide con el explorador (fork vs extend, HEAD/ROOT). "
            "El grafo spring es el layout matplotlib anterior."
        )
    except Exception as exc:
        st.caption(f"Exportación de imagen no disponible: {exc}")

    with st.expander("Tabla de nodos", expanded=False):
        st.dataframe(node_table(summary), use_container_width=True, hide_index=True)


def _handle_tree_event(exp_dir: Path, event: dict[str, Any] | None, *, token: str) -> None:
    """Apply tree component events such as checkout and navigation."""
    if not event:
        return

    action = str(event.get("action", ""))
    node_id = str(event.get("node_id", ""))
    if not node_id:
        return

    dedupe_key = f"jarl_tree_last_event__{token}"
    event_signature = (action, node_id)
    if st.session_state.get(dedupe_key) == event_signature:
        return
    st.session_state[dedupe_key] = event_signature

    if action == "checkout":
        checkout_experiment_node(exp_dir, node_id)
        st.toast(f"Checkout: {node_id}", icon="✅")
        st.rerun(scope="app")
        return

    if action == "goto_launch":
        checkout_experiment_node(exp_dir, node_id)
        st.session_state[TRAINING_TAB_SESSION_KEY] = "Lanzar"
        st.switch_page(nav_page_path("entrenar"))
        return

    if action == "goto_continue":
        checkout_experiment_node(exp_dir, node_id)
        st.session_state[TRAINING_TAB_SESSION_KEY] = "Continuar"
        st.switch_page(nav_page_path("entrenar"))


def _render_live_tree(
    exp_dir: Path,
    *,
    metric_key: str,
    max_depth: int,
    max_visible: int,
    search_filter: str,
    refresh_seconds: float,
    token: str,
) -> None:
    cache_key = f"jarl_tree_payload_cache__{token}"
    cache_revision_key = f"jarl_tree_payload_revision__{token}"

    def _refresh_tree_payload() -> str:
        poll_active_run()
        sess_revision = graph_revision()
        return (
            f"{combined_graph_revision(exp_dir, sess_revision)}:{metric_key}:{max_depth}:{max_visible}:{search_filter}"
        )

    def _build_tree_payload(view_revision: str) -> dict[str, Any]:
        graph = load_graph(exp_dir)
        root_id = experiment_tree_root_id(graph)
        expanded_ids = node_ids_within_depth(
            graph,
            root_id=root_id,
            max_depth=max_depth,
        )
        payload = build_visible_subtree(
            graph,
            exp_dir=exp_dir,
            metric_key=metric_key,
            expanded_ids=expanded_ids,
            max_depth=max_depth,
            max_visible=max_visible,
            search_filter=search_filter,
            session_revision=graph_revision(),
        )
        payload_dict = payload_to_json_dict(payload)
        st.session_state[cache_revision_key] = view_revision
        st.session_state[cache_key] = payload_dict
        return payload_dict

    @st.fragment(run_every=refresh_seconds)
    def _poll_tree() -> None:
        view_revision = _refresh_tree_payload()
        if st.session_state.get(cache_revision_key) != view_revision:
            _build_tree_payload(view_revision)
            st.rerun(scope="app")

    _poll_tree()

    view_revision = _refresh_tree_payload()
    if st.session_state.get(cache_revision_key) != view_revision:
        payload_dict = _build_tree_payload(view_revision)
    else:
        payload_dict = st.session_state.get(cache_key)

    if not payload_dict:
        st.warning("No hay datos de árbol para mostrar.")
        return

    if payload_dict.get("truncated"):
        st.caption(
            f"Subárbol truncado al límite de {payload_dict.get('max_visible')} nodos. "
            "Reduce profundidad o sube el límite."
        )

    event = tree_explorer(
        payload=payload_dict,
        key=f"jarl_tree_component__{token}",
        height=680,
    )
    _handle_tree_event(exp_dir, event, token=token)
    st.caption(
        f"Árbol live · rev `{payload_dict.get('revision', '?')}` · "
        f"actualización cada {refresh_seconds:g} s · clic = checkout"
    )


def _render_node_actions(exp_dir: Path, *, graph: ExperimentGraph[RLRunConfig]) -> None:
    summaries = list_node_summaries(exp_dir, graph=graph)
    node_ids = [node_id for node_id, _ in summaries]
    status_by_node = dict(summaries)
    if not node_ids:
        return

    st.subheader("Acciones sobre nodo")
    if "arbol_action_node" not in st.session_state and graph.current_node.id in node_ids:
        st.session_state["arbol_action_node"] = graph.current_node.id
    selected = st.selectbox(
        "Nodo",
        options=node_ids,
        format_func=lambda node_id: format_node_label(node_id, status_by_node[node_id]),
        key="arbol_action_node",
    )
    workspace = graph.get_node(selected)
    render_status_badge(workspace.status.value)
    key_value(
        {
            "branch": workspace.branch,
            "label": workspace.node_metadata.label,
            "parent": workspace.node_metadata.parent_id or "—",
        }
    )

    col_checkout, col_launch, col_continue = st.columns(3)
    with col_checkout:
        if st.button("Checkout", icon=":material/check_circle:", key="arbol_checkout"):
            checkout_experiment_node(exp_dir, selected)
            st.toast(f"Checkout: {selected}", icon="✅")
            st.rerun()
    with col_launch:
        if st.button("Ir a Lanzar", icon=":material/play_arrow:", key="arbol_goto_launch"):
            checkout_experiment_node(exp_dir, selected)
            st.session_state[TRAINING_TAB_SESSION_KEY] = "Lanzar"
            st.switch_page(nav_page_path("entrenar"))
    with col_continue:
        if st.button("Ir a Continuar", icon=":material/forward:", key="arbol_goto_continue"):
            checkout_experiment_node(exp_dir, selected)
            st.session_state[TRAINING_TAB_SESSION_KEY] = "Continuar"
            st.switch_page(nav_page_path("entrenar"))
