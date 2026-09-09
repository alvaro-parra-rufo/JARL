"""Metrics module: explorer, multi-node compare, and inspection."""

from __future__ import annotations

import streamlit as st

from jarl.app.lib.layout import (
    load_experiment_graph,
    render_context_bar,
    render_module_tab_selector,
    require_experiment_dir,
    select_node_workspace,
)
from jarl.app.lib.metrics_page import (
    METRICAS_APPLIED_PRESET_KEY,
    METRICAS_NODE_CONTEXT_KEY,
    METRICAS_PREFIXES_KEY,
    METRICAS_PRESET_KEY,
    METRICAS_SELECTED_KEY,
    cached_metrics_snapshot,
    render_metrics_compare,
    render_metrics_explorer,
    render_metrics_inspection,
    render_metrics_lineage_concat,
)
from jarl.app.lib.metrics_view import DEFAULT_METRIC_PRESET, metrics_file_cache_key
from jarl.app.lib.session import init_session_state
from jarl.app.lib.sidebar import render_sidebar
from jarl.app.lib.ui import configure_page, page_header

METRICAS_TAB_KEY = "metricas_module_tab"
METRICAS_TABS = ("Explorador", "Comparar", "Linaje continuo", "Inspección")

configure_page("Métricas")
init_session_state(
    {
        METRICAS_APPLIED_PRESET_KEY: DEFAULT_METRIC_PRESET,
        METRICAS_PRESET_KEY: DEFAULT_METRIC_PRESET,
        METRICAS_TAB_KEY: "Explorador",
    }
)
render_sidebar()

page_header(
    "Métricas",
    "Explora metrics.jsonl, compara nodos, concatena extends y revisa config, linaje y TensorBoard.",
)
exp_dir = require_experiment_dir()
if exp_dir is None:
    st.stop()

graph = load_experiment_graph(exp_dir)

active_tab = render_module_tab_selector(
    METRICAS_TABS,
    session_key=METRICAS_TAB_KEY,
    default="Explorador",
)

if active_tab == "Explorador":
    workspace = select_node_workspace(graph, exp_dir, session_key="metricas_node")
    metrics_path = workspace.metrics_jsonl_path
    snapshot = cached_metrics_snapshot(metrics_file_cache_key(metrics_path))

    if st.session_state.get(METRICAS_NODE_CONTEXT_KEY) != workspace.id:
        st.session_state[METRICAS_NODE_CONTEXT_KEY] = workspace.id
        st.session_state[METRICAS_APPLIED_PRESET_KEY] = DEFAULT_METRIC_PRESET
        st.session_state[METRICAS_PRESET_KEY] = DEFAULT_METRIC_PRESET
        st.session_state.pop(METRICAS_SELECTED_KEY, None)
        st.session_state.pop(METRICAS_PREFIXES_KEY, None)

    render_context_bar(workspace, metrics=snapshot.latest)
    render_metrics_explorer(workspace, snapshot, metrics_path=metrics_path)
elif active_tab == "Comparar":
    render_metrics_compare(graph, exp_dir, default_node_id=graph.current_node.id)
elif active_tab == "Linaje continuo":
    render_metrics_lineage_concat(graph, exp_dir, default_node_id=graph.current_node.id)
elif active_tab == "Inspección":
    render_metrics_inspection(graph, exp_dir)
