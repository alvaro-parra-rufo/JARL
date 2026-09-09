"""Home page for the Runner Lab."""

from __future__ import annotations

import json

import streamlit as st

from jarl.app.lib.checkpoints_view import checkpoint_table, list_checkpoint_rows
from jarl.app.lib.inspect_view import load_run_metadata, run_metadata_table
from jarl.app.lib.inspection_panel import render_node_reward_panel
from jarl.app.lib.layout import load_experiment_graph, render_context_bar, require_experiment_dir
from jarl.app.lib.session import current_workspace, init_session_state
from jarl.app.lib.sidebar import render_sidebar
from jarl.app.lib.summaries import node_table
from jarl.app.lib.ui import (
    configure_page,
    empty_state,
    format_float,
    key_value,
    page_header,
    path_block,
    render_status_badge,
)
from jarl.experiments.summaries import config_highlights, load_experiment_summary
from jarl.showcase.summary import load_showcase_summary

configure_page("Inicio")
init_session_state({})
render_sidebar()

page_header("Inicio", "Estado actual del experimento, linaje, config resuelta y artefactos del nodo.")

exp_dir = require_experiment_dir()
if exp_dir is None:
    st.stop()

try:
    graph = load_experiment_graph(exp_dir)
    workspace = current_workspace(exp_dir)
    summary = load_experiment_summary(exp_dir)
except Exception as exc:
    st.error(f"No se pudo cargar el nodo actual: {exc}")
    st.stop()

render_context_bar(workspace)

st.subheader("Experimento")
path_block(exp_dir)

showcase_summary = load_showcase_summary(exp_dir)
if showcase_summary is not None:
    with st.expander("Showcase pipeline", expanded=True):
        plan = showcase_summary.get("plan", {})
        st.caption(
            f"Escenario {plan.get('scenario', '?')} · escala {plan.get('scale', '?')} · "
            f"W&B {'online' if plan.get('wandb_online') else 'offline'}"
        )
        showcase_nodes = showcase_summary.get("nodes", [])
        if showcase_nodes:
            st.dataframe(
                [
                    {
                        "logical": item.get("logical_name"),
                        "node": item.get("node_id"),
                        "status": item.get("status"),
                        "env": item.get("env_id"),
                    }
                    for item in showcase_nodes
                ],
                use_container_width=True,
                hide_index=True,
            )
        tb_lineage = showcase_summary.get("tensorboard_lineage")
        if tb_lineage:
            st.code(f"tensorboard --logdir_spec {tb_lineage}", language="bash")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Nodos", summary.node_count)
col2.metric("Ramas", summary.branch_count)
col3.metric("Train return", format_float(summary.nodes[-1].latest_return if summary.nodes else None))
col4.metric("Eval return", format_float(summary.nodes[-1].latest_eval_return if summary.nodes else None))

active_left, active_right = st.columns((1, 1.35), gap="large")
with active_left:
    st.subheader("Nodo activo")
    render_status_badge(workspace.status.value)
    key_value(
        {
            "node": workspace.id,
            "branch": workspace.branch,
            "label": workspace.node_metadata.label,
            "parent": workspace.node_metadata.parent_id,
            "step": workspace.node_metadata.step,
            "updated": workspace.node_metadata.updated_at,
        }
    )

with active_right:
    st.subheader("Config clave")
    config = graph.resolve_config(workspace)
    key_value(config_highlights(config))

tabs = st.tabs(("Nodos", "Checkpoints", "Linaje", "Metadata", "Recompensa", "Config", "Logs"))
with tabs[0]:
    st.dataframe(node_table(summary), use_container_width=True, hide_index=True)

with tabs[1]:
    rows = list_checkpoint_rows(workspace)
    if rows:
        st.dataframe(checkpoint_table(rows), use_container_width=True, hide_index=True)
    else:
        empty_state("Sin checkpoints", "Entrena el nodo o haz fork desde un checkpoint del padre.")

with tabs[2]:
    lineage = graph.get_lineage(workspace)
    lineage_rows = [
        {
            "node": item.id,
            "status": item.status.value,
            "branch": item.branch,
            "label": item.node_metadata.label,
            "parent_checkpoint": item.parent_checkpoint_step,
        }
        for item in lineage
    ]
    st.dataframe(lineage_rows, use_container_width=True, hide_index=True)
    st.caption(f"Branch heads: {summary.branch_heads}")

with tabs[3]:
    if load_run_metadata(exp_dir) is None:
        empty_state("Sin run_metadata.json", "Se escribe al crear el root.")
    else:
        st.dataframe(run_metadata_table(exp_dir), use_container_width=True, hide_index=True)

with tabs[4]:
    render_node_reward_panel(graph, workspace)

with tabs[5]:
    if workspace.config_path.is_file():
        st.json(json.loads(workspace.config_path.read_text(encoding="utf-8")), expanded=False)
    else:
        empty_state("Sin config resuelta", "El nodo aún no tiene config.json.")

with tabs[6]:
    log_tail = workspace.log_path.read_text(encoding="utf-8", errors="replace") if workspace.log_path.is_file() else ""
    runner_log = exp_dir / "runner_lab.log"
    runner_tail = runner_log.read_text(encoding="utf-8", errors="replace") if runner_log.is_file() else ""
    if log_tail:
        st.code("\n".join(log_tail.splitlines()[-80:]), language="text")
    if runner_tail:
        st.code("\n".join(runner_tail.splitlines()[-80:]), language="text")
    if not log_tail and not runner_tail:
        empty_state("Sin logs", "Aún no hay train.log ni runner_lab.log.")

if workspace.wandb_dir.is_dir():
    st.success(f"W&B local: {workspace.wandb_dir}")
