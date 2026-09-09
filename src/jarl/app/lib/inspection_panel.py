"""Reusable inspection panel for Runner Lab metrics module."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from jarl.agents.ppo.inference.requests import DEFAULT_INFERENCE_ROLE
from jarl.app.lib.graph_ops import pin_checkpoint, promote_checkpoint_best
from jarl.app.lib.inspect_view import (
    artifacts_table,
    branch_heads_table,
    config_diff_dataframe,
    config_overrides_table,
    execution_attempts_table,
    lineage_metrics_dataframe,
    list_video_files,
    load_run_metadata,
    reward_mix_table,
    run_metadata_table,
    scenario_overlay_table,
    video_metrics_by_role,
    video_metrics_table,
)
from jarl.app.lib.layout import render_context_bar
from jarl.app.lib.session import bump_graph_revision, list_node_summaries
from jarl.app.lib.ui import empty_state, path_block, render_parent_checkpoint_banner
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.experiments.tensorboard import launch_tensorboard, tensorboard_compare, tensorboard_lineage
from jarl.training.config import RLRunConfig

__all__ = ["render_inspection_panel", "render_node_reward_panel"]


def render_inspection_panel(  # noqa: PLR0912
    graph: ExperimentGraph[RLRunConfig],
    exp_dir: Path,
    *,
    include_videos: bool = True,
    exclude_inference_videos: bool = False,
    inspect_session_key: str = "inspection_node",
) -> None:
    """Render node inspection tools (diff, artifacts, lineage, TensorBoard).

    Args:
        graph: Loaded experiment graph.
        exp_dir: Active experiment directory.
        include_videos: When ``True``, include the Vídeos tab (train/showcase MP4s).
        exclude_inference_videos: When ``True``, hide ``infer_*`` files and rows with
            role ``inference`` (those belong in the Inferencias module).
        inspect_session_key: Streamlit key for the inspect-node select box.
    """
    path_block(exp_dir)

    metadata = load_run_metadata(exp_dir)
    if metadata is not None:
        st.subheader("Run metadata")
        st.dataframe(run_metadata_table(exp_dir), use_container_width=True, hide_index=True)

    st.subheader("Ramas")
    st.dataframe(branch_heads_table(graph), use_container_width=True, hide_index=True)

    summaries = list_node_summaries(exp_dir, graph=graph)
    node_ids = [node_id for node_id, _ in summaries]
    if not node_ids:
        empty_state("Sin nodos", "Crea un experimento con al menos un nodo.")
        return

    left, right = st.columns(2)
    with left:
        node_a = st.selectbox("Nodo A", options=node_ids, index=0, key=f"{inspect_session_key}_diff_a")
    with right:
        node_b = st.selectbox(
            "Nodo B",
            options=node_ids,
            index=min(1, len(node_ids) - 1),
            key=f"{inspect_session_key}_diff_b",
        )

    diff = graph.get_config_diff(node_a, node_b)
    diff_df = config_diff_dataframe(diff)
    if diff_df.empty:
        st.caption("Sin diferencias de config entre los nodos seleccionados.")
    else:
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

    default_index = node_ids.index(graph.current_node.id) if graph.current_node.id in node_ids else 0
    inspect_node = st.selectbox(
        "Inspeccionar nodo",
        options=node_ids,
        index=default_index,
        key=inspect_session_key,
    )
    workspace = graph.get_node(inspect_node)
    render_context_bar(workspace)
    render_parent_checkpoint_banner(workspace)

    tab_labels = ["Recompensa", "Overrides", "Attempts", "Artifacts", "Linaje métricas", "TensorBoard"]
    if include_videos:
        tab_labels.insert(-1, "Vídeos")

    tabs = st.tabs(tuple(tab_labels))
    tab_index = 0

    with tabs[tab_index]:
        render_node_reward_panel(graph, workspace)
    tab_index += 1

    with tabs[tab_index]:
        overrides_df = config_overrides_table(workspace)
        if overrides_df.empty:
            empty_state("Sin overrides", "El nodo no guardó config_overrides.json.")
        else:
            st.dataframe(overrides_df, use_container_width=True, hide_index=True)
    tab_index += 1

    with tabs[tab_index]:
        attempts_df = execution_attempts_table(workspace)
        if attempts_df.empty:
            empty_state("Sin intentos", "Aún no hay execution_attempts.json.")
        else:
            st.dataframe(attempts_df, use_container_width=True, hide_index=True)
    tab_index += 1

    with tabs[tab_index]:
        artifacts_df = artifacts_table(workspace)
        if artifacts_df.empty:
            empty_state("Sin artifacts", "No hay model archives ni exports registrados.")
        else:
            st.dataframe(artifacts_df, use_container_width=True, hide_index=True)
    tab_index += 1

    with tabs[tab_index]:
        lineage_df = lineage_metrics_dataframe(graph, inspect_node)
        if lineage_df.empty:
            empty_state("Sin métricas de linaje", "Entrena nodos del linaje para comparar.")
        else:
            st.dataframe(lineage_df, use_container_width=True, hide_index=True)
    tab_index += 1

    if include_videos:
        with tabs[tab_index]:
            _render_videos_panel(workspace, exclude_inference=exclude_inference_videos)
        tab_index += 1

    with tabs[tab_index]:
        lineage_spec = tensorboard_lineage(graph, inspect_node)
        compare_nodes = st.multiselect(
            "Comparar nodos",
            options=node_ids,
            default=[inspect_node],
            key=f"{inspect_session_key}_tb_compare",
        )
        compare_spec = tensorboard_compare(graph, compare_nodes) if compare_nodes else ""
        st.code(f"tensorboard --logdir_spec {lineage_spec}", language="bash")
        if compare_spec:
            st.code(f"tensorboard --logdir_spec {compare_spec}", language="bash")
        if st.button("Lanzar TensorBoard (lineage)", icon=":material/open_in_new:"):
            launch_tensorboard(logdir_spec=lineage_spec, port=6006)
            st.success("TensorBoard lanzado en http://localhost:6006")

    st.subheader("Checkpoints")
    ckpt_step = st.number_input("Checkpoint step", min_value=0, step=1, value=0, key=f"{inspect_session_key}_ckpt")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Promover best", icon=":material/military_tech:", key=f"{inspect_session_key}_best"):
            step = promote_checkpoint_best(exp_dir, node_id=inspect_node, graph=graph)
            bump_graph_revision()
            st.success(f"Best promovido: step {step}" if step is not None else "Sin candidato best.")
    with col2:
        if st.button(
            "Pinear checkpoint",
            icon=":material/push_pin:",
            disabled=ckpt_step <= 0,
            key=f"{inspect_session_key}_pin",
        ):
            pin_checkpoint(exp_dir, node_id=inspect_node, checkpoint_step=int(ckpt_step), graph=graph)
            bump_graph_revision()
            st.success(f"Checkpoint {ckpt_step} pineado.")


def render_node_reward_panel(graph: ExperimentGraph[RLRunConfig], workspace: NodeWorkspace) -> None:
    """Show the resolved reward mix and scenario overlay for ``workspace``."""
    mix_df = reward_mix_table(graph, workspace)
    overlay_df = scenario_overlay_table(graph.resolve_config(workspace))
    if mix_df.empty:
        empty_state(
            "Recompensa nativa",
            "Este nodo usa la recompensa nativa del mapa Navix (sin mix configurable).",
        )
    else:
        active = mix_df[mix_df["weight"] > 0]
        if active.empty:
            st.caption("Todos los canales del mix están a peso 0.")
        else:
            st.caption("Canales activos: " + ", ".join(f"{row.channel}={row.weight:g}" for row in active.itertuples()))
        st.dataframe(mix_df, use_container_width=True, hide_index=True)
    if overlay_df.empty:
        st.caption("Sin overlay de escenario.")
        return
    st.markdown("**Overlay de escenario**")
    st.dataframe(overlay_df, use_container_width=True, hide_index=True)


def _render_videos_panel(workspace: NodeWorkspace, *, exclude_inference: bool = False) -> None:
    videos = list_video_files(workspace)
    metrics_df = video_metrics_table(workspace)
    by_role = video_metrics_by_role(workspace)
    if exclude_inference:
        videos = [path for path in videos if not path.name.startswith("infer_")]
        if not metrics_df.empty and "role" in metrics_df.columns:
            metrics_df = metrics_df[metrics_df["role"] != DEFAULT_INFERENCE_ROLE]
        by_role = {role: role_df for role, role_df in by_role.items() if role != DEFAULT_INFERENCE_ROLE}
    if metrics_df.empty and not videos:
        empty_state("Sin vídeos", "Activa grabación de vídeo y entrena el nodo.")
        return
    if by_role:
        for role, role_df in sorted(by_role.items()):
            st.markdown(f"**Rol: {role}**")
            st.dataframe(role_df, use_container_width=True, hide_index=True)
    elif not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    for video_path in videos:
        st.video(str(video_path))
