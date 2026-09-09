"""Page render helpers for the Runner Lab inference module."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from jarl.agents.ppo.inference.requests import (
    INFERENCE_LOG_NAME,
    InferenceLaunchRequest,
    default_inference_name_prefix,
    inference_env_options,
    inference_episodes_default,
    inference_horizon_default,
    inference_run_config_summary,
    list_inference_jobs,
    sanitize_name_prefix,
    videos_for_prefix,
)
from jarl.app.lib.checkpoints_view import (
    checkpoint_ref,
    format_checkpoint_option,
    list_checkpoint_rows,
)
from jarl.app.lib.inference_view import (
    inference_timing_metrics,
    video_metrics_for_prefix,
)
from jarl.app.lib.inspection_panel import render_node_reward_panel
from jarl.app.lib.layout import render_context_bar, select_node_workspace
from jarl.app.lib.live_ui import (
    DEFAULT_LOG_REFRESH_SECONDS,
    INFERENCE_FINISH_NOTIFIED_KEY,
    render_inference_status_fragment,
    render_live_terminal_fragment,
)
from jarl.app.lib.runner_bridge import launch_inference_video
from jarl.app.lib.session import active_inference_run
from jarl.app.lib.ui import empty_state, format_float
from jarl.envs.navix.catalog import get_contract
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig

LAST_INFERENCE_JOB_KEY = "jarl_last_inference_job"
SELECTED_INFERENCE_PREFIX_KEY = "inferencias_saved_job"
INFERENCE_HORIZON_KEY = "inferencias_max_episode_steps"
INFERENCE_HORIZON_NODE_KEY = "inferencias_horizon_node_id"
INFERENCE_HORIZON_WIDGET_KEY = "_inferencias_horizon_widget_key"
NODE_HORIZON_DEFAULT_KEY = "_inferencias_node_horizon_default"
INFERENCE_EPISODES_KEY = "inferencias_final_video_episodes"
INFERENCE_EPISODES_NODE_KEY = "inferencias_episodes_node_id"
INFERENCE_ENV_KEY = "inferencias_env_id"
INFERENCE_ENV_NODE_KEY = "inferencias_env_node_id"

__all__ = ["LAST_INFERENCE_JOB_KEY", "render_inference_page"]


def _reset_horizon_from_node() -> None:
    """Restore the horizon widget from the active node config (``on_click`` callback)."""
    widget_key = str(st.session_state[INFERENCE_HORIZON_WIDGET_KEY])
    st.session_state[widget_key] = st.session_state[NODE_HORIZON_DEFAULT_KEY]


def _horizon_widget_key(workspace_id: str) -> str:
    return f"{INFERENCE_HORIZON_KEY}__{workspace_id}"


def _episodes_widget_key(workspace_id: str) -> str:
    return f"{INFERENCE_EPISODES_KEY}__{workspace_id}"


def _prepare_inference_horizon_widget(*, workspace_id: str, node_horizon: int) -> str:
    """Seed horizon session state before the ``number_input`` widget is created."""
    widget_key = _horizon_widget_key(workspace_id)
    st.session_state[NODE_HORIZON_DEFAULT_KEY] = node_horizon
    st.session_state[INFERENCE_HORIZON_WIDGET_KEY] = widget_key
    if st.session_state.get(INFERENCE_HORIZON_NODE_KEY) != workspace_id:
        st.session_state[INFERENCE_HORIZON_NODE_KEY] = workspace_id
        st.session_state[widget_key] = node_horizon
    elif widget_key not in st.session_state:
        st.session_state[widget_key] = node_horizon
    return widget_key


def _prepare_inference_episodes_widget(*, workspace_id: str) -> str:
    """Seed episode count session state before the ``number_input`` widget is created."""
    widget_key = _episodes_widget_key(workspace_id)
    default_episodes = inference_episodes_default()
    if st.session_state.get(INFERENCE_EPISODES_NODE_KEY) != workspace_id:
        st.session_state[INFERENCE_EPISODES_NODE_KEY] = workspace_id
        st.session_state[widget_key] = default_episodes
    elif widget_key not in st.session_state:
        st.session_state[widget_key] = default_episodes
    return widget_key


def _prepare_inference_env_widget(*, workspace_id: str, node_env_id: str) -> tuple[str, ...]:
    """Seed env selection before the ``selectbox`` widget is created."""
    if st.session_state.get(INFERENCE_ENV_NODE_KEY) != workspace_id:
        st.session_state[INFERENCE_ENV_NODE_KEY] = workspace_id
        st.session_state[INFERENCE_ENV_KEY] = node_env_id
    elif INFERENCE_ENV_KEY not in st.session_state:
        st.session_state[INFERENCE_ENV_KEY] = node_env_id
    return inference_env_options(node_env_id=node_env_id)


def render_inference_page(
    graph: ExperimentGraph[RLRunConfig],
    exp_dir: Path,
) -> None:
    """Render the inference launch form, live log, and result panel."""
    workspace = select_node_workspace(graph, exp_dir, session_key="inferencias_node", label="Nodo fuente")
    render_context_bar(workspace)
    node_config = graph.resolve_config(workspace)
    node_horizon = inference_horizon_default(node_config)

    checkpoint_rows = list_checkpoint_rows(workspace)
    if not checkpoint_rows:
        empty_state(
            "Sin checkpoints",
            "Entrena el nodo o haz fork desde un checkpoint del padre antes de inferir.",
        )
        return

    selected_row = st.selectbox(
        "Checkpoint",
        options=checkpoint_rows,
        format_func=format_checkpoint_option,
        key="inferencias_checkpoint_row",
    )
    checkpoint = checkpoint_ref(workspace, selected_row.checkpoint_step)
    default_prefix = default_inference_name_prefix(checkpoint.checkpoint_step)

    controls = st.columns(2)
    node_env_id = node_config.environment.env_id
    env_options = list(_prepare_inference_env_widget(workspace_id=workspace.id, node_env_id=node_env_id))
    current_env = str(st.session_state.get(INFERENCE_ENV_KEY, node_env_id))
    if current_env not in env_options:
        env_options = [current_env, *env_options]
    with controls[0]:
        env_id = st.selectbox(
            "Mapa Navix",
            options=env_options,
            index=env_options.index(current_env),
            key=INFERENCE_ENV_KEY,
            help="Por defecto el mapa del nodo; puedes evaluar en otro entorno del catálogo.",
        )
        try:
            contract = get_contract(env_id)
            st.caption(f"transfer_group: `{contract.transfer_group}`")
        except KeyError:
            st.warning("Mapa no registrado en el catálogo Navix.")
    with controls[1]:
        seed = st.number_input(
            "seed",
            min_value=0,
            value=int(node_config.environment.seed),
            key="inferencias_seed",
        )

    name_prefix = sanitize_name_prefix(
        st.text_input("Prefijo del vídeo", value=default_prefix, key="inferencias_name_prefix")
    )

    horizon_widget_key = _prepare_inference_horizon_widget(
        workspace_id=workspace.id,
        node_horizon=node_horizon,
    )
    episodes_widget_key = _prepare_inference_episodes_widget(workspace_id=workspace.id)
    rollout_col, episodes_col, reset_col = st.columns([3, 2, 1])
    with rollout_col:
        horizon_input = int(
            st.number_input(
                "Horizon (máx. pasos/episodio)",
                min_value=0,
                step=1,
                help="0 = sin override (nodo o default Navix). Por defecto 100.",
                key=horizon_widget_key,
            )
        )
    with episodes_col:
        episodes_input = int(
            st.number_input(
                "Episodios",
                min_value=1,
                step=1,
                help="Número de episodios greedy a grabar en el vídeo.",
                key=episodes_widget_key,
            )
        )
    with reset_col:
        st.write("")
        st.button(
            "Del nodo",
            help="Restaurar horizon del nodo (o 100 si el nodo no lo fija)",
            key="inferencias_horizon_reset",
            on_click=_reset_horizon_from_node,
        )

    horizon_override = horizon_input if horizon_input > 0 else None
    effective_config = node_config.apply_overrides({"environment.env_id": env_id, "environment.seed": int(seed)})
    panel_horizon = horizon_input if horizon_input > 0 else inference_horizon_default(node_config)
    _render_node_config_panel(
        graph,
        workspace,
        effective_config,
        episode_horizon=panel_horizon,
        episode_count=episodes_input,
    )

    horizon_label = str(horizon_input) if horizon_input > 0 else f"default ({inference_horizon_default(node_config)})"
    st.caption(
        f"Checkpoint step **{checkpoint.checkpoint_step}** · "
        f"horizon **{horizon_label}** · "
        f"**{episodes_input}** episodios a "
        f"**{int(effective_config.video.video_fps)}** FPS."
    )

    inference_active = active_inference_run() is not None
    if st.button(
        "Lanzar inferencia",
        type="primary",
        icon=":material/play_circle:",
        disabled=inference_active or not name_prefix,
        key="inferencias_launch",
    ):
        request = InferenceLaunchRequest(
            node_id=workspace.id,
            checkpoint_step=checkpoint.checkpoint_step,
            env_id=env_id,
            seed=int(seed),
            name_prefix=name_prefix,
            max_episode_steps=horizon_override,
            final_video_episodes=episodes_input,
        )
        launch_inference_video(experiment_dir=exp_dir, request=request)
        st.session_state[INFERENCE_FINISH_NOTIFIED_KEY] = False
        st.session_state[LAST_INFERENCE_JOB_KEY] = asdict(request)
        st.session_state[_saved_job_key(workspace.id)] = name_prefix
        st.success("Inferencia lanzada en segundo plano.")
        st.rerun()

    st.subheader("Monitor en vivo")
    render_inference_status_fragment()
    render_live_terminal_fragment(
        exp_dir / INFERENCE_LOG_NAME,
        buffer_key="jarl_terminal_inference",
        empty_message="Esperando inference_runner_lab.log…",
    )

    st.subheader("Resultado")
    selected_job = _select_saved_inference_job(workspace)
    _render_inference_result_fragment(workspace, selected_job)


def _render_node_config_panel(
    graph: ExperimentGraph[RLRunConfig],
    workspace: NodeWorkspace,
    config: RLRunConfig,
    *,
    episode_horizon: int,
    episode_count: int,
) -> None:
    """Show resolved node settings that drive greedy video rendering."""
    with st.expander("Config del nodo (inferencia)", expanded=True):
        summary = inference_run_config_summary(
            config,
            episode_horizon=episode_horizon,
            episode_count=episode_count,
        )
        cols = st.columns(4)
        cols[0].metric("Algoritmo", str(summary["algorithm"]))
        cols[1].metric("Horizon", int(summary["episode_horizon"]))
        cols[2].metric("Episodios", int(summary["final_video_episodes"]))
        cols[3].metric("FPS", int(summary["video_fps"]))
        st.caption(
            f"env **{summary['env_id']}** · seed **{summary['seed']}** · "
            f"escala **{summary['video_scale']}** · vista **{summary['video_view_mode']}**"
        )
        with st.popover("JSON completo"):
            st.json(config.model_dump(), expanded=False)
    with st.expander("Recompensa del nodo", expanded=False):
        render_node_reward_panel(graph, workspace)


def _render_inference_result_fragment(workspace: NodeWorkspace, job: dict[str, Any] | None) -> None:
    """Poll disk for MP4/metrics so results appear when the subprocess finishes."""
    result_slot = st.empty()
    _draw_inference_result(workspace, result_slot, job)

    @st.fragment(run_every=DEFAULT_LOG_REFRESH_SECONDS)
    def _poll_result() -> None:
        _draw_inference_result(workspace, result_slot, job)

    _poll_result()


def _draw_inference_result(
    workspace: NodeWorkspace,
    slot: st.delta_generator.DeltaGenerator,
    job: dict[str, Any] | None,
) -> None:
    try:
        resolved = _resolve_result_job(workspace, job)
    except Exception as exc:
        slot.error(f"No se pudo cargar el resultado de inferencia: {exc}")
        return
    with slot.container():
        if resolved is None:
            st.caption("Lanza una inferencia para ver el vídeo, tiempos y returns aquí.")
            return
        _render_inference_result(workspace, resolved)


def _saved_job_key(workspace_id: str) -> str:
    """Return the Streamlit session key for the saved-job selector of one node."""
    return f"{SELECTED_INFERENCE_PREFIX_KEY}__{workspace_id}"


def _format_inference_job(job: dict[str, Any]) -> str:
    """Return a compact label for a saved inference job."""
    parts = [str(job["name_prefix"])]
    checkpoint_step = job.get("checkpoint_step")
    if checkpoint_step is not None:
        parts.append(f"ckpt {checkpoint_step}")
    env_id = job.get("env_id")
    if isinstance(env_id, str) and env_id:
        parts.append(env_id)
    return " · ".join(parts)


def _select_saved_inference_job(workspace: NodeWorkspace) -> dict[str, Any] | None:
    """Render a selector of disk-backed inference jobs for the current node."""
    jobs = list_inference_jobs(workspace)
    active = active_inference_run()
    if active is not None and active.node_id == workspace.id:
        st.caption(f"Job en curso: `{active.name_prefix}`")
        return {
            "node_id": active.node_id,
            "name_prefix": active.name_prefix,
            "checkpoint_step": active.checkpoint_step,
        }
    if not jobs:
        return None
    prefixes = [str(job["name_prefix"]) for job in jobs]
    labels = {str(job["name_prefix"]): _format_inference_job(job) for job in jobs}
    widget_key = _saved_job_key(workspace.id)
    current = st.session_state.get(widget_key)
    if current not in prefixes:
        st.session_state[widget_key] = prefixes[0]
    selected_prefix = str(
        st.selectbox(
            "Inferencias guardadas",
            options=prefixes,
            format_func=lambda prefix: labels.get(str(prefix), str(prefix)),
            key=widget_key,
            help="Vídeos greedy ya escritos en este nodo. Elige uno para volver a verlo.",
        )
    )
    for job in jobs:
        if str(job["name_prefix"]) == selected_prefix:
            return job
    return jobs[0]


def _resolve_result_job(workspace: NodeWorkspace, selected: dict[str, Any] | None) -> dict[str, Any] | None:
    """Prefer the in-flight job, then the selector, then the last session job or disk."""
    active = active_inference_run()
    if active is not None and active.node_id == workspace.id:
        return {
            "node_id": active.node_id,
            "name_prefix": active.name_prefix,
            "checkpoint_step": active.checkpoint_step,
        }
    if selected is not None:
        return selected
    last_job = st.session_state.get(LAST_INFERENCE_JOB_KEY)
    if isinstance(last_job, dict) and last_job.get("node_id") == workspace.id:
        return last_job
    jobs = list_inference_jobs(workspace)
    return jobs[0] if jobs else None


def _render_inference_result(workspace: NodeWorkspace, job: dict[str, Any]) -> None:  # noqa: PLR0912
    name_prefix = str(job["name_prefix"])
    checkpoint_step = job.get("checkpoint_step")
    max_episode_steps = job.get("max_episode_steps")
    final_video_episodes = job.get("final_video_episodes")
    caption_parts = [f"Job **{name_prefix}**"]
    if checkpoint_step is not None:
        caption_parts.append(f"checkpoint step **{checkpoint_step}**")
    if max_episode_steps is not None:
        caption_parts.append(f"horizon **{max_episode_steps}**")
    if final_video_episodes is not None:
        caption_parts.append(f"**{final_video_episodes}** episodios")
    st.caption(" · ".join(caption_parts))

    metrics_df = video_metrics_for_prefix(workspace, name_prefix)
    videos = videos_for_prefix(workspace, name_prefix)
    failed_rows = (
        metrics_df[metrics_df["status"] == "failed"] if "status" in metrics_df.columns else metrics_df.iloc[0:0]
    )

    if metrics_df.empty and not videos:
        if active_inference_run() is not None:
            st.info("Inferencia en cursor… el vídeo y las métricas aparecerán al terminar.")
        else:
            st.warning(
                "Sin vídeos ni métricas para este job. Revisa el monitor en vivo "
                "(dispositivo JAX, moviepy, checkpoint)."
            )
        return

    timing = inference_timing_metrics(metrics_df)
    if timing:
        cols = st.columns(len(timing))
        labels = {
            "rollout_s": "Rollout (s)",
            "transfer_s": "Transfer (s)",
            "encode_s": "Encode (s)",
            "total_s": "Total (s)",
        }
        for column, (key, value) in zip(cols, timing.items(), strict=True):
            column.metric(labels.get(key, key), format_float(value))

    if not failed_rows.empty:
        for _, row in failed_rows.iterrows():
            reason = row.get("reason")
            st.error(f"Episodio fallido: {reason or 'motivo desconocido'}")

    if not metrics_df.empty:
        display_cols = [
            column
            for column in (
                "status",
                "episode_index",
                "episode_return",
                "episode_length",
                "rollout_seconds",
                "transfer_seconds",
                "encode_seconds",
                "video_file",
                "role",
                "checkpoint_step",
                "env_id",
                "reason",
            )
            if column in metrics_df.columns
        ]
        st.dataframe(metrics_df[display_cols], use_container_width=True, hide_index=True)

    if not videos:
        completed = metrics_df[metrics_df["status"] == "completed"] if "status" in metrics_df.columns else metrics_df
        if completed.empty and not failed_rows.empty:
            return
        st.caption("Métricas registradas; esperando archivos MP4.")
        return

    for video_path in videos:
        st.video(str(video_path))
