"""Streamlit fragments for live Runner Lab feeds."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from jarl.app.lib.live_feed import (
    cursor_key_for_path,
    read_log_delta,
    reset_log_feed,
    update_terminal_buffer,
)
from jarl.app.lib.session import (
    active_agentic_case_run,
    active_inference_run,
    active_run,
    graph_revision,
    poll_active_agentic_case_run,
    poll_active_inference_run,
    poll_active_run,
)
from jarl.app.lib.ui import format_float, render_status_badge
from jarl.io.tail import read_metrics_latest

DEFAULT_LOG_REFRESH_SECONDS = 2.0
DEFAULT_METRICS_REFRESH_SECONDS = 5.0
DEFAULT_RUN_STATUS_REFRESH_SECONDS = 3.0

RUN_FINISH_NOTIFIED_KEY = "jarl_run_finish_notified"
LAST_RUN_EXIT_CODE_KEY = "jarl_last_run_exit_code"

AGENTIC_CASE_FINISH_NOTIFIED_KEY = "jarl_agentic_case_finish_notified"
LAST_AGENTIC_CASE_EXIT_CODE_KEY = "jarl_last_agentic_case_exit_code"
AGENTIC_LIVE_WATCH_KEY = "jarl_agentic_live_watch"
AGENTIC_LAUNCH_UNBLOCK_KEY = "jarl_agentic_launch_unblock"
AGENTIC_CONVERSATION_WAKE_KEY = "jarl_agentic_conversation_wake"
AGENTIC_RESULT_WATCH_KEY = "jarl_agentic_result_watch"
AGENTIC_CONVERSATION_REFRESH_NONCE_KEY = "jarl_conversation_refresh_nonce"

__all__ = [
    "AGENTIC_CASE_FINISH_NOTIFIED_KEY",
    "AGENTIC_CONVERSATION_REFRESH_NONCE_KEY",
    "AGENTIC_CONVERSATION_WAKE_KEY",
    "AGENTIC_LAUNCH_UNBLOCK_KEY",
    "AGENTIC_LIVE_WATCH_KEY",
    "AGENTIC_RESULT_WATCH_KEY",
    "DEFAULT_LOG_REFRESH_SECONDS",
    "DEFAULT_METRICS_REFRESH_SECONDS",
    "DEFAULT_RUN_STATUS_REFRESH_SECONDS",
    "LAST_AGENTIC_CASE_EXIT_CODE_KEY",
    "prepare_live_feeds_for_launch",
    "render_agentic_case_idle_status",
    "render_inference_status_fragment",
    "render_live_metrics_fragment",
    "render_live_terminal_fragment",
    "render_run_status_fragment",
    "render_sidebar_run_fragment",
    "reset_run_finish_state",
]


def reset_run_finish_state() -> None:
    """Clear run-finish notification flags before launching a new subprocess."""
    st.session_state[RUN_FINISH_NOTIFIED_KEY] = False
    st.session_state.pop(LAST_RUN_EXIT_CODE_KEY, None)


def prepare_live_feeds_for_launch(
    *,
    log_paths: list[Path],
    terminal_buffer_keys: list[str] | None = None,
) -> None:
    """Reset tail cursors and terminal buffers when a new subprocess starts."""
    reset_run_finish_state()
    buffer_keys = terminal_buffer_keys or []
    for index, path in enumerate(log_paths):
        buffer_key = buffer_keys[index] if index < len(buffer_keys) else f"jarl_terminal_{index}"
        reset_log_feed(path, buffer_key)


INFERENCE_FINISH_NOTIFIED_KEY = "jarl_inference_finish_notified"


def render_inference_status_fragment(*, refresh_seconds: float = DEFAULT_RUN_STATUS_REFRESH_SECONDS) -> None:
    """Poll inference subprocess state inside a fragment."""

    @st.fragment(run_every=refresh_seconds)
    def _poll_inference() -> None:
        exit_code = poll_active_inference_run()
        if exit_code is not None:
            if not st.session_state.get(INFERENCE_FINISH_NOTIFIED_KEY):
                st.session_state[INFERENCE_FINISH_NOTIFIED_KEY] = True
                if exit_code == 0:
                    st.toast("Inferencia completada", icon="✅")
                else:
                    st.toast(f"Inferencia terminó con código {exit_code}", icon="⚠️")
            if exit_code == 0:
                st.success("Inferencia completada")
            else:
                st.error(f"Inferencia terminó con código {exit_code}")
            return

        run = active_inference_run()
        if run is not None:
            render_status_badge("training")
            st.caption(f"{run.name_prefix} · ckpt {run.checkpoint_step}")
            st.caption(str(run.log_path))
            return

        st.caption("Sin inferencia activa")

    _poll_inference()


def render_live_terminal_fragment(
    path: Path,
    *,
    buffer_key: str,
    empty_message: str,
    refresh_seconds: float = DEFAULT_LOG_REFRESH_SECONDS,
    max_lines: int = 500,
) -> None:
    """Render a log file with incremental reads inside a Streamlit fragment."""

    @st.fragment(run_every=refresh_seconds)
    def _poll_log() -> None:
        cursor_key = f"{buffer_key}__{cursor_key_for_path(path)}"
        lines = read_log_delta(path, cursor_key=cursor_key)
        text = update_terminal_buffer(buffer_key, lines, max_lines=max_lines)
        if text:
            st.code(text, language="text")
        else:
            st.caption(empty_message)

    _poll_log()


def render_live_metrics_fragment(
    metrics_path: Path,
    *,
    refresh_seconds: float = DEFAULT_METRICS_REFRESH_SECONDS,
    fragment_key: str = "live_metrics",
) -> None:
    """Poll ``metrics.jsonl`` and render compact live KPI cards."""

    @st.fragment(run_every=refresh_seconds)
    def _poll_metrics() -> None:
        latest = read_metrics_latest(metrics_path)
        if not latest:
            st.caption("Esperando métricas de entrenamiento…")
            return

        cols = st.columns(4)
        cols[0].metric("Train return", format_float(latest.get("rollout/episode_return")))
        cols[1].metric("Eval return", format_float(latest.get("eval/episode_return")))
        cols[2].metric("SPS", format_float(latest.get("time/sps")))
        cols[3].metric("Env steps", format_float(latest.get("steps/nr_env_steps")))
        if active_run() is not None:
            st.caption(f"Actualizando cada {refresh_seconds:g} s · DAG rev {graph_revision()}")

    _ = fragment_key
    _poll_metrics()


def _notify_run_finished(exit_code: int) -> None:
    if st.session_state.get(RUN_FINISH_NOTIFIED_KEY):
        return
    st.session_state[RUN_FINISH_NOTIFIED_KEY] = True
    st.session_state[LAST_RUN_EXIT_CODE_KEY] = int(exit_code)
    if exit_code == 0:
        st.toast("Entrenamiento completado", icon="✅")
    else:
        st.toast(f"Entrenamiento terminó con código {exit_code}", icon="⚠️")


def render_run_status_fragment(*, refresh_seconds: float = DEFAULT_RUN_STATUS_REFRESH_SECONDS) -> None:
    """Poll subprocess state inside a fragment without ``st.rerun()`` on completion."""

    @st.fragment(run_every=refresh_seconds)
    def _poll_run() -> None:
        _render_training_run_status_content(refresh_seconds=refresh_seconds)

    _poll_run()


def _render_training_run_status_content(*, refresh_seconds: float) -> None:
    exit_code = poll_active_run()
    if exit_code is not None:
        _notify_run_finished(exit_code)
        if exit_code == 0:
            st.success("Entrenamiento completado")
        else:
            st.error(f"Entrenamiento terminó con código {exit_code}")
        return

    run = active_run()
    if run is not None:
        render_status_badge("training")
        st.caption(str(run.log_path))
        st.caption(f"DAG rev {graph_revision()} · actualizando cada {refresh_seconds:g} s")
        return

    last_exit = st.session_state.get(LAST_RUN_EXIT_CODE_KEY)
    if last_exit is not None:
        if int(last_exit) == 0:
            st.success("Último entrenamiento completado")
        else:
            st.error(f"Último entrenamiento terminó con código {last_exit}")
        return

    st.caption("Sin proceso activo")


def render_sidebar_run_fragment(
    *,
    refresh_seconds: float = DEFAULT_RUN_STATUS_REFRESH_SECONDS,
    testing_page_path: str,
    training_page_path: str,
) -> None:
    """Poll all subprocess kinds for the sidebar Run block on every page."""
    from jarl.app.lib.navigation import nav_page_path

    @st.fragment(run_every=refresh_seconds)
    def _poll_sidebar_run() -> None:
        if active_run() is not None:
            st.page_link(
                nav_page_path(training_page_path),
                label="Monitor de entrenamiento",
                icon=":material/terminal:",
            )
            _render_training_run_status_content(refresh_seconds=refresh_seconds)
            return
        if (
            active_agentic_case_run() is not None
            or st.session_state.get(AGENTIC_LIVE_WATCH_KEY)
            or st.session_state.get(AGENTIC_LAUNCH_UNBLOCK_KEY)
        ):
            st.page_link(
                nav_page_path(testing_page_path),
                label="Monitor de caso agéntico",
                icon=":material/science:",
            )
            _render_agentic_case_run_status_content(refresh_seconds=refresh_seconds)
            return
        if st.session_state.get(LAST_AGENTIC_CASE_EXIT_CODE_KEY) is not None:
            render_agentic_case_idle_status()
            return
        _render_idle_training_run_status_content()

    _poll_sidebar_run()


def _render_agentic_case_run_status_content(*, refresh_seconds: float) -> None:
    exit_code = poll_active_agentic_case_run()
    if exit_code is not None:
        _notify_agentic_case_finished(int(exit_code))
        if exit_code == 0:
            st.success("Último caso: Pasó.")
        else:
            from jarl.agentic.cases.labels import agentic_case_exit_code_label

            label = agentic_case_exit_code_label(int(exit_code))
            st.warning(f"Último caso: {label}.")
        return

    run = active_agentic_case_run()
    if run is not None:
        render_status_badge("training")
        st.caption(f"Caso `{run.case_id}`")
        st.caption(str(run.log_path))
        st.caption(f"Actualizando cada {refresh_seconds:g} s")
        return

    last_exit = st.session_state.get(LAST_AGENTIC_CASE_EXIT_CODE_KEY)
    if last_exit is not None:
        if int(last_exit) == 0:
            st.success("Último caso: Pasó.")
        else:
            from jarl.agentic.cases.labels import agentic_case_exit_code_label

            label = agentic_case_exit_code_label(int(last_exit))
            st.warning(f"Último caso: {label}.")
        return

    st.caption("Sin caso activo")


def _render_idle_training_run_status_content() -> None:
    exit_code = poll_active_run()
    if exit_code is not None:
        _notify_run_finished(exit_code)
        if exit_code == 0:
            st.success("Entrenamiento completado")
        else:
            st.error(f"Entrenamiento terminó con código {exit_code}")
        return

    last_exit = st.session_state.get(LAST_RUN_EXIT_CODE_KEY)
    if last_exit is not None:
        if int(last_exit) == 0:
            st.success("Último entrenamiento completado")
        else:
            st.error(f"Último entrenamiento terminó con código {last_exit}")
        return

    st.caption("Sin proceso activo")


def _notify_agentic_case_finished(exit_code: int) -> None:
    """Record agentic case exit code, toast once, and signal Testing live feeds."""
    if st.session_state.get(AGENTIC_CASE_FINISH_NOTIFIED_KEY):
        st.session_state[LAST_AGENTIC_CASE_EXIT_CODE_KEY] = int(exit_code)
        return
    from jarl.agentic.cases.labels import agentic_case_exit_code_label

    st.session_state[AGENTIC_CASE_FINISH_NOTIFIED_KEY] = True
    st.session_state[LAST_AGENTIC_CASE_EXIT_CODE_KEY] = int(exit_code)
    st.session_state[AGENTIC_LIVE_WATCH_KEY] = False
    st.session_state[AGENTIC_RESULT_WATCH_KEY] = True
    st.session_state[AGENTIC_LAUNCH_UNBLOCK_KEY] = True
    st.session_state[AGENTIC_CONVERSATION_WAKE_KEY] = True
    st.session_state[AGENTIC_CONVERSATION_REFRESH_NONCE_KEY] = (
        int(st.session_state.get(AGENTIC_CONVERSATION_REFRESH_NONCE_KEY, 0)) + 1
    )
    if exit_code == 0:
        st.toast("Caso pasó", icon="✅")
    else:
        label = agentic_case_exit_code_label(int(exit_code))
        st.toast(f"Caso: {label}", icon="⚠️")


def render_agentic_case_idle_status() -> None:
    """Render last agentic case status once without polling."""
    last_exit = st.session_state.get(LAST_AGENTIC_CASE_EXIT_CODE_KEY)
    if last_exit is None:
        st.caption("Sin caso activo")
        return
    if int(last_exit) == 0:
        st.success("Último caso: Pasó.")
    else:
        from jarl.agentic.cases.labels import agentic_case_exit_code_label

        label = agentic_case_exit_code_label(int(last_exit))
        st.warning(f"Último caso: {label}.")


def render_idle_run_status() -> None:
    """Render run status once when no background polling fragment is active."""
    _render_idle_training_run_status_content()
