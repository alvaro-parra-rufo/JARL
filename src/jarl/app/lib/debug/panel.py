"""Paint the conversation pane or the model-debug timeline."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from jarl.agentic.audit import load_run_events
from jarl.agentic.debug import DebugSegment, ModelDebugView
from jarl.agentic.errors import SessionError
from jarl.agentic.run_events import LlmEndEvent, LlmErrorEvent, RunWindows
from jarl.agentic.session import load_session
from jarl.agentic.tools.specs import BoundTool
from jarl.agentic.transcript import ConversationTranscript, TranscriptMessage
from jarl.app.lib.conversation import (
    MessageRenderContext,
    render_messages_with_generation_chips,
)
from jarl.app.lib.debug.flag import is_debug_enabled
from jarl.app.lib.ui import empty_state

__all__ = ["render_conversation", "render_model_debug"]


def render_conversation(
    exp_dir: Path | None,
    transcript: ConversationTranscript | None,
    *,
    running: bool = False,
) -> None:
    """Render the visible thread or the debug timeline, according to the flag."""
    if is_debug_enabled():
        if exp_dir is None:
            empty_state("Sin experimento", "Ejecuta un caso para ver el debug del modelo.")
            return
        render_model_debug(exp_dir, running=running)
        return
    _render_visible_transcript(exp_dir, transcript, running=running)


def render_model_debug(exp_dir: Path, *, running: bool = False) -> None:
    """Render ``ModelDebugView`` for ``exp_dir``."""
    loaded = _load_model_debug_view(exp_dir, running=running)
    if loaded is None:
        return
    view, thread_id = loaded
    _render_sticky_header(view)
    if running and _llm_generation_in_flight(exp_dir, thread_id):
        st.caption("Generando…")
    if not view.segments:
        st.caption("Sin tramos todavía.")
        return
    for index, segment in enumerate(view.segments):
        _render_segment(segment, index=index, exp_dir=exp_dir)


def _render_visible_transcript(
    exp_dir: Path | None,
    transcript: ConversationTranscript | None,
    *,
    running: bool,
) -> None:
    """Render the user-facing thread with generation chips when possible."""
    loaded = None if exp_dir is None else _load_model_debug_view(exp_dir, running=running, quiet=True)
    if loaded is not None and loaded[0].segments:
        view, thread_id = loaded
        messages, generations = _flatten_segments_for_chat(view.segments)
    elif transcript is not None:
        messages = transcript.visible_messages()
        generations = ()
        thread_id = transcript.thread_id
    else:
        st.caption("Sin sesión LangGraph todavía.")
        return

    if not messages:
        if running:
            st.caption("Esperando primer turno o tool call…")
        else:
            st.caption("Esperando mensajes en el checkpointer…")
        return

    source = "en vivo" if running else "checkpointer"
    st.caption(f"Thread `{thread_id}` · {len(messages)} mensajes · {source}")
    render_messages_with_generation_chips(
        messages,
        generations,
        context=MessageRenderContext(experiment_dir=exp_dir),
    )
    if running and exp_dir is not None and _llm_generation_in_flight(exp_dir, thread_id):
        st.caption("Generando…")


def _load_model_debug_view(
    exp_dir: Path,
    *,
    running: bool,
    quiet: bool = False,
) -> tuple[ModelDebugView, str] | None:
    """Return view and thread id, or ``None`` when the experiment cannot be loaded."""
    try:
        session = load_session(exp_dir)
    except (SessionError, OSError, ValueError, TypeError) as exc:
        if not quiet:
            st.error(f"No se pudo leer la sesión: {exc}")
        return None
    if session is None:
        if not quiet:
            empty_state("Sin sesión", "No hay agentic_session.json en este experimento.")
        return None
    try:
        view = ModelDebugView.from_experiment(exp_dir, running=running)
    except (SessionError, OSError, ValueError, TypeError) as exc:
        if not quiet:
            st.error(f"No se pudo armar el debug del modelo: {exc}")
        return None
    return view, session.thread_id


def _flatten_segments_for_chat(
    segments: tuple[DebugSegment, ...],
) -> tuple[tuple[TranscriptMessage, ...], tuple[LlmEndEvent, ...]]:
    """Flatten debug segments into a user-facing message list plus generation chips."""
    messages: list[TranscriptMessage] = []
    generations: list[LlmEndEvent] = []
    for segment in segments:
        messages.extend(message for message in segment.messages if message.role != "system")
        generations.extend(segment.context.generations)
    return tuple(messages), tuple(generations)


def _llm_generation_in_flight(exp_dir: Path, thread_id: str) -> bool:
    """Return whether run events show an open ``llm_start`` without end/error."""
    try:
        events = load_run_events(exp_dir, thread_id)
    except (OSError, ValueError, TypeError):
        return False
    return RunWindows.from_events(events).llm_generation_in_flight


def _render_sticky_header(view: ModelDebugView) -> None:
    """Render the sticky LangGraph phase and experiment node labels."""
    parts: list[str] = []
    if view.phase:
        parts.append(f"Fase `{view.phase}`")
    if view.experiment_node_id:
        parts.append(f"Nodo `{view.experiment_node_id}`")
    source = "en vivo" if view.running else "checkpointer"
    parts.append(source)
    st.caption(" · ".join(parts))


def _render_segment(segment: DebugSegment, *, index: int, exp_dir: Path) -> None:
    """Render one timeline band: bind context, then messages."""
    context = segment.context
    label = context.node or "sin nodo"
    with st.container(border=True):
        st.markdown(f"**Tramo · {label}**")
        if context.inferred:
            st.caption("Contexto inferido (sin eventos graph_node).")
        with st.expander("System", expanded=False, key=f"jarl_debug_sys_{index}"):
            prompt = context.system_prompt
            if prompt:
                st.code(prompt, language=None)
            else:
                st.caption("Sin system_prompt en el compile.")
        with st.expander("Tools", expanded=False, key=f"jarl_debug_tools_{index}"):
            _render_tools(context.tools, segment_index=index)
        _render_segment_body(segment, exp_dir=exp_dir)


def _render_tools(tools: tuple[BoundTool, ...], *, segment_index: int) -> None:
    """Render bound-tool chips and closed schema expanders."""
    if not tools:
        st.caption("Sin tools bindadas.")
        return
    st.caption(" · ".join(tool.name for tool in tools))
    for tool in tools:
        with st.expander(tool.name, expanded=False, key=f"jarl_debug_tool_{segment_index}_{tool.name}"):
            if tool.description:
                st.markdown(tool.description)
            st.json(tool.schema)


def _render_segment_body(segment: DebugSegment, *, exp_dir: Path) -> None:
    """Render messages with LLM chips next to AI rows, then leftover protocol rows."""
    render_messages_with_generation_chips(
        segment.messages,
        segment.context.generations,
        context=MessageRenderContext(experiment_dir=exp_dir),
    )
    for error in segment.context.errors:
        _render_llm_error(error)


def _render_llm_error(error: LlmErrorEvent) -> None:
    """Render a protocol row for a generation that failed without an AI message."""
    with st.container(border=True):
        st.markdown("**LLM · error**")
        if error.error_type:
            st.badge(error.error_type, color="red")
        if error.error:
            st.caption(error.error)
