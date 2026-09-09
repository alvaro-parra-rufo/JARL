"""Render ``TranscriptMessage`` rows for Testing, debug, and future chat views."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from jarl.agentic.run_events import LlmEndEvent
from jarl.agentic.tool_attachments import extract_tool_attachments, resolve_attachment_path
from jarl.agentic.transcript import TranscriptMessage
from jarl.app.lib.conversation.format import format_llm_generation_chip

__all__ = [
    "MessageRenderContext",
    "ToolPayloadError",
    "parse_tool_payload_error",
    "render_messages_with_generation_chips",
    "render_transcript_message",
    "render_transcript_messages",
    "tool_retry_attempts",
]

_RETRY_BADGE_FROM = 2
"""Show «Intento N» from the second consecutive call of the same tool."""


@dataclass(frozen=True, slots=True)
class MessageRenderContext:
    """Optional context used to enrich tool rows in Runner Lab."""

    experiment_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ToolPayloadError:
    """Failure fields extracted from a tool JSON payload."""

    error: str
    code: str | None = None


def parse_tool_payload_error(content: str) -> ToolPayloadError | None:
    """Return error/code when the tool payload is a handler failure JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, str) or not error:
        return None
    raw_code = payload.get("code")
    code = raw_code if isinstance(raw_code, str) and raw_code else None
    return ToolPayloadError(error=error, code=code)


def tool_retry_attempts(messages: Sequence[TranscriptMessage]) -> tuple[int | None, ...]:
    """Return 1-based streak indexes for consecutive same-name tool rows.

    Human and system rows reset the streak. Assistant rows do not, so a model
    retry (AI then tool, AI then tool) still counts as consecutive.
    """
    attempts: list[int | None] = []
    last_name: str | None = None
    streak = 0
    for message in messages:
        if message.role in {"human", "system"}:
            last_name = None
            streak = 0
            attempts.append(None)
            continue
        if message.role != "tool":
            attempts.append(None)
            continue
        name = message.tool_name or ""
        if name and name == last_name:
            streak += 1
        else:
            streak = 1
            last_name = name or None
        attempts.append(streak)
    return tuple(attempts)


def render_transcript_messages(
    messages: Sequence[TranscriptMessage],
    *,
    context: MessageRenderContext | None = None,
) -> None:
    """Render a message list with retry indexes computed once."""
    render_messages_with_generation_chips(messages, (), context=context)


def render_messages_with_generation_chips(
    messages: Sequence[TranscriptMessage],
    generations: Sequence[LlmEndEvent],
    *,
    context: MessageRenderContext | None = None,
) -> None:
    """Render transcript rows, attaching generation chips to assistant messages in order."""
    pending = list(generations)
    attempts = tool_retry_attempts(messages)
    for message, attempt in zip(messages, attempts, strict=True):
        render_transcript_message(message, attempt=attempt, context=context)
        if message.role == "ai" and pending:
            st.caption(format_llm_generation_chip(pending.pop(0)))
    for leftover in pending:
        st.caption(format_llm_generation_chip(leftover))


def render_transcript_message(
    message: TranscriptMessage,
    *,
    attempt: int | None = None,
    context: MessageRenderContext | None = None,
) -> None:
    """Render one transcript row, including tool error and retry badges."""
    with st.container(border=True):
        st.markdown(f"**{_transcript_role_label(message)}**")
        payload_error = parse_tool_payload_error(message.content) if message.role == "tool" else None
        if payload_error is not None:
            st.badge("Error", color="red")
            if payload_error.code is not None:
                st.caption(payload_error.code)
        if attempt is not None and attempt >= _RETRY_BADGE_FROM:
            st.badge(f"Intento {attempt}")
        if message.content:
            if message.role == "tool":
                _render_tool_payload(
                    message.content,
                    tool_name=message.tool_name,
                    context=context,
                )
            else:
                st.markdown(message.content)
        for tool_call in message.tool_calls:
            with st.expander(f"tool_call · {tool_call.name or 'sin nombre'}"):
                st.json({"id": tool_call.id, "args": tool_call.args})


def _transcript_role_label(message: TranscriptMessage) -> str:
    """Return the Spanish role label for a transcript row."""
    if message.role == "human":
        return "Usuario"
    if message.role == "ai":
        return "Asistente"
    if message.role == "tool":
        tool_name = message.tool_name or "tool"
        return f"Tool · {tool_name}"
    if message.role == "system":
        return "Sistema"
    return "Mensaje"


def _render_tool_payload(
    content: str,
    *,
    tool_name: str | None,
    context: MessageRenderContext | None,
) -> None:
    """Render a tool result as JSON and any displayable attachments."""
    stripped = content.strip()
    payload: dict[str, object] | None = None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                payload = parsed
                st.json(parsed)
            else:
                st.json(parsed)
                return
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        st.code(content, language=None)
        return
    _render_tool_attachments(payload, tool_name=tool_name, context=context)


def _render_tool_attachments(
    payload: dict[str, object],
    *,
    tool_name: str | None,
    context: MessageRenderContext | None,
) -> None:
    """Render videos and other attachments referenced by a tool payload."""
    if context is None or context.experiment_dir is None:
        return
    attachments = extract_tool_attachments(tool_name, payload)
    if not attachments:
        return
    for attachment in attachments:
        absolute_path = resolve_attachment_path(
            context.experiment_dir,
            payload,
            attachment.relative_path,
        )
        if absolute_path is None:
            st.caption(f"Artefacto no encontrado: `{attachment.relative_path}`")
            continue
        if attachment.kind == "video":
            st.caption(attachment.label or attachment.relative_path)
            st.video(str(absolute_path))
            continue
        st.caption(attachment.label or attachment.relative_path)
        st.code(str(absolute_path), language=None)
