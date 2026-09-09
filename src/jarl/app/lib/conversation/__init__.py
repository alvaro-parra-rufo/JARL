"""Shared conversation UI for Testing, model-debug, and future chatbot views."""

from __future__ import annotations

from jarl.app.lib.conversation.format import format_llm_generation_chip, format_llm_usage
from jarl.app.lib.conversation.live import (
    CONVERSATION_CACHE_DIR_KEY,
    CONVERSATION_CACHE_REVISION_KEY,
    CONVERSATION_CACHE_TRANSCRIPT_KEY,
    poll_conversation_transcript,
    reset_conversation_live_cache,
    track_conversation_experiment_dir,
)
from jarl.app.lib.conversation.messages import (
    MessageRenderContext,
    ToolPayloadError,
    parse_tool_payload_error,
    render_messages_with_generation_chips,
    render_transcript_message,
    render_transcript_messages,
    tool_retry_attempts,
)

__all__ = [
    "CONVERSATION_CACHE_DIR_KEY",
    "CONVERSATION_CACHE_REVISION_KEY",
    "CONVERSATION_CACHE_TRANSCRIPT_KEY",
    "MessageRenderContext",
    "ToolPayloadError",
    "format_llm_generation_chip",
    "format_llm_usage",
    "parse_tool_payload_error",
    "poll_conversation_transcript",
    "render_messages_with_generation_chips",
    "render_transcript_message",
    "render_transcript_messages",
    "reset_conversation_live_cache",
    "tool_retry_attempts",
    "track_conversation_experiment_dir",
]
