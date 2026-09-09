"""LangChain callbacks that record LLM metadata into run events."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from jarl.agentic.audit import RunTracker
from jarl.agentic.run_events import LlmEndEvent, LlmErrorEvent, LlmStartEvent, LlmUsage

__all__ = ["LlmGenerationCallback"]


class LlmGenerationCallback(BaseCallbackHandler):
    """Append ``llm_start`` / ``llm_end`` / ``llm_error`` rows to an open run.

    Records LangGraph node, duration, and ``AIMessage.usage_metadata`` tokens.
    Never persists prompts, completions, or message contents. Chat models emit
    ``on_chat_model_start`` then ``on_llm_end`` / ``on_llm_error`` (there is
    no ``on_chat_model_end``).
    """

    def __init__(self, tracker: RunTracker) -> None:
        """Bind this callback to the already open ``RunTracker``."""
        self._tracker = tracker
        self._started_at: dict[UUID, float] = {}
        self._nodes: dict[UUID, str] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record generation start for the active LangGraph node."""
        del serialized, messages, parent_run_id, kwargs
        node = _langgraph_node(metadata, tags)
        self._nodes[run_id] = node
        self._started_at[run_id] = time.monotonic()
        self._tracker.append_event(LlmStartEvent(node=node))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record generation end: duration and usage, never contents."""
        del parent_run_id
        node = self._nodes.pop(run_id, _langgraph_node(kwargs.get("metadata"), tags))
        started = self._started_at.pop(run_id, None)
        duration_ms = None if started is None else (time.monotonic() - started) * 1000.0
        self._tracker.append_event(
            LlmEndEvent(
                node=node,
                duration_ms=duration_ms,
                usage=_usage_from_llm_result(response),
            )
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a provider error without prompt or completion text."""
        del parent_run_id
        node = self._nodes.pop(run_id, _langgraph_node(kwargs.get("metadata"), tags))
        self._started_at.pop(run_id, None)
        self._tracker.append_event(
            LlmErrorEvent(
                node=node,
                error_type=type(error).__name__,
                error=str(error),
            )
        )


def _langgraph_node(metadata: Mapping[str, object] | None, tags: Sequence[str] | None) -> str:
    """Return the enclosing LangGraph agent node for this generation.

    Nested ``create_agent`` graphs report ``langgraph_node`` as the inner
    ``model`` (or ``tools``) node. Prefer the parent agent name so events align
    with ``graph_node`` rows.
    """
    if metadata is not None:
        agent_name = metadata.get("lc_agent_name")
        if isinstance(agent_name, str) and agent_name:
            return agent_name
        checkpoint_ns = metadata.get("checkpoint_ns")
        if isinstance(checkpoint_ns, str) and checkpoint_ns:
            head = checkpoint_ns.split(":", 1)[0]
            if head:
                return head
        raw = metadata.get("langgraph_node")
        if isinstance(raw, str) and raw:
            return raw
    if tags:
        for tag in tags:
            if tag and not tag.startswith("seq:") and ":" not in tag:
                return tag
    return ""


def _usage_from_llm_result(response: LLMResult) -> LlmUsage | None:
    """Return usage from the top-choice chat generation (``generations[0][0]``).

    A single chat call may return several candidates when ``n > 1``. Usage is
    taken from the first candidate only so input tokens are not double-counted
    (same policy as ``LLMResult.flatten``). Agentic ReAct invokes use one
    prompt per call; multi-prompt batches are out of scope here.
    """
    if not response.generations or not response.generations[0]:
        return None
    generation = response.generations[0][0]
    if not isinstance(generation, ChatGeneration):
        return None
    message = generation.message
    if not isinstance(message, AIMessage):
        return None
    return LlmUsage.from_mapping(message.usage_metadata)
