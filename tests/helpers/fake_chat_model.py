"""Fake chat models for agentic unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

__all__ = ["RecordingFakeChatModel", "ToolBindingFakeChatModel"]


class ToolBindingFakeChatModel(GenericFakeChatModel):
    """Fake chat model that supports ``bind_tools`` like production LLMs."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ToolBindingFakeChatModel:
        """Return ``self`` so LangGraph agent compilation succeeds in tests."""
        del tools, tool_choice, kwargs
        return self


class RecordingFakeChatModel(ToolBindingFakeChatModel):
    """Fake chat model that records message batches passed to the LLM."""

    captured_message_batches: list[list[BaseMessage]] = Field(default_factory=list)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured_message_batches.append(list(messages))
        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])
