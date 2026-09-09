"""LangGraph state schemas for agentic workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages

__all__ = ["AgentState"]


@dataclass(frozen=True, slots=True)
class AgentState:
    """Shared state carried through the experiment LangGraph.

    Args:
        messages: Conversation messages accumulated for the thread.
        experiment_dir: Absolute experiment directory.
        thread_id: LangGraph thread identifier.
    """

    # Channel reducer: each invoke payload appends; it does not replace the thread.
    messages: Annotated[list[AnyMessage], add_messages]
    experiment_dir: str
    thread_id: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | AgentState) -> AgentState:
        """Build state from a LangGraph invoke result or payload."""
        if isinstance(payload, AgentState):
            return payload
        raw_messages = payload.get("messages", [])
        messages = raw_messages if isinstance(raw_messages, list) else []
        return cls(
            messages=messages,
            experiment_dir=str(payload.get("experiment_dir", "")),
            thread_id=str(payload.get("thread_id", "")),
        )

    @property
    def message_count(self) -> int:
        """Return how many messages the thread currently holds."""
        return len(self.messages)

    def to_payload(self) -> dict[str, object]:
        """Return the shallow channel mapping LangGraph invoke expects."""
        return {
            "messages": self.messages,
            "experiment_dir": self.experiment_dir,
            "thread_id": self.thread_id,
        }

    def tool_called(self, name: str, *, after: int = 0) -> bool:
        """Return whether an AI message after ``after`` called ``name``."""
        for message in self.messages[after:]:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls:
                call_name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
                if call_name == name:
                    return True
        return False
