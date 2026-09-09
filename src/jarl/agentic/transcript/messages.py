"""Normalized conversation message rows for agentic transcripts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

TranscriptRole = Literal["human", "ai", "tool", "system", "unknown"]

__all__ = [
    "ToolCallSummary",
    "TranscriptMessage",
    "TranscriptRole",
]


@dataclass(frozen=True, slots=True)
class ToolCallSummary:
    """Compact tool invocation attached to an assistant message."""

    name: str
    args: dict[str, object]
    id: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ToolCallSummary:
        """Deserialize one tool-call summary."""
        args = payload.get("args", {})
        if not isinstance(args, dict):
            args = {}
        return cls(
            name=str(payload.get("name", "")),
            args={str(key): value for key, value in args.items()},
            id=str(payload.get("id", "")),
        )


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    """One normalized message in a LangGraph conversation thread."""

    index: int
    role: TranscriptRole
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCallSummary, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        payload: dict[str, object] = {
            "index": self.index,
            "role": self.role,
            "content": self.content,
        }
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TranscriptMessage:
        """Deserialize one transcript row."""
        raw_calls = payload.get("tool_calls", ())
        tool_calls: tuple[ToolCallSummary, ...] = ()
        if isinstance(raw_calls, list):
            tool_calls = tuple(ToolCallSummary.from_dict(item) for item in raw_calls if isinstance(item, dict))
        raw_role = payload.get("role", "unknown")
        role: TranscriptRole
        if raw_role in {"human", "ai", "tool", "system", "unknown"}:
            role = cast("TranscriptRole", raw_role)
        else:
            role = "unknown"
        tool_name = payload.get("tool_name")
        tool_call_id = payload.get("tool_call_id")
        return cls(
            index=int(payload["index"]),
            role=role,
            content=str(payload.get("content", "")),
            tool_name=None if tool_name is None else str(tool_name),
            tool_call_id=None if tool_call_id is None else str(tool_call_id),
            tool_calls=tool_calls,
        )

    @classmethod
    def from_langchain(cls, index: int, message: BaseMessage) -> TranscriptMessage:
        """Build a display row from one LangChain message."""
        return cls(
            index=index,
            role=cls.role_for_message(message),
            content=cls.normalize_content(message.content),
            tool_name=message.name if isinstance(message, ToolMessage) else None,
            tool_call_id=message.tool_call_id if isinstance(message, ToolMessage) else None,
            tool_calls=cls.tool_calls_for_message(message),
        )

    @staticmethod
    def normalize_content(content: object) -> str:
        """Flatten LangChain message content into displayable text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        parts.append(str(block.get("text", "")))
                    else:
                        parts.append(json.dumps(block, ensure_ascii=False))
                    continue
                parts.append(str(block))
            return "\n".join(part for part in parts if part)
        return str(content)

    @staticmethod
    def role_for_message(message: BaseMessage) -> TranscriptRole:
        """Map a LangChain message to a transcript role."""
        if isinstance(message, HumanMessage):
            return "human"
        if isinstance(message, AIMessage):
            return "ai"
        if isinstance(message, ToolMessage):
            return "tool"
        if isinstance(message, SystemMessage):
            return "system"
        message_type = getattr(message, "type", None)
        if message_type in {"human", "ai", "tool", "system"}:
            return message_type
        return "unknown"

    @staticmethod
    def tool_calls_for_message(message: BaseMessage) -> tuple[ToolCallSummary, ...]:
        """Extract tool-call summaries from an assistant message."""
        if not isinstance(message, AIMessage):
            return ()
        summaries: list[ToolCallSummary] = []
        for tool_call in message.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = str(tool_call.get("name", ""))
            args = tool_call.get("args", {})
            call_id = str(tool_call.get("id", ""))
            if not isinstance(args, dict):
                args = {}
            summaries.append(ToolCallSummary(name=name, args=args, id=call_id))
        return tuple(summaries)
