"""Tests for transcript message normalization."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from jarl.agentic.transcript import TranscriptMessage


class TestNormalizeMessageContent:
    def test_string_content(self) -> None:
        assert TranscriptMessage.normalize_content("hola") == "hola"

    def test_multimodal_text_blocks(self) -> None:
        content = [{"type": "text", "text": "parte uno"}, {"type": "text", "text": "parte dos"}]
        assert TranscriptMessage.normalize_content(content) == "parte uno\nparte dos"

    def test_from_langchain(self) -> None:
        message = TranscriptMessage.from_langchain(0, HumanMessage(content="pregunta"))

        assert message.role == "human"
        assert message.content == "pregunta"
        assert message.index == 0

    def test_from_langchain_tool_calls(self) -> None:
        message = TranscriptMessage.from_langchain(
            1,
            AIMessage(
                content="",
                tool_calls=[{"name": "graph_summary", "args": {"x": 1}, "id": "c1", "type": "tool_call"}],
            ),
        )

        assert message.role == "ai"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "graph_summary"
