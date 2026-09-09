"""Tests for conversation transcript helpers (no Streamlit widgets)."""

from __future__ import annotations

from jarl.agentic.transcript import TranscriptMessage
from jarl.app.lib.conversation import parse_tool_payload_error, tool_retry_attempts


def _tool(name: str, content: str = "{}", *, index: int = 0) -> TranscriptMessage:
    return TranscriptMessage(index=index, role="tool", content=content, tool_name=name)


class TestParseToolPayloadError:
    def test_reads_error_and_code(self) -> None:
        hint = parse_tool_payload_error('{"error": "nodo ausente", "code": "not_found"}')

        assert hint is not None
        assert hint.error == "nodo ausente"
        assert hint.code == "not_found"

    def test_ignores_success_payload(self) -> None:
        hint = parse_tool_payload_error('{"ok": true, "node_id": "n1"}')

        assert hint is None

    def test_ignores_plain_text(self) -> None:
        hint = parse_tool_payload_error("falló sin json")

        assert hint is None


class TestToolRetryAttempts:
    def test_counts_across_assistant_rows(self) -> None:
        messages = (
            TranscriptMessage(index=0, role="ai", content=""),
            _tool("graph_fork", '{"error": "busy", "code": "conflict"}', index=1),
            TranscriptMessage(index=2, role="ai", content=""),
            _tool("graph_fork", '{"error": "busy", "code": "conflict"}', index=3),
        )

        attempts = tool_retry_attempts(messages)

        assert attempts == (None, 1, None, 2)

    def test_human_resets_streak(self) -> None:
        messages = (
            _tool("graph_fork", index=0),
            TranscriptMessage(index=1, role="human", content="otra"),
            _tool("graph_fork", index=2),
        )

        attempts = tool_retry_attempts(messages)

        assert attempts == (1, None, 1)
