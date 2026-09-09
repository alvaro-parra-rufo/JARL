"""Tests for ConversationTranscript value object."""

from __future__ import annotations

from pathlib import Path

from jarl.agentic.transcript import ConversationTranscript, TranscriptMessage


class TestVisibleMessages:
    def test_hides_system_rows_by_default(self, tmp_path: Path) -> None:
        transcript = ConversationTranscript(
            experiment_dir=tmp_path,
            thread_id="thread-1",
            messages=(
                TranscriptMessage(index=0, role="human", content="pregunta"),
                TranscriptMessage(index=1, role="system", content="nudge"),
                TranscriptMessage(index=2, role="ai", content="respuesta"),
            ),
        )

        visible = transcript.visible_messages()

        assert [message.role for message in visible] == ["human", "ai"]
        assert [message.index for message in visible] == [0, 2]
        assert [message.content for message in visible] == ["pregunta", "respuesta"]

    def test_exclude_roles_can_keep_system_rows(self, tmp_path: Path) -> None:
        transcript = ConversationTranscript(
            experiment_dir=tmp_path,
            thread_id="thread-1",
            messages=(
                TranscriptMessage(index=0, role="system", content="nudge"),
                TranscriptMessage(index=1, role="tool", content="{}"),
            ),
        )

        visible = transcript.visible_messages(exclude_roles=frozenset())

        assert [message.role for message in visible] == ["system", "tool"]

    def test_fingerprint_changes_with_messages(self, tmp_path: Path) -> None:
        empty = ConversationTranscript(experiment_dir=tmp_path, thread_id="t", messages=())
        filled = ConversationTranscript(
            experiment_dir=tmp_path,
            thread_id="t",
            messages=(TranscriptMessage(index=0, role="human", content="hola"),),
        )

        assert empty.fingerprint() != filled.fingerprint()
