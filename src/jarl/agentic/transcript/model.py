"""Immutable conversation transcript value and load facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jarl.agentic.transcript.messages import TranscriptMessage

__all__ = ["ConversationTranscript"]


@dataclass(frozen=True, slots=True)
class ConversationTranscript:
    """Conversation thread loaded from the LangGraph checkpointer."""

    experiment_dir: Path
    thread_id: str
    messages: tuple[TranscriptMessage, ...]
    checkpoint_ns: str = ""

    @classmethod
    def load(
        cls,
        exp_dir: Path,
        *,
        thread_id: str | None = None,
        checkpoint_ns: str | None = None,
    ) -> ConversationTranscript | None:
        """Load the display transcript for an experiment session.

        When ``checkpoint_ns`` is omitted, uses the runtime active-namespace
        pointer if present, otherwise the parent namespace (``""``).
        """
        from jarl.agentic.transcript.checkpoint import CheckpointTranscriptReader

        reader = CheckpointTranscriptReader.for_experiment(exp_dir, thread_id=thread_id)
        if reader is None:
            return None
        return reader.load(checkpoint_ns=checkpoint_ns)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "experiment_dir": str(self.experiment_dir),
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "messages": [message.to_dict() for message in self.messages],
        }

    def visible_messages(
        self,
        *,
        exclude_roles: frozenset[str] = frozenset({"system"}),
    ) -> tuple[TranscriptMessage, ...]:
        """Return transcript rows whose role is not in ``exclude_roles``.

        The default hides ``system`` rows such as continuation nudges. Compiled
        node system prompts live on the session, not in this transcript.
        """
        return tuple(message for message in self.messages if message.role not in exclude_roles)

    def fingerprint(self) -> str:
        """Return a cheap revision token for UI cache invalidation."""
        if not self.messages:
            return f"{self.checkpoint_ns}:0"
        last = self.messages[-1]
        return f"{self.checkpoint_ns}:{len(self.messages)}:{last.role}:{len(last.content)}"
