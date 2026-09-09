"""Read LangGraph checkpoints as conversation transcripts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from jarl.agentic.langgraph.state import AgentState
from jarl.agentic.session import load_session
from jarl.agentic.transcript.messages import TranscriptMessage
from jarl.agentic.transcript.model import ConversationTranscript
from jarl.agentic.transcript.pointer import ActiveCheckpointPointer

__all__ = [
    "CheckpointTranscriptReader",
    "ConversationRevision",
]


@dataclass(frozen=True, slots=True)
class ConversationRevision:
    """Cheap disk revision for conversation live polling.

    Opaque ``token`` changes when the checkpointer file or active-namespace
    pointer changes. Does not parse message contents.
    """

    token: str

    @classmethod
    def missing(cls) -> ConversationRevision:
        """Return a stable revision when no session or artefacts exist."""
        return cls(token="")


class CheckpointTranscriptReader:
    """Read-only loader of conversation messages from ``checkpoints.sqlite``."""

    def __init__(self, exp_dir: Path, thread_id: str) -> None:
        """Bind a reader to one experiment session thread."""
        self._exp_dir = exp_dir.resolve()
        self._thread_id = thread_id
        self._pointer = ActiveCheckpointPointer.for_thread(self._exp_dir, self._thread_id)

    @classmethod
    def for_experiment(
        cls,
        exp_dir: Path,
        *,
        thread_id: str | None = None,
    ) -> CheckpointTranscriptReader | None:
        """Build a reader, resolving ``thread_id`` from the session when omitted.

        Returns ``None`` when no session exists and ``thread_id`` is not provided.
        """
        resolved = exp_dir.resolve()
        resolved_thread = thread_id
        if resolved_thread is None:
            session = load_session(resolved)
            if session is None:
                return None
            resolved_thread = session.thread_id
        return cls(resolved, resolved_thread)

    @property
    def experiment_dir(self) -> Path:
        """Resolved experiment directory."""
        return self._exp_dir

    @property
    def thread_id(self) -> str:
        """LangGraph session thread id."""
        return self._thread_id

    @property
    def pointer(self) -> ActiveCheckpointPointer:
        """Active nested-namespace pointer for this session thread."""
        return self._pointer

    def revision(self) -> ConversationRevision:
        """Return a cheap token for sqlite + active-namespace pointer changes."""
        from jarl.agentic.langgraph.compile import langgraph_checkpoint_db_path

        db_token = _path_stat_token(langgraph_checkpoint_db_path(self._exp_dir))
        pointer_path = self._pointer.path
        pointer_stat = _path_stat_token(pointer_path)
        pointer_ns = self._pointer.read() or ""
        return ConversationRevision(token=f"{db_token}|{pointer_stat}|{pointer_ns}")

    def load(self, *, checkpoint_ns: str | None = None) -> ConversationTranscript:
        """Load messages for ``checkpoint_ns`` or the active pointer / parent.

        Args:
            checkpoint_ns: Explicit namespace. ``None`` means auto: pointer if set,
                otherwise the parent namespace (``""``).
        """
        resolved_ns = (self._pointer.read() or "") if checkpoint_ns is None else checkpoint_ns
        return self.load_namespace(resolved_ns)

    def load_parent(self) -> ConversationTranscript:
        """Load the parent-session checkpoint (``checkpoint_ns=""``)."""
        return self.load_namespace("")

    def load_namespace(self, checkpoint_ns: str) -> ConversationTranscript:
        """Load messages for one concrete ``checkpoint_ns``."""
        from jarl.agentic.langgraph.compile import langgraph_checkpoint_db_path

        checkpoint_path = langgraph_checkpoint_db_path(self._exp_dir)
        if not checkpoint_path.is_file():
            return ConversationTranscript(
                experiment_dir=self._exp_dir,
                thread_id=self._thread_id,
                messages=(),
                checkpoint_ns=checkpoint_ns,
            )
        raw_messages = self._read_messages(checkpoint_ns)
        messages = tuple(TranscriptMessage.from_langchain(index, message) for index, message in enumerate(raw_messages))
        return ConversationTranscript(
            experiment_dir=self._exp_dir,
            thread_id=self._thread_id,
            messages=messages,
            checkpoint_ns=checkpoint_ns,
        )

    def _read_messages(self, checkpoint_ns: str) -> list[BaseMessage]:
        from jarl.agentic.langgraph.compile import langgraph_checkpoint_db_path

        checkpoint_path = langgraph_checkpoint_db_path(self._exp_dir)
        conn = sqlite3.connect(
            f"file:{checkpoint_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        try:
            saver = SqliteSaver(conn)
            # Never call setup() on a read-only connection — the writer owns schema.
            checkpoint_tuple = saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": self._thread_id,
                        "checkpoint_ns": checkpoint_ns,
                    }
                }
            )
            if checkpoint_tuple is None:
                return []
            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            if not isinstance(channel_values, dict):
                return []
            return [
                message
                for message in AgentState.from_mapping(channel_values).messages
                if isinstance(message, BaseMessage)
            ]
        except (sqlite3.Error, TypeError, ValueError, KeyError):
            return []
        finally:
            conn.close()


def _path_stat_token(path: Path) -> str:
    """Return ``mtime_ns:size`` for ``path``, or ``missing`` when absent."""
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    return f"{stat.st_mtime_ns}:{stat.st_size}"
