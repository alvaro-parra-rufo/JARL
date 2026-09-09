"""Runtime pointer to the active nested LangGraph checkpoint namespace."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Self

from jarl.agentic.audit import run_dir
from jarl.utils.io import write_text_atomic

ACTIVE_CHECKPOINT_NS_FILENAME: Final[str] = "active_checkpoint_ns"
"""Filename for the nested ``checkpoint_ns`` pointer under a run directory."""

__all__ = [
    "ACTIVE_CHECKPOINT_NS_FILENAME",
    "ActiveCheckpointPointer",
    "active_checkpoint_ns_path",
]


class ActiveCheckpointPointer:
    """Per-thread on-disk pointer to the active nested ``checkpoint_ns``.

    Lives at ``<exp_dir>/.agentic/runs/<thread_id>/active_checkpoint_ns``. Use as a
    context manager around one invoke turn: enter clears a stale pointer, exit
    always clears (including ``KeyboardInterrupt``).
    """

    def __init__(self, exp_dir: Path, thread_id: str) -> None:
        """Bind the pointer to one experiment session thread."""
        self._exp_dir = exp_dir.resolve()
        self._thread_id = thread_id
        self._path = active_checkpoint_ns_path(self._exp_dir, self._thread_id)

    @classmethod
    def for_thread(cls, exp_dir: Path, thread_id: str) -> Self:
        """Return a pointer bound to ``exp_dir`` and ``thread_id``."""
        return cls(exp_dir, thread_id)

    def __enter__(self) -> Self:
        """Clear any leftover pointer from a previous interrupted turn."""
        self.clear()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Always clear the pointer when the invoke turn ends."""
        del exc_type, exc, traceback
        self.clear()

    @property
    def path(self) -> Path:
        """Path of the pointer file."""
        return self._path

    @property
    def experiment_dir(self) -> Path:
        """Resolved experiment directory."""
        return self._exp_dir

    @property
    def thread_id(self) -> str:
        """LangGraph session thread id for this pointer."""
        return self._thread_id

    def read(self) -> str | None:
        """Return the active nested namespace, or ``None`` when unset."""
        if not self._path.is_file():
            return None
        text = self._path.read_text(encoding="utf-8").strip()
        return text or None

    def set(self, checkpoint_ns: str) -> None:
        """Persist ``checkpoint_ns`` as the active nested namespace."""
        if not checkpoint_ns:
            self.clear()
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self._path, checkpoint_ns)

    def clear(self) -> None:
        """Remove the active-namespace pointer if present."""
        if self._path.is_file():
            self._path.unlink()


def active_checkpoint_ns_path(exp_dir: Path, thread_id: str) -> Path:
    """Return ``<exp_dir>/.agentic/runs/<thread_id>/active_checkpoint_ns``."""
    return run_dir(exp_dir.resolve(), thread_id) / ACTIVE_CHECKPOINT_NS_FILENAME
