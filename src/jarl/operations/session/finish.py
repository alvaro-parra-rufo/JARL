"""Declare that the current agent objective is already complete."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FinishRequest",
    "FinishResponse",
    "finish",
]


@dataclass(frozen=True, slots=True)
class FinishRequest:
    """Empty request: finishing has no caller-supplied arguments."""


@dataclass(frozen=True, slots=True)
class FinishResponse:
    """Outcome of ``finish``."""

    thread_id: str
    finished: bool = True

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return {
            "finished": self.finished,
            "thread_id": self.thread_id,
        }


def finish(request: FinishRequest, *, thread_id: str) -> FinishResponse:
    """Declare that the current objective is complete.

    This does not inspect or validate a case. Callers decide what finish means.
    """
    del request
    return FinishResponse(thread_id=thread_id)
