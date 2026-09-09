"""Append-only progress feed for live agentic execution monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from jarl.agentic.audit import AGENTIC_DIRNAME
from jarl.metadata import now_iso

AGENTIC_PROGRESS_FILENAME: Final[str] = "progress.jsonl"
"""Human-readable progress events under ``<exp_dir>/.agentic/``."""

__all__ = [
    "AGENTIC_PROGRESS_FILENAME",
    "ProgressEvent",
    "append_progress_event",
    "load_progress_events",
    "progress_path",
]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One append-only progress row for subscribers (UI tails, logs)."""

    ts: str
    kind: str
    message: str
    data: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable payload."""
        payload: dict[str, object] = {
            "ts": self.ts,
            "kind": self.kind,
            "message": self.message,
        }
        if self.data is not None:
            payload["data"] = self.data
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ProgressEvent:
        """Deserialize one stored progress event."""
        data = payload.get("data")
        if data is not None and not isinstance(data, dict):
            raise ValueError("Progress event data must be a JSON object or null.")
        return cls(
            ts=str(payload["ts"]),
            kind=str(payload["kind"]),
            message=str(payload["message"]),
            data=None if data is None else {str(key): value for key, value in data.items()},
        )


def progress_path(exp_dir: Path) -> Path:
    """Return ``<exp_dir>/.agentic/progress.jsonl``."""
    return exp_dir.resolve() / AGENTIC_DIRNAME / AGENTIC_PROGRESS_FILENAME


def append_progress_event(
    exp_dir: Path,
    kind: str,
    message: str,
    *,
    data: dict[str, object] | None = None,
) -> ProgressEvent:
    """Append one progress event for subscribers to tail."""
    event = ProgressEvent(ts=now_iso(), kind=kind, message=message, data=data)
    path = progress_path(exp_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
    return event


def load_progress_events(exp_dir: Path) -> tuple[ProgressEvent, ...]:
    """Load all progress events for an experiment."""
    path = progress_path(exp_dir)
    if not path.is_file():
        return ()
    events: list[ProgressEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        events.append(ProgressEvent.from_dict(payload))
    return tuple(events)
