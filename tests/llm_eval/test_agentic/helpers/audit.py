"""Read ``agentic_audit.jsonl`` for deterministic tool-eval assertions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarl.agentic.audit import audit_index_path

__all__ = ["AuditEvent", "load_audit_events"]


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One parsed audit index row."""

    tool: str
    kind: str
    request: dict[str, Any]
    result: dict[str, Any] | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> AuditEvent:
        """Build an event from a JSON object."""
        request = row.get("request", {})
        result = row.get("result")
        return cls(
            tool=str(row["tool"]),
            kind=str(row["kind"]),
            request=request if isinstance(request, dict) else {},
            result=result if isinstance(result, dict) else None,
        )


def load_audit_events(exp_dir: Path) -> list[AuditEvent]:
    """Load all audit events for an experiment directory."""
    audit_path = audit_index_path(exp_dir)
    if not audit_path.is_file():
        return []
    return [
        AuditEvent.from_row(json.loads(line)) for line in audit_path.read_text(encoding="utf-8").splitlines() if line
    ]
