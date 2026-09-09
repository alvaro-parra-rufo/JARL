"""Agent session metadata persisted beside the experiment manifest."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from jarl.agentic.errors import SessionError
from jarl.agentic.tools.specs import ToolFilterBy, merge_tool_filters
from jarl.metadata import now_iso
from jarl.utils import short_uuid
from jarl.utils.io import write_text_atomic

AGENTIC_SESSION_FILENAME: Final[str] = "agentic_session.json"
"""Filename for agent-only session metadata at the experiment root."""

__all__ = [
    "AGENTIC_SESSION_FILENAME",
    "AgenticSession",
    "CompiledNodeInfo",
    "load_or_create_session",
    "load_session",
    "save_session",
    "session_path",
]


class CompiledNodeInfo(BaseModel):
    """Bind snapshot for one compiled LangGraph node."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str | None = Field(
        default=None,
        description="Exact ``system_prompt`` last passed to ``create_agent`` for this node.",
    )
    filter_by: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Effective ``ToolFilterBy`` for this bind. Set keys only; JSON lists.",
    )
    tool_names: tuple[str, ...] = Field(
        default=(),
        description="Names of the ``StructuredTool`` objects bound for this node.",
    )

    @classmethod
    def from_bind(
        cls,
        *,
        system_prompt: str | None,
        filter_by: ToolFilterBy,
        tool_names: Iterable[str],
    ) -> CompiledNodeInfo:
        """Build a snapshot from the filter and tools used at ``create_agent``."""
        snapshot = {key: sorted(str(item) for item in values) for key, values in filter_by.items() if values}
        return cls(
            system_prompt=system_prompt,
            filter_by=snapshot,
            tool_names=tuple(tool_names),
        )

    def as_filter_by(self) -> ToolFilterBy:
        """Return this node's snapshot as ``ToolRegistry.filter_by`` kwargs."""
        return merge_tool_filters({key: set(values) for key, values in self.filter_by.items()})


class AgenticSession(BaseModel):
    """Agent-only session metadata (no experiment ``current_node`` duplication)."""

    model_config = ConfigDict(frozen=True)

    thread_id: str = Field(description="LangGraph thread identifier for this session.")
    created_at: str = Field(description="ISO-8601 session creation timestamp.")
    updated_at: str = Field(description="ISO-8601 last session update timestamp.")
    audit_reads: bool = Field(
        default=False,
        description="When ``True``, read tools are also written to the audit index.",
    )
    compiled_nodes: dict[str, CompiledNodeInfo] = Field(
        default_factory=dict,
        description="Last compile bind snapshot keyed by ``create_agent`` name.",
    )

    @classmethod
    def create_new(cls, *, audit_reads: bool = False) -> AgenticSession:
        """Create a fresh session with a new ``thread_id``."""
        timestamp = now_iso()
        return cls(
            thread_id=f"thread_{short_uuid()}",
            created_at=timestamp,
            updated_at=timestamp,
            audit_reads=audit_reads,
        )

    def touch(self) -> AgenticSession:
        """Return a copy with ``updated_at`` set to now."""
        return self.model_copy(update={"updated_at": now_iso()})

    def with_compiled_nodes(self, nodes: Mapping[str, CompiledNodeInfo]) -> AgenticSession:
        """Return a copy that replaces the compile snapshot with ``nodes``."""
        return self.model_copy(
            update={
                "compiled_nodes": dict(nodes),
                "updated_at": now_iso(),
            }
        )


def session_path(exp_dir: Path) -> Path:
    """Return the path to ``agentic_session.json`` under ``exp_dir``."""
    return exp_dir / AGENTIC_SESSION_FILENAME


def load_session(exp_dir: Path) -> AgenticSession | None:
    """Load session metadata when present."""
    path = session_path(exp_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AgenticSession.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        msg = f"Invalid agentic session file: {path}"
        raise SessionError(msg) from exc


def save_session(exp_dir: Path, session: AgenticSession) -> None:
    """Persist session metadata atomically."""
    write_text_atomic(session_path(exp_dir), session.model_dump_json(indent=2))


def load_or_create_session(exp_dir: Path, *, audit_reads: bool | None = None) -> AgenticSession:
    """Load an existing session or create and persist a new one.

    When ``audit_reads`` is provided, an existing session is updated and persisted
    if the flag changed.
    """
    existing = load_session(exp_dir)
    if existing is not None:
        if audit_reads is not None and existing.audit_reads != audit_reads:
            updated = existing.model_copy(
                update={"audit_reads": audit_reads, "updated_at": now_iso()},
            )
            save_session(exp_dir, updated)
            return updated
        return existing
    resolved_reads = audit_reads if audit_reads is not None else False
    session = AgenticSession.create_new(audit_reads=resolved_reads)
    save_session(exp_dir, session)
    return session
