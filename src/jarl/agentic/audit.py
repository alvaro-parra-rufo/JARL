"""Unified audit index and per-run tracking for agentic workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from jarl.agentic.run_events import RunEvent, parse_run_event
from jarl.metadata import now_iso
from jarl.utils.io import write_text_atomic

AGENTIC_AUDIT_FILENAME: Final[str] = "agentic_audit.jsonl"
"""Append-only audit index at the experiment root."""

AGENTIC_DIRNAME: Final[str] = ".agentic"
"""Root directory for agentic artefacts under an experiment."""

AGENTIC_RUNS_DIRNAME: Final[str] = "runs"
"""Per-thread run detail directory under ``.agentic/``."""

AGENTIC_SUBAGENTS_DIRNAME: Final[str] = "subagents"
"""Nested directory for subagent runs under a parent agent run."""

RUN_MANIFEST_FILENAME: Final[str] = "manifest.json"
"""Run manifest filename inside a run directory."""

RUN_EVENTS_FILENAME: Final[str] = "events.jsonl"
"""Run event log filename inside a run directory."""

RunKind = Literal["agent", "subagent"]
RunStatus = Literal["running", "completed", "failed"]
AuditKind = Literal["read", "mutation", "train", "subagent", "error"]

__all__ = [
    "AGENTIC_AUDIT_FILENAME",
    "AGENTIC_DIRNAME",
    "AGENTIC_RUNS_DIRNAME",
    "AGENTIC_SUBAGENTS_DIRNAME",
    "RUN_EVENTS_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "AuditIndexEvent",
    "RunManifest",
    "RunTracker",
    "SubgraphRunLink",
    "append_audit_index",
    "audit_index_path",
    "load_audit_index",
    "load_run_events",
    "open_run",
    "run_dir",
    "subagent_run_dir",
]


@dataclass(frozen=True, slots=True)
class SubgraphRunLink:
    """Parent-index linkage for one subagent invocation.

    Args:
        sub_thread_id: Child run thread id (``{parent}:{suffix}``).
        parent_tool: Invoking tool name when known.
        run_dir: Experiment-relative path under the parent run
            (``.agentic/runs/<parent>/subagents/<suffix>``).
    """

    sub_thread_id: str
    parent_tool: str | None
    run_dir: str


@dataclass(frozen=True, slots=True)
class AuditIndexEvent:
    """Summarized audit row written to ``agentic_audit.jsonl``."""

    ts: str
    thread_id: str
    tool: str
    kind: AuditKind
    request: dict[str, object]
    result: dict[str, object] | None = None
    node_before: str | None = None
    node_after: str | None = None
    sub_thread_id: str | None = None
    parent_tool: str | None = None
    run_dir: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AuditIndexEvent:
        """Deserialize one audit index event from a JSON object."""
        request = payload.get("request")
        if not isinstance(request, dict):
            raise ValueError("Audit event request must be a JSON object.")
        result = payload.get("result")
        if result is not None and not isinstance(result, dict):
            raise ValueError("Audit event result must be a JSON object or null.")
        kind = str(payload["kind"])
        if kind not in {"read", "mutation", "train", "subagent", "error"}:
            msg = f"Unsupported audit event kind: {kind!r}"
            raise ValueError(msg)
        return cls(
            ts=str(payload["ts"]),
            thread_id=str(payload["thread_id"]),
            tool=str(payload["tool"]),
            kind=cast("AuditKind", kind),
            request={str(key): value for key, value in request.items()},
            result=None if result is None else {str(key): value for key, value in result.items()},
            node_before=_optional_string(payload.get("node_before")),
            node_after=_optional_string(payload.get("node_after")),
            sub_thread_id=_optional_string(payload.get("sub_thread_id")),
            parent_tool=_optional_string(payload.get("parent_tool")),
            run_dir=_optional_string(payload.get("run_dir")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize for JSONL persistence, omitting unset optional fields."""
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


class RunManifest(BaseModel):
    """Manifest for a single agent or subagent run."""

    model_config = ConfigDict(frozen=True)

    thread_id: str = Field(description="LangGraph thread for this run.")
    parent_thread_id: str | None = Field(
        default=None,
        description="Parent thread when ``run_kind`` is ``subagent``.",
    )
    run_kind: RunKind = Field(description="Whether this run is the main agent or a subagent.")
    graph_id: str = Field(description="LangGraph module identifier.")
    parent_tool: str | None = Field(
        default=None,
        description="Invoking tool name for subagent runs.",
    )
    status: RunStatus = Field(default="running", description="Lifecycle status of the run.")
    created_at: str = Field(description="ISO-8601 creation timestamp.")
    updated_at: str = Field(description="ISO-8601 last update timestamp.")
    llm_profile: str | None = Field(
        default=None,
        description="Resolved LLM profile name when a catalog was used.",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Resolved LLM provider identifier.",
    )
    llm_model: str | None = Field(
        default=None,
        description="Resolved LLM model name.",
    )


class RunTracker:
    """Append-only event log and manifest updates for one run directory."""

    def __init__(self, run_root: Path, manifest: RunManifest) -> None:
        """Bind a tracker to an on-disk run directory."""
        self._run_root = run_root
        self._manifest_path = run_root / RUN_MANIFEST_FILENAME
        self._events_path = run_root / RUN_EVENTS_FILENAME
        self._manifest = manifest

    @property
    def thread_id(self) -> str:
        """Thread id for this run."""
        return self._manifest.thread_id

    @property
    def path(self) -> Path:
        """Directory containing ``manifest.json`` and ``events.jsonl``."""
        return self._run_root

    @property
    def manifest(self) -> RunManifest:
        """Current run manifest."""
        return self._manifest

    def append_event(self, event: RunEvent) -> None:
        """Append a typed event to ``events.jsonl``."""
        self._run_root.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json(exclude_none=True) + "\n")

    def set_status(self, status: RunStatus) -> None:
        """Update run status and persist the manifest."""
        self._manifest = self._manifest.model_copy(
            update={"status": status, "updated_at": now_iso()},
        )
        write_text_atomic(self._manifest_path, self._manifest.model_dump_json(indent=2))

    def complete(self) -> None:
        """Mark the run as completed."""
        self.set_status("completed")

    def fail(self) -> None:
        """Mark the run as failed."""
        self.set_status("failed")


def agentic_root(exp_dir: Path) -> Path:
    """Return ``<exp_dir>/.agentic``."""
    return exp_dir / AGENTIC_DIRNAME


def audit_index_path(exp_dir: Path) -> Path:
    """Return ``<exp_dir>/agentic_audit.jsonl``."""
    return exp_dir / AGENTIC_AUDIT_FILENAME


def run_dir(exp_dir: Path, thread_id: str) -> Path:
    """Return ``<exp_dir>/.agentic/runs/<thread_id>`` for an agent session run."""
    return agentic_root(exp_dir) / AGENTIC_RUNS_DIRNAME / thread_id


def subagent_run_dir(exp_dir: Path, parent_thread_id: str, suffix: str) -> Path:
    """Return ``<exp_dir>/.agentic/runs/<parent>/subagents/<suffix>``.

    Args:
        exp_dir: Experiment root directory.
        parent_thread_id: Parent agent session thread id.
        suffix: Short subagent suffix (graph leaf name or explicit override).

    Returns:
        Nested on-disk directory for the child subagent run.
    """
    if not suffix or "/" in suffix or suffix in {".", ".."}:
        msg = f"Invalid subagent run suffix: {suffix!r}"
        raise ValueError(msg)
    return run_dir(exp_dir, parent_thread_id) / AGENTIC_SUBAGENTS_DIRNAME / suffix


def append_audit_index(exp_dir: Path, event: AuditIndexEvent) -> None:
    """Append a summarized audit row to ``agentic_audit.jsonl``."""
    path = audit_index_path(exp_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")


def load_audit_index(exp_dir: Path) -> tuple[AuditIndexEvent, ...]:
    """Load summarized audit events for an experiment.

    Missing audit files represent workflows without audited tool calls and
    therefore return an empty tuple.
    """
    path = audit_index_path(exp_dir)
    if not path.is_file():
        return ()
    events: list[AuditIndexEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Each audit index line must contain a JSON object.")
        events.append(AuditIndexEvent.from_dict(payload))
    return tuple(events)


def load_run_events(exp_dir: Path, thread_id: str) -> tuple[RunEvent, ...]:
    """Load typed events from ``.agentic/runs/<thread_id>/events.jsonl``.

    A missing file means the thread has not recorded a run yet and returns an
    empty tuple.
    """
    path = run_dir(exp_dir, thread_id) / RUN_EVENTS_FILENAME
    if not path.is_file():
        return ()
    events: list[RunEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Each run event line must contain a JSON object.")
        events.append(parse_run_event(payload))
    return tuple(events)


def open_run(
    exp_dir: Path,
    thread_id: str,
    *,
    parent_thread_id: str | None = None,
    run_kind: RunKind = "agent",
    graph_id: str = "experiment",
    parent_tool: str | None = None,
    thread_suffix: str | None = None,
    llm_profile: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> RunTracker:
    """Create or resume tracking for an agent or nested subagent run.

    Agent runs live at ``.agentic/runs/<thread_id>/``. Subagent runs nest under
    the parent at ``.agentic/runs/<parent>/subagents/<suffix>/``.
    """
    resolved = exp_dir.resolve()
    if run_kind == "subagent":
        if parent_thread_id is None:
            msg = "parent_thread_id is required for subagent runs."
            raise ValueError(msg)
        suffix = thread_suffix or _subagent_suffix(thread_id, parent_thread_id)
        root = subagent_run_dir(resolved, parent_thread_id, suffix)
    else:
        root = run_dir(resolved, thread_id)
    manifest_path = root / RUN_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return RunTracker(root, manifest)

    timestamp = now_iso()
    manifest = RunManifest(
        thread_id=thread_id,
        parent_thread_id=parent_thread_id,
        run_kind=run_kind,
        graph_id=graph_id,
        parent_tool=parent_tool,
        status="running",
        created_at=timestamp,
        updated_at=timestamp,
        llm_profile=llm_profile,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_text_atomic(manifest_path, manifest.model_dump_json(indent=2))
    return RunTracker(root, manifest)


def _subagent_suffix(thread_id: str, parent_thread_id: str) -> str:
    """Derive the nested directory suffix from a child thread id."""
    prefix = f"{parent_thread_id}:"
    if thread_id.startswith(prefix):
        suffix = thread_id.removeprefix(prefix)
        if suffix and "/" not in suffix:
            return suffix
    msg = (
        f"Cannot derive subagent suffix from thread_id={thread_id!r} "
        f"and parent_thread_id={parent_thread_id!r}; pass thread_suffix."
    )
    raise ValueError(msg)


def _optional_string(value: object) -> str | None:
    """Return an optional JSON value as a string."""
    return None if value is None else str(value)
