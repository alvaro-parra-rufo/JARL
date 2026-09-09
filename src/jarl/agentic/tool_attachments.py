"""Discover displayable artifacts inside compact tool JSON payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jarl.experiments.io.layout import ExperimentLayout
from jarl.experiments.io.rollouts import ROLLOUT_VIDEO_FILENAME

__all__ = [
    "ToolResultAttachment",
    "extract_tool_attachments",
    "register_attachment_extractor",
    "resolve_attachment_path",
]

AttachmentKind = Literal["video", "file"]
"""Supported attachment kinds for UI renderers."""

AttachmentExtractor = Callable[[Mapping[str, object]], tuple["ToolResultAttachment", ...]]
"""Callable that returns attachments for one parsed tool payload."""


@dataclass(frozen=True, slots=True)
class ToolResultAttachment:
    """One artifact referenced by a tool result."""

    kind: AttachmentKind
    relative_path: str
    label: str = ""


_ATTACHMENT_EXTRACTORS: dict[str, AttachmentExtractor] = {}
"""Per-tool attachment extractors keyed by tool name."""


def register_attachment_extractor(tool_name: str, extractor: AttachmentExtractor) -> None:
    """Register a tool-specific attachment extractor."""
    if not tool_name.strip():
        raise ValueError("Tool name must not be empty.")
    _ATTACHMENT_EXTRACTORS[tool_name] = extractor


def extract_tool_attachments(
    tool_name: str | None,
    payload: Mapping[str, object],
) -> tuple[ToolResultAttachment, ...]:
    """Return displayable attachments declared by a compact tool payload."""
    if tool_name and tool_name in _ATTACHMENT_EXTRACTORS:
        return _ATTACHMENT_EXTRACTORS[tool_name](payload)
    return _extract_generic_attachments(payload)


def resolve_attachment_path(
    experiment_dir: Path,
    payload: Mapping[str, object],
    relative_path: str,
) -> Path | None:
    """Resolve a node-relative artifact path under ``experiment_dir``."""
    if not relative_path.strip():
        return None
    layout = ExperimentLayout(experiment_dir)
    node_id = _payload_node_id(payload)
    if node_id is None:
        return None
    candidate = layout.node_dir(node_id) / relative_path
    return candidate if candidate.is_file() else None


def _extract_generic_attachments(payload: Mapping[str, object]) -> tuple[ToolResultAttachment, ...]:
    paths = _collect_artifact_paths(payload)
    attachments: list[ToolResultAttachment] = []
    for relative_path in paths:
        attachment = _attachment_from_relative_path(relative_path)
        if attachment is not None:
            attachments.append(attachment)
    return tuple(attachments)


def _attachment_from_relative_path(relative_path: str) -> ToolResultAttachment | None:
    lowered = relative_path.lower()
    if lowered.endswith(".mp4"):
        return ToolResultAttachment(
            kind="video",
            relative_path=relative_path,
            label=Path(relative_path).name,
        )
    return None


def _collect_artifact_paths(payload: Mapping[str, object]) -> tuple[str, ...]:
    paths: list[str] = []
    raw_paths = payload.get("artifact_paths")
    if isinstance(raw_paths, (list, tuple)):
        paths.extend(item for item in raw_paths if isinstance(item, str))
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        nested_paths = summary.get("artifact_paths")
        if isinstance(nested_paths, (list, tuple)):
            paths.extend(item for item in nested_paths if isinstance(item, str))
    return tuple(dict.fromkeys(paths))


def _payload_node_id(payload: Mapping[str, object]) -> str | None:
    identity = payload.get("identity")
    if isinstance(identity, Mapping):
        node_id = identity.get("node_id")
        if isinstance(node_id, str) and node_id:
            return node_id
    node_id = payload.get("node_id")
    if isinstance(node_id, str) and node_id:
        return node_id
    return None


def _graph_checkpoint_rollout_attachments(
    payload: Mapping[str, object],
) -> tuple[ToolResultAttachment, ...]:
    attachments = list(_extract_generic_attachments(payload))
    if any(item.relative_path.endswith(ROLLOUT_VIDEO_FILENAME) for item in attachments):
        return tuple(attachments)
    return tuple(attachments)


register_attachment_extractor("graph_checkpoint_rollout", _graph_checkpoint_rollout_attachments)
