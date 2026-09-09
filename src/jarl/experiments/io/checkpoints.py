"""Checkpoint metadata registry and references for node workspaces."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarl.experiments.node import NodeWorkspace

from jarl.experiments.io.layout import CHECKPOINT_DIRNAME
from jarl.metadata import now_iso
from jarl.utils import write_text_atomic

__all__ = [
    "CHECKPOINT_ALIAS_BEST",
    "CHECKPOINT_ALIAS_FINAL",
    "CHECKPOINT_ALIAS_LATEST",
    "CHECKPOINT_BEST_LENGTH_METRIC",
    "CHECKPOINT_BEST_RETURN_METRIC",
    "BestCheckpointAlias",
    "CheckpointOrigin",
    "CheckpointRecord",
    "CheckpointRef",
    "CheckpointRegistry",
    "CheckpointStatus",
    "checkpoint_ref_from_alias",
    "select_best_checkpoint",
]

CHECKPOINT_ALIAS_LATEST = "latest"
"""Alias for the most recently saved checkpoint."""

CHECKPOINT_ALIAS_FINAL = "final"
"""Alias for the checkpoint active when a training run completes."""

CHECKPOINT_ALIAS_BEST = "best"
"""Alias for the checkpoint selected by `select_best_checkpoint`."""

CHECKPOINT_BEST_RETURN_METRIC = "eval/episode_return"
"""Primary metric maximized when resolving the ``best`` checkpoint alias."""

CHECKPOINT_BEST_LENGTH_METRIC = "eval/episode_length"
"""Secondary metric minimized on return ties when resolving ``best``."""


class CheckpointStatus(StrEnum):
    """Persistence status for a checkpoint record."""

    SAVED = "saved"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Read-only reference to a checkpoint stored under another node.

    Args:
        node_id: Identifier of the node that owns the checkpoint.
        checkpoint_step: Orbax step identifier within that node.
    """

    node_id: str
    checkpoint_step: int

    def to_dict(self) -> dict[str, int | str]:
        """Convert the reference to a JSON-serializable dictionary."""
        return {"node_id": self.node_id, "checkpoint_step": self.checkpoint_step}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointRef:
        """Build a reference from a parsed JSON object.

        Args:
            data: Parsed JSON object with ``node_id`` and ``checkpoint_step``.

        Returns:
            Parsed `CheckpointRef`.
        """
        return cls(node_id=str(data["node_id"]), checkpoint_step=int(data["checkpoint_step"]))  # type: ignore[arg-type]


def checkpoint_ref_from_alias(workspace: NodeWorkspace, alias: str) -> CheckpointRef | None:
    """Resolve a checkpoint alias to a ``CheckpointRef``, if available."""
    if alias not in {CHECKPOINT_ALIAS_LATEST, CHECKPOINT_ALIAS_BEST, CHECKPOINT_ALIAS_FINAL}:
        return None
    record = workspace.resolve_checkpoint_alias(alias)
    if record is None:
        return None
    return CheckpointRef(node_id=workspace.id, checkpoint_step=record.checkpoint_step)


def select_best_checkpoint(records: Sequence[CheckpointRecord]) -> CheckpointRecord | None:
    """Select the best saved checkpoint under the canonical ``best`` policy.

    Ranking is deterministic:

    1. maximize `CHECKPOINT_BEST_RETURN_METRIC`
    2. on ties, minimize `CHECKPOINT_BEST_LENGTH_METRIC` (missing length sorts last)
    3. on remaining ties, maximize `checkpoint_step`

    Args:
        records: Checkpoint records to consider.

    Returns:
        Winning saved record with a return metric, or ``None`` when none qualify.
    """
    candidates = [
        record
        for record in records
        if record.status == CheckpointStatus.SAVED and CHECKPOINT_BEST_RETURN_METRIC in record.metrics
    ]
    if not candidates:
        return None
    return max(candidates, key=_best_checkpoint_sort_key)


def _best_checkpoint_sort_key(record: CheckpointRecord) -> tuple[float, float, int]:
    """Return a sortable key where larger values are better under the ``best`` policy."""
    episode_return = float(record.metrics[CHECKPOINT_BEST_RETURN_METRIC])
    raw_length = record.metrics.get(CHECKPOINT_BEST_LENGTH_METRIC)
    # Negate length so shorter episodes rank higher; missing length loses length ties.
    length_score = -float(raw_length) if raw_length is not None else float("-inf")
    return (episode_return, length_score, record.checkpoint_step)


@dataclass(frozen=True, slots=True)
class CheckpointOrigin:
    """Lineage origin for a checkpoint loaded from a parent node.

    Args:
        node_id: Parent node identifier.
        checkpoint_step: Orbax step restored from the parent.
    """

    node_id: str
    checkpoint_step: int

    def to_dict(self) -> dict[str, int | str]:
        """Convert the origin to a JSON-serializable dictionary."""
        return {"node_id": self.node_id, "checkpoint_step": self.checkpoint_step}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointOrigin:
        """Build an origin from a parsed JSON object.

        Args:
            data: Parsed JSON object with ``node_id`` and ``checkpoint_step``.

        Returns:
            Parsed `CheckpointOrigin`.
        """
        return cls(node_id=str(data["node_id"]), checkpoint_step=int(data["checkpoint_step"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Typed checkpoint metadata persisted in ``checkpoints.json``.

    Args:
        node_step: Node-relative training step when the checkpoint was saved.
        checkpoint_step: Orbax step identifier in ``checkpoint/``.
        status: Persistence status for the checkpoint entry.
        relative_path: Path relative to the node root.
        created_at: ISO-8601 timestamp when the record was written.
        metrics: Scalar metrics snapshot captured at save time.
        global_step: Optional global training step supplied by the caller.
        optimizer_updates: Optional optimizer update counter supplied by the caller.
        origin: Optional parent checkpoint this node initially restored from.
    """

    node_step: int
    checkpoint_step: int
    status: CheckpointStatus
    relative_path: str
    created_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    global_step: int | None = None
    optimizer_updates: int | None = None
    origin: CheckpointOrigin | None = None

    @classmethod
    def for_step(
        cls,
        *,
        node_step: int,
        checkpoint_step: int,
        metrics: dict[str, float] | None = None,
        global_step: int | None = None,
        optimizer_updates: int | None = None,
        origin: CheckpointOrigin | None = None,
        status: CheckpointStatus = CheckpointStatus.SAVED,
    ) -> CheckpointRecord:
        """Build a saved checkpoint record with the standard relative path.

        Args:
            node_step: Node-relative training step when the checkpoint was saved.
            checkpoint_step: Orbax step identifier in ``checkpoint/``.
            metrics: Scalar metrics snapshot captured at save time.
            global_step: Optional global training step supplied by the caller.
            optimizer_updates: Optional optimizer update counter supplied by the caller.
            origin: Optional parent checkpoint this node initially restored from.
            status: Persistence status for the checkpoint entry.

        Returns:
            Checkpoint record pointing at ``checkpoint/{checkpoint_step}``.
        """
        return cls(
            node_step=node_step,
            checkpoint_step=checkpoint_step,
            status=status,
            relative_path=f"{CHECKPOINT_DIRNAME}/{checkpoint_step}",
            created_at=now_iso(),
            metrics=dict(metrics or {}),
            global_step=global_step,
            optimizer_updates=optimizer_updates,
            origin=origin,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        payload: dict[str, Any] = {
            "node_step": self.node_step,
            "checkpoint_step": self.checkpoint_step,
            "status": self.status.value,
            "relative_path": self.relative_path,
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
        }
        if self.global_step is not None:
            payload["global_step"] = self.global_step
        if self.optimizer_updates is not None:
            payload["optimizer_updates"] = self.optimizer_updates
        if self.origin is not None:
            payload["origin"] = self.origin.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointRecord:
        """Build a record from a parsed JSON object.

        Args:
            data: Parsed JSON object for one checkpoint entry.

        Returns:
            Parsed `CheckpointRecord`.
        """
        metrics_raw = data.get("metrics", {})
        if not isinstance(metrics_raw, dict):
            msg = "Checkpoint metrics must be a JSON object."
            raise TypeError(msg)
        metrics = {str(key): float(value) for key, value in metrics_raw.items()}
        origin_raw = data.get("origin")
        origin = None
        if origin_raw is not None:
            if not isinstance(origin_raw, dict):
                msg = "Checkpoint origin must be a JSON object."
                raise TypeError(msg)
            origin = CheckpointOrigin.from_dict(origin_raw)
        global_step = data.get("global_step")
        optimizer_updates = data.get("optimizer_updates")
        return cls(
            node_step=int(data["node_step"]),  # type: ignore[arg-type]
            checkpoint_step=int(data["checkpoint_step"]),  # type: ignore[arg-type]
            status=CheckpointStatus(str(data["status"])),
            relative_path=str(data["relative_path"]),
            created_at=str(data["created_at"]),
            metrics=metrics,
            global_step=int(global_step) if global_step is not None else None,  # type: ignore[arg-type]
            optimizer_updates=int(optimizer_updates) if optimizer_updates is not None else None,  # type: ignore[arg-type]
            origin=origin,
        )


@dataclass(frozen=True, slots=True)
class BestCheckpointAlias:
    """Promotion metadata for the ``best`` checkpoint alias.

    Args:
        checkpoint_step: Orbax step promoted as best.
        metric_name: Metric used for the promotion decision.
        metric_value: Metric value recorded with the promotion.
        reason: Caller-provided explanation for the promotion.
    """

    checkpoint_step: int
    metric_name: str
    metric_value: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the alias metadata to a JSON-serializable dictionary."""
        return {
            "checkpoint_step": self.checkpoint_step,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BestCheckpointAlias:
        """Build promotion metadata from a parsed JSON object.

        Args:
            data: Parsed JSON object for a ``best`` alias entry.

        Returns:
            Parsed `BestCheckpointAlias`.
        """
        return cls(
            checkpoint_step=int(data["checkpoint_step"]),  # type: ignore[arg-type]
            metric_name=str(data["metric_name"]),
            metric_value=float(data["metric_value"]),  # type: ignore[arg-type]
            reason=str(data["reason"]),
        )


class CheckpointRegistry:
    """In-memory checkpoint index backed by ``checkpoints.json``.

    Args:
        path: Registry file path (typically ``checkpoints.json``).
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize an empty or disk-backed checkpoint registry."""
        self._path = Path(path)
        self._records: dict[int, CheckpointRecord] = {}
        self._aliases: dict[str, int | BestCheckpointAlias] = {}
        self._pinned_checkpoint_steps: set[int] = set()

    @property
    def path(self) -> Path:
        """Registry file path."""
        return self._path

    def load(self) -> None:
        """Load registry entries from disk when the file exists."""
        self._records.clear()
        self._aliases.clear()
        self._pinned_checkpoint_steps.clear()
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            msg = f"Checkpoint registry must be a JSON object: {self._path}"
            raise TypeError(msg)
        checkpoints = data.get("checkpoints", [])
        if not isinstance(checkpoints, list):
            msg = f"Checkpoint registry checkpoints must be a JSON list: {self._path}"
            raise TypeError(msg)
        for item in checkpoints:
            if not isinstance(item, dict):
                msg = f"Checkpoint registry entries must be JSON objects: {self._path}"
                raise TypeError(msg)
            record = CheckpointRecord.from_dict(item)
            self._records[record.checkpoint_step] = record
        aliases = data.get("aliases", {})
        if not isinstance(aliases, dict):
            msg = f"Checkpoint registry aliases must be a JSON object: {self._path}"
            raise TypeError(msg)
        for name, value in aliases.items():
            alias_name = str(name)
            if alias_name == CHECKPOINT_ALIAS_BEST:
                if not isinstance(value, dict):
                    msg = "Best checkpoint alias must be a JSON object."
                    raise TypeError(msg)
                self._aliases[alias_name] = BestCheckpointAlias.from_dict(value)
            else:
                self._aliases[alias_name] = int(value)  # type: ignore[arg-type]
        pinned = data.get("pinned_checkpoint_steps", [])
        if not isinstance(pinned, list):
            msg = f"Pinned checkpoint steps must be a JSON list: {self._path}"
            raise TypeError(msg)
        self._pinned_checkpoint_steps = {int(step) for step in pinned}
        self._validate_loaded_aliases()
        self._validate_pinned_steps()

    def save(self) -> Path:
        """Persist the registry atomically to disk.

        Returns:
            Path to the written registry file.
        """
        payload: dict[str, Any] = {
            "checkpoints": [record.to_dict() for record in self._sorted_records()],
            "aliases": self._serialize_aliases(),
            "pinned_checkpoint_steps": sorted(self._pinned_checkpoint_steps),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return write_text_atomic(self._path, json.dumps(payload, indent=2, default=str))

    def register(self, record: CheckpointRecord) -> None:
        """Register or replace a checkpoint record by ``checkpoint_step``.

        Args:
            record: Checkpoint metadata entry to store.
        """
        self._records[record.checkpoint_step] = record

    def list(self) -> list[CheckpointRecord]:
        """Return all checkpoint records in stable sort order."""
        return self._sorted_records()

    def get(self, checkpoint_step: int) -> CheckpointRecord | None:
        """Return one checkpoint record by Orbax step when present.

        Args:
            checkpoint_step: Orbax step identifier.

        Returns:
            Matching record, or ``None`` when not registered.
        """
        return self._records.get(checkpoint_step)

    def pin(self, checkpoint_step: int) -> None:
        """Mark a checkpoint step as pinned so retention policies can preserve it.

        Args:
            checkpoint_step: Orbax step identifier to pin.

        Raises:
            KeyError: If the checkpoint record does not exist or is not saved.
        """
        self.require_saved(checkpoint_step)
        self._pinned_checkpoint_steps.add(checkpoint_step)

    @property
    def pinned_checkpoint_steps(self) -> frozenset[int]:
        """Pinned Orbax steps referenced by downstream nodes."""
        return frozenset(self._pinned_checkpoint_steps)

    def require_saved(self, checkpoint_step: int) -> CheckpointRecord:
        """Return a saved checkpoint record or raise when it is missing.

        Args:
            checkpoint_step: Orbax step identifier.

        Returns:
            Matching saved checkpoint record.

        Raises:
            KeyError: If the record is missing or not in ``saved`` status.
        """
        record = self.get(checkpoint_step)
        if record is None:
            msg = f"No checkpoint record registered for step {checkpoint_step}"
            raise KeyError(msg)
        if record.status != CheckpointStatus.SAVED:
            msg = f"Checkpoint step {checkpoint_step} is not saved (status={record.status.value!r})"
            raise KeyError(msg)
        return record

    def set_alias_latest(self, checkpoint_step: int) -> None:
        """Update the ``latest`` alias to ``checkpoint_step``.

        Args:
            checkpoint_step: Orbax step identifier to reference.

        Raises:
            KeyError: If the checkpoint record does not exist or is not saved.
        """
        self.require_saved(checkpoint_step)
        self._aliases[CHECKPOINT_ALIAS_LATEST] = checkpoint_step

    def set_alias_final(self, checkpoint_step: int) -> None:
        """Update the ``final`` alias to ``checkpoint_step``.

        Args:
            checkpoint_step: Orbax step identifier to reference.

        Raises:
            KeyError: If the checkpoint record does not exist or is not saved.
        """
        self.require_saved(checkpoint_step)
        self._aliases[CHECKPOINT_ALIAS_FINAL] = checkpoint_step

    def promote_best(
        self,
        checkpoint_step: int,
        *,
        metric_name: str,
        metric_value: float,
        reason: str,
    ) -> None:
        """Promote a checkpoint to the ``best`` alias with caller metadata.

        Args:
            checkpoint_step: Orbax step identifier to promote.
            metric_name: Metric used for the promotion decision.
            metric_value: Metric value recorded with the promotion.
            reason: Caller-provided explanation for the promotion.

        Raises:
            KeyError: If the checkpoint record does not exist or is not saved.
        """
        self.require_saved(checkpoint_step)
        self._aliases[CHECKPOINT_ALIAS_BEST] = BestCheckpointAlias(
            checkpoint_step=checkpoint_step,
            metric_name=metric_name,
            metric_value=metric_value,
            reason=reason,
        )

    def resolve_alias(self, alias: str) -> CheckpointRecord | None:
        """Resolve a named alias to a checkpoint record when present.

        Args:
            alias: Alias name such as ``latest``, ``final``, or ``best``.

        Returns:
            Matching checkpoint record, or ``None`` when the alias is unset.

        Raises:
            ValueError: If the alias is set but points to a missing checkpoint.
        """
        if alias not in self._aliases:
            return None
        return self._resolve_alias_target(alias)

    def get_alias_step(self, alias: str) -> int | None:
        """Return the Orbax step referenced by an alias when present.

        Args:
            alias: Alias name such as ``latest``, ``final``, or ``best``.

        Returns:
            Orbax step identifier, or ``None`` when the alias is unset.
        """
        value = self._aliases.get(alias)
        if value is None:
            return None
        if isinstance(value, BestCheckpointAlias):
            return value.checkpoint_step
        return int(value)

    def _alias_checkpoint_step(self, value: int | BestCheckpointAlias) -> int:
        if isinstance(value, BestCheckpointAlias):
            return value.checkpoint_step
        return int(value)

    def _resolve_alias_target(self, alias: str) -> CheckpointRecord:
        value = self._aliases[alias]
        checkpoint_step = self._alias_checkpoint_step(value)
        record = self.get(checkpoint_step)
        if record is None:
            msg = f"Checkpoint alias {alias!r} points to missing step {checkpoint_step}"
            raise ValueError(msg)
        return record

    def _validate_loaded_aliases(self) -> None:
        for alias_name in self._aliases:
            self._resolve_alias_target(alias_name)

    def _validate_pinned_steps(self) -> None:
        for checkpoint_step in self._pinned_checkpoint_steps:
            if self.get(checkpoint_step) is None:
                msg = f"Pinned checkpoint step {checkpoint_step} is not registered"
                raise ValueError(msg)

    def _sorted_records(self) -> list[CheckpointRecord]:
        return sorted(self._records.values(), key=lambda record: record.checkpoint_step)

    def _serialize_aliases(self) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for name, value in self._aliases.items():
            if isinstance(value, BestCheckpointAlias):
                serialized[name] = value.to_dict()
            else:
                serialized[name] = int(value)
        return serialized
