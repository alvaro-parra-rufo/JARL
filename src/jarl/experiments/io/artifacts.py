"""Artifact registry IO for node workspaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarl.experiments.io.layout import (
    MODELS_DIRNAME,
    ROLLOUTS_DIRNAME,
    VIDEO_METRICS_JSONL_FILENAME,
    VIDEOS_DIRNAME,
)
from jarl.utils import write_text_atomic

__all__ = [
    "ARTIFACT_KIND_MODEL",
    "ARTIFACT_KIND_ROLLOUT",
    "ARTIFACT_KIND_VIDEO",
    "ARTIFACT_KIND_VIDEO_METRICS",
    "MODEL_ALIAS_BEST",
    "MODEL_ALIAS_FINAL",
    "MODEL_ALIAS_LATEST",
    "ArtifactIdentity",
    "ArtifactRecord",
    "ArtifactRegistry",
    "BestModelArtifactAlias",
    "ModelArtifactAlias",
]

ArtifactIdentity = tuple[str, str, int]
"""Artifact identity as ``(kind, name, step)``."""

ARTIFACT_KIND_VIDEO = "video"
"""Artifact kind for rendered rollout videos under ``videos/``."""

ARTIFACT_KIND_VIDEO_METRICS = "video_metrics"
"""Artifact kind for ``video_metrics.jsonl``."""

ARTIFACT_KIND_MODEL = "model"
"""Artifact kind reserved for model archives under ``models/`` (Phase 4)."""

ARTIFACT_KIND_ROLLOUT = "rollout"
"""Artifact kind for materialized checkpoint rollouts under ``rollouts/``."""

MODEL_ALIAS_LATEST = "latest"
"""Alias pointer to the latest canonical model archive."""

MODEL_ALIAS_FINAL = "final"
"""Alias pointer to the final canonical model archive after a successful run."""

MODEL_ALIAS_BEST = "best"
"""Alias pointer to a caller-promoted best canonical model archive."""


@dataclass(frozen=True, slots=True)
class ModelArtifactAlias:
    """Pointer from a model alias to a canonical ``(name, step)`` archive.

    Args:
        name: Canonical model identifier.
        step: Training step associated with the canonical archive.
    """

    name: str
    step: int

    def to_dict(self) -> dict[str, str | int]:
        """Convert the alias pointer to a JSON-serializable dictionary."""
        return {"name": self.name, "step": self.step}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelArtifactAlias:
        """Build an alias pointer from a parsed JSON object."""
        return cls(name=str(data["name"]), step=int(data["step"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BestModelArtifactAlias:
    """Promotion metadata for the ``best`` model alias pointer.

    Args:
        name: Canonical model identifier.
        step: Training step associated with the canonical archive.
        metric_name: Metric used for the promotion decision.
        metric_value: Metric value recorded with the promotion.
        reason: Caller-provided explanation for the promotion.
    """

    name: str
    step: int
    metric_name: str
    metric_value: float
    reason: str

    def to_dict(self) -> dict[str, str | int | float]:
        """Convert the alias metadata to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "step": self.step,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BestModelArtifactAlias:
        """Build promotion metadata from a parsed JSON object."""
        return cls(
            name=str(data["name"]),
            step=int(data["step"]),  # type: ignore[arg-type]
            metric_name=str(data["metric_name"]),
            metric_value=float(data["metric_value"]),  # type: ignore[arg-type]
            reason=str(data["reason"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Exportable artifact entry persisted in ``artifacts.json``.

    Args:
        name: Human-readable artifact identifier within ``kind``.
        kind: Artifact category (for example ``video`` or ``model``).
        relative_path: Path relative to the node root.
        step: Training step associated with the artifact.
        metadata: Optional extra metadata for export or display.
    """

    name: str
    kind: str
    relative_path: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> ArtifactIdentity:
        """Return the stable artifact identity ``(kind, name, step)``."""
        return (self.kind, self.name, self.step)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "kind": self.kind,
            "relative_path": self.relative_path,
            "step": self.step,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ArtifactRecord:
        """Build a record from a parsed JSON object.

        Args:
            data: Parsed JSON object for one artifact entry.

        Returns:
            Parsed `ArtifactRecord`.

        Raises:
            KeyError: If required fields are missing.
            TypeError: If field types are invalid.
        """
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            msg = "Artifact metadata must be a JSON object."
            raise TypeError(msg)
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            relative_path=str(data["relative_path"]),
            step=int(data["step"]),  # type: ignore[arg-type]
            metadata={str(key): value for key, value in metadata.items()},
        )

    @classmethod
    def video(
        cls, name: str, step: int, *, filename: str | None = None, metadata: dict[str, Any] | None = None
    ) -> ArtifactRecord:
        """Build a video artifact record under ``videos/``.

        Args:
            name: Video identifier used in the registry.
            step: Training step when the video was recorded.
            filename: Optional file name under ``videos/``. Defaults to ``{name}.mp4``.
            metadata: Optional extra metadata.

        Returns:
            Video artifact record with a ``videos/`` relative path.
        """
        video_filename = filename or f"{name}.mp4"
        return cls(
            name=name,
            kind=ARTIFACT_KIND_VIDEO,
            relative_path=f"{VIDEOS_DIRNAME}/{video_filename}",
            step=step,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def video_metrics(cls, step: int, *, metadata: dict[str, Any] | None = None) -> ArtifactRecord:
        """Build a record for the node's ``video_metrics.jsonl`` file.

        Args:
            step: Training step associated with the metrics snapshot.
            metadata: Optional extra metadata.

        Returns:
            Video-metrics artifact record pointing at ``video_metrics.jsonl``.
        """
        return cls(
            name=VIDEO_METRICS_JSONL_FILENAME,
            kind=ARTIFACT_KIND_VIDEO_METRICS,
            relative_path=VIDEO_METRICS_JSONL_FILENAME,
            step=step,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def model(cls, name: str, step: int, *, filename: str, metadata: dict[str, Any] | None = None) -> ArtifactRecord:
        """Build a model artifact record under ``models/``.

        Args:
            name: Model identifier used in the registry.
            step: Training step when the model was saved.
            filename: File name under ``models/``.
            metadata: Optional extra metadata.

        Returns:
            Model artifact record with a ``models/`` relative path.
        """
        return cls(
            name=name,
            kind=ARTIFACT_KIND_MODEL,
            relative_path=f"{MODELS_DIRNAME}/{filename}",
            step=step,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def rollout(
        cls,
        rollout_id: str,
        step: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Build a rollout artifact record pointing at its manifest."""
        return cls(
            name=rollout_id,
            kind=ARTIFACT_KIND_ROLLOUT,
            relative_path=f"{ROLLOUTS_DIRNAME}/{rollout_id}/rollout.json",
            step=step,
            metadata=dict(metadata or {}),
        )


class ArtifactRegistry:
    """In-memory artifact registry backed by ``artifacts.json``.

    Args:
        path: Registry file path (typically ``artifacts.json``).
        root: Node root used to resolve ``relative_path`` values.
    """

    def __init__(self, path: str | Path, *, root: str | Path) -> None:
        """Initialize an empty or disk-backed registry."""
        self._path = Path(path)
        self._root = Path(root)
        self._records: dict[ArtifactIdentity, ArtifactRecord] = {}
        self._model_aliases: dict[str, ModelArtifactAlias | BestModelArtifactAlias] = {}

    @property
    def path(self) -> Path:
        """Registry file path."""
        return self._path

    @property
    def root(self) -> Path:
        """Node root used to resolve artifact paths."""
        return self._root

    def load(self) -> None:
        """Load registry entries from disk when the file exists."""
        self._records.clear()
        self._model_aliases.clear()
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            self._load_artifact_entries(data)
            return
        if not isinstance(data, dict):
            msg = f"Artifact registry must be a JSON list or object: {self._path}"
            raise TypeError(msg)
        artifacts = data.get("artifacts", [])
        if not isinstance(artifacts, list):
            msg = f"Artifact registry artifacts must be a JSON list: {self._path}"
            raise TypeError(msg)
        self._load_artifact_entries(artifacts)
        aliases = data.get("model_aliases", {})
        if not isinstance(aliases, dict):
            msg = f"Artifact registry model_aliases must be a JSON object: {self._path}"
            raise TypeError(msg)
        for alias_name, value in aliases.items():
            alias_key = str(alias_name)
            if alias_key == MODEL_ALIAS_BEST:
                if not isinstance(value, dict):
                    msg = "Best model alias must be a JSON object."
                    raise TypeError(msg)
                self._model_aliases[alias_key] = BestModelArtifactAlias.from_dict(value)
            else:
                if not isinstance(value, dict):
                    msg = "Model alias pointers must be JSON objects."
                    raise TypeError(msg)
                self._model_aliases[alias_key] = ModelArtifactAlias.from_dict(value)
        self._validate_loaded_model_aliases()

    def save(self) -> Path:
        """Persist the registry atomically to disk.

        Returns:
            Path to the written registry file.
        """
        payload: dict[str, object] = {
            "artifacts": [record.to_dict() for record in self._sorted_records()],
            "model_aliases": self._serialize_model_aliases(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return write_text_atomic(self._path, json.dumps(payload, indent=2, default=str))

    def register(self, record: ArtifactRecord) -> None:
        """Register or replace an artifact by ``(kind, name, step)``.

        Args:
            record: Artifact entry to store.

        Raises:
            ValueError: If ``relative_path`` is absolute or escapes the node root.
        """
        _resolve_under_root(self._root, record.relative_path)
        self._records[record.identity] = record

    def list(self) -> list[ArtifactRecord]:
        """Return all registered artifacts in stable sort order."""
        return self._sorted_records()

    def get(self, kind: str, name: str, step: int) -> ArtifactRecord | None:
        """Return one artifact by identity when present.

        Args:
            kind: Artifact category.
            name: Artifact identifier within ``kind``.
            step: Training step associated with the artifact.

        Returns:
            Matching record, or ``None`` when not registered.
        """
        return self._records.get((kind, name, step))

    def resolve_path(self, kind: str, name: str, step: int) -> Path | None:
        """Resolve the absolute path for a registered artifact.

        Args:
            kind: Artifact category.
            name: Artifact identifier within ``kind``.
            step: Training step associated with the artifact.

        Returns:
            Absolute path under the node root, or ``None`` when not registered.

        Raises:
            ValueError: If the stored relative path escapes the node root.
        """
        record = self.get(kind, name, step)
        if record is None:
            return None
        return _resolve_under_root(self._root, record.relative_path)

    def set_model_alias(self, alias: str, *, name: str, step: int) -> None:
        """Point a model alias at a canonical ``(name, step)`` archive.

        Args:
            alias: Alias name such as ``latest`` or ``final``.
            name: Canonical model identifier.
            step: Training step associated with the canonical archive.

        Raises:
            KeyError: If the canonical artifact does not exist.
        """
        self._require_model_record(name, step)
        self._model_aliases[alias] = ModelArtifactAlias(name=name, step=step)

    def promote_model_best(
        self,
        *,
        name: str,
        step: int,
        metric_name: str,
        metric_value: float,
        reason: str,
    ) -> None:
        """Point the ``best`` alias at a canonical archive with promotion metadata.

        Args:
            name: Canonical model identifier.
            step: Training step associated with the canonical archive.
            metric_name: Metric used for the promotion decision.
            metric_value: Metric value recorded with the promotion.
            reason: Caller-provided explanation for the promotion.

        Raises:
            KeyError: If the canonical artifact does not exist.
        """
        self._require_model_record(name, step)
        self._model_aliases[MODEL_ALIAS_BEST] = BestModelArtifactAlias(
            name=name,
            step=step,
            metric_name=metric_name,
            metric_value=metric_value,
            reason=reason,
        )

    def resolve_model_alias(self, alias: str) -> ArtifactRecord | None:
        """Resolve a model alias pointer to its canonical artifact record.

        Args:
            alias: Alias name such as ``latest``, ``final``, or ``best``.

        Returns:
            Canonical artifact record, or ``None`` when the alias is unset.

        Raises:
            ValueError: If the alias is set but points to a missing artifact.
        """
        pointer = self._model_aliases.get(alias)
        if pointer is None:
            return None
        return self._resolve_model_alias_target(alias, pointer)

    def _load_artifact_entries(self, entries: list[object]) -> None:
        for item in entries:
            if not isinstance(item, dict):
                msg = f"Artifact registry entries must be JSON objects: {self._path}"
                raise TypeError(msg)
            record = ArtifactRecord.from_dict(item)
            self._records[record.identity] = record

    def _require_model_record(self, name: str, step: int) -> ArtifactRecord:
        record = self.get(ARTIFACT_KIND_MODEL, name, step)
        if record is None:
            msg = f"No model artifact registered for identity ({ARTIFACT_KIND_MODEL!r}, {name!r}, {step})"
            raise KeyError(msg)
        return record

    def _resolve_model_alias_target(
        self,
        alias: str,
        pointer: ModelArtifactAlias | BestModelArtifactAlias,
    ) -> ArtifactRecord:
        record = self.get(ARTIFACT_KIND_MODEL, pointer.name, pointer.step)
        if record is None:
            msg = (
                f"Model alias {alias!r} points to missing artifact "
                f"({ARTIFACT_KIND_MODEL!r}, {pointer.name!r}, {pointer.step})"
            )
            raise ValueError(msg)
        return record

    def _validate_loaded_model_aliases(self) -> None:
        for alias_name, pointer in self._model_aliases.items():
            self._resolve_model_alias_target(alias_name, pointer)

    def _serialize_model_aliases(self) -> dict[str, object]:
        serialized: dict[str, object] = {}
        for alias_name, pointer in self._model_aliases.items():
            serialized[alias_name] = pointer.to_dict()
        return serialized

    def _sorted_records(self) -> list[ArtifactRecord]:
        return sorted(self._records.values(), key=lambda record: record.identity)


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    """Resolve a node-relative path and reject escapes outside ``root``."""
    candidate = Path(relative_path)
    if candidate.is_absolute():
        msg = f"Artifact relative_path must be relative to the node root: {relative_path!r}"
        raise ValueError(msg)
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        msg = f"Artifact relative_path escapes the node root: {relative_path!r}"
        raise ValueError(msg)
    return resolved
