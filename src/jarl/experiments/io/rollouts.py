"""Versioned checkpoint rollout artifact persistence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from pydantic import TypeAdapter

from jarl.envs.navix.telemetry.contracts import (
    NavixCaptureProfile,
    NavixRolloutSummary,
    NavixTraceArrays,
)
from jarl.experiments.io.artifacts import ArtifactRecord
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.io.layout import ROLLOUTS_DIRNAME
from jarl.metadata import now_iso
from jarl.utils import dict_hash, write_text_atomic

ROLLOUT_SCHEMA_VERSION = 1
"""Current checkpoint rollout artifact schema version."""

ROLLOUT_MANIFEST_FILENAME = "rollout.json"
"""Rollout artifact manifest filename."""

ROLLOUT_TRACE_FILENAME = "trace.npz"
"""Compressed rollout trace filename."""

ROLLOUT_VIDEO_FILENAME = "rollout.mp4"
"""Optional rollout video filename."""

_ROLLOUT_ID_HEX_LENGTH = 24
"""Hex characters retained from the stable rollout identity hash."""

_OPTIONAL_TRACE_FIELDS = frozenset(
    {
        "full_symbolic",
        "first_person_symbolic",
        "policy_observations",
        "player_positions",
        "player_directions",
        "player_pockets",
        "rgb_frames",
        "event_happened",
        "event_positions",
        "event_colours",
        "event_types",
    }
)
"""Trace fields omitted when disabled by the capture profile."""

_STATE_TRACE_FIELDS = frozenset(
    {
        "full_symbolic",
        "first_person_symbolic",
        "policy_observations",
        "player_positions",
        "player_directions",
        "player_pockets",
        "rgb_frames",
    }
)
"""Trace arrays aligned to state timesteps ``0..T``."""

_TRANSITION_TRACE_FIELDS = frozenset(
    {
        "actions",
        "rewards",
        "step_types",
        "dones",
        "active_mask",
        "event_happened",
        "event_positions",
        "event_colours",
        "event_types",
    }
)
"""Trace arrays aligned to transitions ``0..T-1``."""

_SUMMARY_ADAPTER: TypeAdapter[NavixRolloutSummary] = TypeAdapter(NavixRolloutSummary)
"""Pydantic adapter for nested rollout summary serialization."""

_PROFILE_ADAPTER: TypeAdapter[NavixCaptureProfile] = TypeAdapter(NavixCaptureProfile)
"""Pydantic adapter for capture profile serialization."""

__all__ = [
    "ROLLOUT_MANIFEST_FILENAME",
    "ROLLOUT_SCHEMA_VERSION",
    "ROLLOUT_TRACE_FILENAME",
    "ROLLOUT_VIDEO_FILENAME",
    "LoadedRolloutArtifact",
    "RolloutArtifactFiles",
    "RolloutArtifactManifest",
    "RolloutArtifactRef",
    "TraceArrayMetadata",
    "WriteRolloutArtifactResult",
    "compute_rollout_id",
    "load_cached_rollout_artifact",
    "load_rollout_artifact",
    "write_rollout_artifact",
]


class RolloutArtifactWorkspace(Protocol):
    """Workspace surface required by the rollout artifact writer."""

    id: str

    @property
    def rollouts_dir(self) -> Path:
        """Return the node rollout directory."""
        ...

    def register_artifact(self, record: ArtifactRecord) -> Path:
        """Register a validated rollout artifact."""
        ...


@dataclass(frozen=True, slots=True)
class RolloutArtifactRef:
    """Reference to one materialized rollout under a node workspace."""

    rollout_id: str
    checkpoint_step: int

    def to_dict(self) -> dict[str, int | str]:
        """Convert the reference to a JSON-serializable dictionary."""
        return {
            "rollout_id": self.rollout_id,
            "checkpoint_step": self.checkpoint_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RolloutArtifactRef:
        """Build a rollout reference from a parsed JSON object."""
        return cls(
            rollout_id=str(data["rollout_id"]),
            checkpoint_step=int(data["checkpoint_step"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TraceArrayMetadata:
    """Name, dtype, and shape of one persisted NPZ array."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert array metadata to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TraceArrayMetadata:
        """Build array metadata from a parsed JSON object."""
        shape = data["shape"]
        if not isinstance(shape, list):
            msg = "Trace array shape must be a JSON list."
            raise TypeError(msg)
        return cls(
            name=str(data["name"]),
            dtype=str(data["dtype"]),
            shape=tuple(int(size) for size in shape),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RolloutArtifactFiles:
    """Relative filenames contained in one rollout materialization."""

    trace: str = ROLLOUT_TRACE_FILENAME
    video: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert file references to a JSON-serializable dictionary."""
        return {"trace": self.trace, "video": self.video}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RolloutArtifactFiles:
        """Build file references from a parsed JSON object."""
        video = data.get("video")
        return cls(
            trace=str(data["trace"]),
            video=None if video is None else str(video),
        )


@dataclass(frozen=True, slots=True)
class RolloutArtifactManifest:
    """Versioned manifest for one persisted checkpoint rollout."""

    schema_version: int
    rollout_id: str
    source_checkpoint: CheckpointRef
    target_node_id: str
    algorithm_name: str
    env_id: str
    seed: int
    max_steps: int
    capture_profile: NavixCaptureProfile
    effective_config: dict[str, object]
    summary: NavixRolloutSummary
    trace_arrays: tuple[TraceArrayMetadata, ...]
    files: RolloutArtifactFiles
    created_at: str

    def to_dict(self) -> dict[str, object]:
        """Convert the manifest to a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "rollout_id": self.rollout_id,
            "source_checkpoint": self.source_checkpoint.to_dict(),
            "target_node_id": self.target_node_id,
            "algorithm_name": self.algorithm_name,
            "env_id": self.env_id,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "capture_profile": _PROFILE_ADAPTER.dump_python(
                self.capture_profile,
                mode="json",
            ),
            "effective_config": self.effective_config,
            "summary": _SUMMARY_ADAPTER.dump_python(self.summary, mode="json"),
            "trace_arrays": [item.to_dict() for item in self.trace_arrays],
            "files": self.files.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RolloutArtifactManifest:
        """Build and validate a manifest from a parsed JSON object."""
        schema_version = int(data["schema_version"])  # type: ignore[arg-type]
        if schema_version != ROLLOUT_SCHEMA_VERSION:
            msg = f"Unsupported rollout schema_version {schema_version}; expected {ROLLOUT_SCHEMA_VERSION}."
            raise ValueError(msg)
        source = data["source_checkpoint"]
        profile = data["capture_profile"]
        config = data["effective_config"]
        summary = data["summary"]
        trace_arrays = data["trace_arrays"]
        files = data["files"]
        if not isinstance(source, dict):
            msg = "Rollout source_checkpoint must be a JSON object."
            raise TypeError(msg)
        if not isinstance(config, dict):
            msg = "Rollout effective_config must be a JSON object."
            raise TypeError(msg)
        if not isinstance(trace_arrays, list):
            msg = "Rollout trace_arrays must be a JSON list."
            raise TypeError(msg)
        if not isinstance(files, dict):
            msg = "Rollout files must be a JSON object."
            raise TypeError(msg)
        return cls(
            schema_version=schema_version,
            rollout_id=str(data["rollout_id"]),
            source_checkpoint=CheckpointRef.from_dict(source),
            target_node_id=str(data["target_node_id"]),
            algorithm_name=str(data["algorithm_name"]),
            env_id=str(data["env_id"]),
            seed=int(data["seed"]),  # type: ignore[arg-type]
            max_steps=int(data["max_steps"]),  # type: ignore[arg-type]
            capture_profile=_PROFILE_ADAPTER.validate_python(profile),
            effective_config={str(key): value for key, value in config.items()},
            summary=_SUMMARY_ADAPTER.validate_python(summary),
            trace_arrays=tuple(TraceArrayMetadata.from_dict(cast(dict[str, object], item)) for item in trace_arrays),
            files=RolloutArtifactFiles.from_dict(files),
            created_at=str(data["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class LoadedRolloutArtifact:
    """Loaded rollout manifest, trace, and resolved file paths."""

    manifest: RolloutArtifactManifest
    trace: NavixTraceArrays
    manifest_path: Path
    trace_path: Path
    video_path: Path | None


@dataclass(frozen=True, slots=True)
class WriteRolloutArtifactResult:
    """Result of materializing or reusing a rollout artifact."""

    ref: RolloutArtifactRef
    manifest: RolloutArtifactManifest
    manifest_path: Path
    trace_path: Path
    video_path: Path | None
    cache_hit: bool


def compute_rollout_id(
    *,
    source_checkpoint: CheckpointRef,
    target_node_id: str,
    algorithm_name: str,
    env_id: str,
    seed: int,
    max_steps: int,
    capture_profile: NavixCaptureProfile,
    effective_config: dict[str, object],
    schema_version: int = ROLLOUT_SCHEMA_VERSION,
) -> str:
    """Return the stable materialization identity for one rollout request."""
    payload = {
        "schema_version": schema_version,
        "source_checkpoint": source_checkpoint.to_dict(),
        "target_node_id": target_node_id,
        "algorithm_name": algorithm_name,
        "env_id": env_id,
        "seed": seed,
        "max_steps": max_steps,
        "capture_profile": _PROFILE_ADAPTER.dump_python(
            capture_profile,
            mode="json",
        ),
        "effective_config": effective_config,
    }
    return dict_hash(payload)[:_ROLLOUT_ID_HEX_LENGTH]


def write_rollout_artifact(
    workspace: RolloutArtifactWorkspace,
    *,
    rollout_id: str,
    source_checkpoint: CheckpointRef,
    algorithm_name: str,
    env_id: str,
    seed: int,
    max_steps: int,
    capture_profile: NavixCaptureProfile,
    effective_config: dict[str, object],
    trace: NavixTraceArrays,
    summary: NavixRolloutSummary,
    episode_index: int = 0,
    video_path: Path | None = None,
) -> WriteRolloutArtifactResult:
    """Atomically persist, validate, and register one rollout materialization."""
    _validate_rollout_id(rollout_id)
    expected_id = compute_rollout_id(
        source_checkpoint=source_checkpoint,
        target_node_id=workspace.id,
        algorithm_name=algorithm_name,
        env_id=env_id,
        seed=seed,
        max_steps=max_steps,
        capture_profile=capture_profile,
        effective_config=effective_config,
    )
    if rollout_id != expected_id:
        msg = f"rollout_id {rollout_id!r} does not match computed identity {expected_id!r}."
        raise ValueError(msg)
    expects_video = capture_profile.rgb_view_mode is not None
    if expects_video != (video_path is not None):
        msg = "video_path must be provided exactly when the capture profile requests RGB."
        raise ValueError(msg)
    if video_path is not None and not video_path.is_file():
        msg = f"Rollout video source does not exist: {video_path}"
        raise FileNotFoundError(msg)

    workspace.rollouts_dir.mkdir(parents=True, exist_ok=True)
    destination = workspace.rollouts_dir / rollout_id
    cached = load_cached_rollout_artifact(
        workspace,
        rollout_id,
        require_video=expects_video,
    )
    if cached is not None:
        return cached
    if destination.exists():
        shutil.rmtree(destination)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{rollout_id}.",
            dir=workspace.rollouts_dir,
        )
    )
    try:
        arrays = _trim_trace_arrays(trace, episode_index)
        trace_path = staging / ROLLOUT_TRACE_FILENAME
        np.savez_compressed(trace_path, **arrays)
        files = RolloutArtifactFiles(video=ROLLOUT_VIDEO_FILENAME if video_path is not None else None)
        if video_path is not None:
            shutil.copy2(video_path, staging / ROLLOUT_VIDEO_FILENAME)
        artifact_paths = (
            f"{ROLLOUTS_DIRNAME}/{rollout_id}/{ROLLOUT_MANIFEST_FILENAME}",
            f"{ROLLOUTS_DIRNAME}/{rollout_id}/{ROLLOUT_TRACE_FILENAME}",
            *((f"{ROLLOUTS_DIRNAME}/{rollout_id}/{ROLLOUT_VIDEO_FILENAME}",) if video_path is not None else ()),
        )
        persisted_summary = replace(
            summary,
            identity=replace(
                summary.identity,
                env_id=env_id,
                seed=seed,
                rollout_id=rollout_id,
                node_id=workspace.id,
                checkpoint_step=source_checkpoint.checkpoint_step,
                algorithm_name=algorithm_name,
            ),
            artifact_paths=artifact_paths,
        )
        manifest = RolloutArtifactManifest(
            schema_version=ROLLOUT_SCHEMA_VERSION,
            rollout_id=rollout_id,
            source_checkpoint=source_checkpoint,
            target_node_id=workspace.id,
            algorithm_name=algorithm_name,
            env_id=env_id,
            seed=seed,
            max_steps=max_steps,
            capture_profile=capture_profile,
            effective_config=dict(effective_config),
            summary=persisted_summary,
            trace_arrays=tuple(
                TraceArrayMetadata(
                    name=name,
                    dtype=str(array.dtype),
                    shape=tuple(int(size) for size in array.shape),
                )
                for name, array in sorted(arrays.items())
            ),
            files=files,
            created_at=now_iso(),
        )
        write_text_atomic(
            staging / ROLLOUT_MANIFEST_FILENAME,
            json.dumps(manifest.to_dict(), indent=2),
        )
        _validate_loaded_rollout(load_rollout_artifact(staging))
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    loaded = load_rollout_artifact(destination)
    _register_rollout(workspace, loaded.manifest)
    return _write_result(loaded, cache_hit=False)


def load_rollout_artifact(path: str | Path) -> LoadedRolloutArtifact:
    """Load and validate a rollout artifact directory or manifest path."""
    source = Path(path)
    manifest_path = source / ROLLOUT_MANIFEST_FILENAME if source.is_dir() else source
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Rollout manifest must be a JSON object: {manifest_path}"
        raise TypeError(msg)
    manifest = RolloutArtifactManifest.from_dict(payload)
    artifact_dir = manifest_path.parent
    trace_path = artifact_dir / manifest.files.trace
    if not trace_path.is_file():
        msg = f"Rollout trace is missing: {trace_path}"
        raise ValueError(msg)
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    values: dict[str, object] = {}
    for field_name in NavixTraceArrays._fields:
        if field_name in arrays:
            values[field_name] = arrays[field_name]
        elif field_name in _OPTIONAL_TRACE_FIELDS:
            values[field_name] = None
        else:
            msg = f"Rollout trace is missing required array {field_name!r}."
            raise ValueError(msg)
    trace = NavixTraceArrays(**values)  # type: ignore[arg-type]
    video_path = artifact_dir / manifest.files.video if manifest.files.video is not None else None
    loaded = LoadedRolloutArtifact(
        manifest=manifest,
        trace=trace,
        manifest_path=manifest_path,
        trace_path=trace_path,
        video_path=video_path,
    )
    _validate_loaded_rollout(loaded)
    return loaded


def load_cached_rollout_artifact(
    workspace: RolloutArtifactWorkspace,
    rollout_id: str,
    *,
    require_video: bool,
) -> WriteRolloutArtifactResult | None:
    """Load and re-register an exact cached rollout materialization."""
    _validate_rollout_id(rollout_id)
    loaded = _load_valid_cache(
        workspace.rollouts_dir / rollout_id,
        rollout_id,
        require_video,
    )
    if loaded is None:
        return None
    _register_rollout(workspace, loaded.manifest)
    return _write_result(loaded, cache_hit=True)


def _trim_trace_arrays(
    trace: NavixTraceArrays,
    episode_index: int,
) -> dict[str, np.ndarray]:
    lengths = np.asarray(trace.episode_lengths)
    if episode_index < 0 or episode_index >= len(lengths):
        msg = f"episode_index {episode_index} is outside batch size {len(lengths)}."
        raise IndexError(msg)
    length = int(lengths[episode_index])
    arrays: dict[str, np.ndarray] = {}
    for field_name in NavixTraceArrays._fields:
        value = getattr(trace, field_name)
        if value is None:
            continue
        array = np.asarray(value)
        if field_name in _STATE_TRACE_FIELDS:
            selected = array[episode_index : episode_index + 1, : length + 1]
        elif field_name in _TRANSITION_TRACE_FIELDS:
            selected = array[episode_index : episode_index + 1, :length]
        else:
            selected = array[episode_index : episode_index + 1]
        arrays[field_name] = selected
    return arrays


def _load_valid_cache(
    destination: Path,
    rollout_id: str,
    expects_video: bool,
) -> LoadedRolloutArtifact | None:
    if not destination.is_dir():
        return None
    try:
        loaded = load_rollout_artifact(destination)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if loaded.manifest.rollout_id != rollout_id:
        return None
    if expects_video != (loaded.video_path is not None):
        return None
    return loaded


def _validate_loaded_rollout(loaded: LoadedRolloutArtifact) -> None:
    manifest = loaded.manifest
    expected_rollout_id = compute_rollout_id(
        source_checkpoint=manifest.source_checkpoint,
        target_node_id=manifest.target_node_id,
        algorithm_name=manifest.algorithm_name,
        env_id=manifest.env_id,
        seed=manifest.seed,
        max_steps=manifest.max_steps,
        capture_profile=manifest.capture_profile,
        effective_config=manifest.effective_config,
        schema_version=manifest.schema_version,
    )
    if manifest.rollout_id != expected_rollout_id:
        msg = "Rollout manifest identity does not match its replay inputs."
        raise ValueError(msg)
    if loaded.video_path is not None and not loaded.video_path.is_file():
        msg = f"Rollout video is missing: {loaded.video_path}"
        raise ValueError(msg)
    expected = {item.name: item for item in loaded.manifest.trace_arrays}
    actual_names = {
        field_name for field_name in NavixTraceArrays._fields if getattr(loaded.trace, field_name) is not None
    }
    if set(expected) != actual_names:
        msg = "Rollout trace arrays do not match manifest metadata."
        raise ValueError(msg)
    for name, metadata in expected.items():
        array = np.asarray(getattr(loaded.trace, name))
        if str(array.dtype) != metadata.dtype or tuple(array.shape) != metadata.shape:
            msg = f"Rollout trace array {name!r} does not match manifest dtype/shape."
            raise ValueError(msg)


def _register_rollout(
    workspace: RolloutArtifactWorkspace,
    manifest: RolloutArtifactManifest,
) -> None:
    workspace.register_artifact(
        ArtifactRecord.rollout(
            manifest.rollout_id,
            manifest.source_checkpoint.checkpoint_step,
            metadata={
                "seed": manifest.seed,
                "env_id": manifest.env_id,
                "algorithm_name": manifest.algorithm_name,
                "has_video": manifest.files.video is not None,
            },
        )
    )


def _write_result(
    loaded: LoadedRolloutArtifact,
    *,
    cache_hit: bool,
) -> WriteRolloutArtifactResult:
    return WriteRolloutArtifactResult(
        ref=RolloutArtifactRef(
            rollout_id=loaded.manifest.rollout_id,
            checkpoint_step=loaded.manifest.source_checkpoint.checkpoint_step,
        ),
        manifest=loaded.manifest,
        manifest_path=loaded.manifest_path,
        trace_path=loaded.trace_path,
        video_path=loaded.video_path,
        cache_hit=cache_hit,
    )


def _validate_rollout_id(rollout_id: str) -> None:
    if (
        len(rollout_id) != _ROLLOUT_ID_HEX_LENGTH
        or not rollout_id.isascii()
        or not all(character in "0123456789abcdef" for character in rollout_id)
    ):
        msg = f"rollout_id must be {_ROLLOUT_ID_HEX_LENGTH} lowercase hex characters."
        raise ValueError(msg)
