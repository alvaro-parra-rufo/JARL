"""Model archive IO for exportable agent weights."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp

from jarl.experiments.io.artifacts import (
    MODEL_ALIAS_BEST,
    MODEL_ALIAS_FINAL,
    MODEL_ALIAS_LATEST,
    ArtifactRecord,
)
from jarl.utils import write_text_atomic

__all__ = [
    "MODEL_ALIAS_BEST",
    "MODEL_ALIAS_FINAL",
    "MODEL_ALIAS_LATEST",
    "MODEL_ARCHIVE_FORMAT_VERSION",
    "MODEL_MANIFEST_FILENAME",
    "canonical_model_filename",
    "load_model_archive",
    "make_model_archive_save_callback",
    "save_model_archive",
]

MODEL_ARCHIVE_FORMAT_VERSION = 1
"""Current jarl-native model archive manifest version."""

MODEL_MANIFEST_FILENAME = "manifest.json"
"""Manifest file name stored inside a ``.model`` archive."""


def canonical_model_filename(name: str, step: int) -> str:
    """Return the immutable on-disk filename for a canonical model archive.

    Args:
        name: Canonical model identifier.
        step: Training step associated with the archive.

    Returns:
        File name such as ``policy_step42.model``.
    """
    safe_name = _validate_safe_path_segment(name, field="name")
    return f"{safe_name}_step{step}.model"


def save_model_archive(
    models_dir: str | Path,
    *,
    name: str,
    step: int,
    components: dict[str, Any],
    algorithm_name: str | None = None,
    env_id: str | None = None,
) -> Path:
    """Save named pytree components into a versioned ``.model`` zip archive.

    Each ``(name, step)`` pair maps to a distinct immutable file. Saving a
    later step never removes or overwrites archives from earlier steps.

    Args:
        models_dir: Directory where model archives are stored.
        name: Canonical archive base name without extension.
        step: Training step associated with the archive.
        components: Mapping from component name to pytree payload.
        algorithm_name: Optional algorithm identifier stored in the manifest.
        env_id: Optional environment identifier stored in the manifest.

    Returns:
        Path to the written ``.model`` archive.

    Raises:
        ValueError: If ``name`` or component keys are unsafe path segments.
    """
    safe_name = _validate_safe_path_segment(name, field="name")
    if not components:
        msg = "Model archive requires at least one component."
        raise ValueError(msg)

    models_path = Path(models_dir).resolve()
    models_path.mkdir(parents=True, exist_ok=True)
    archive_filename = canonical_model_filename(safe_name, step)
    archive_path = _resolve_under_directory(models_path, archive_filename)

    with tempfile.TemporaryDirectory(prefix="jarl_model_archive_") as temp_dir:
        workspace = Path(temp_dir)
        staging = workspace / "payload"
        staging.mkdir()
        component_names = sorted(_validate_safe_path_segment(key, field="component") for key in components)
        checkpointer = ocp.PyTreeCheckpointer()
        for component_name in component_names:
            component_dir = staging / component_name
            checkpointer.save(component_dir, components[component_name])
        manifest: dict[str, Any] = {
            "format_version": MODEL_ARCHIVE_FORMAT_VERSION,
            "components": component_names,
            "name": safe_name,
            "step": step,
        }
        if algorithm_name is not None:
            manifest["algorithm_name"] = algorithm_name
        if env_id is not None:
            manifest["env_id"] = env_id
        write_text_atomic(staging / MODEL_MANIFEST_FILENAME, json.dumps(manifest, indent=2))
        temp_archive = models_path / f".{archive_path.name}.tmp"
        _write_zip_from_directory(staging, temp_archive)
        os.replace(temp_archive, archive_path)
    return archive_path


def load_model_archive(archive_path: str | Path) -> dict[str, Any]:
    """Restore all pytree components from a ``.model`` zip archive.

    Args:
        archive_path: Path to a ``.model`` archive under ``models/``.

    Returns:
        Mapping from component name to restored pytree payload.

    Raises:
        ValueError: If the archive manifest is invalid or unsafe to extract.
    """
    path = Path(archive_path)
    with tempfile.TemporaryDirectory(prefix="jarl_model_restore_") as temp_dir:
        staging = Path(temp_dir)
        with zipfile.ZipFile(path, "r") as archive:
            _safe_extract_zip(archive, staging)
        manifest_path = staging / MODEL_MANIFEST_FILENAME
        if not manifest_path.exists():
            msg = f"Model archive missing {MODEL_MANIFEST_FILENAME}: {path}"
            raise ValueError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest, path)
        component_names = manifest["components"]
        checkpointer = ocp.PyTreeCheckpointer()
        return {name: checkpointer.restore(staging / name) for name in component_names}


def build_model_artifact_record(
    *,
    name: str,
    step: int,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactRecord:
    """Build an artifact record for a saved ``.model`` archive.

    Args:
        name: Canonical model identifier used in the registry.
        step: Training step when the archive was written.
        filename: Optional archive file name under ``models/``.
        metadata: Optional extra metadata stored in ``artifacts.json``.

    Returns:
        Model artifact record pointing at the archive path.
    """
    archive_filename = filename or canonical_model_filename(name, step)
    return ArtifactRecord.model(
        name=name,
        step=step,
        filename=archive_filename,
        metadata=dict(metadata or {}),
    )


def make_model_archive_save_callback(
    save_fn: Any,
) -> Any:
    """Build a host callback compatible with ``jax.debug.callback``.

    Bind ``save_fn`` on the host to a workspace method such as
    ``workspace.save_model_archive_from_host``. The callback accepts
    ``(step, name, policy, critic)`` pytrees only and must not capture
    ``NodeWorkspace`` inside compiled code.

    Args:
        save_fn: Host callable invoked with ``(step, name, policy, critic)``.

    Returns:
        Callback suitable for ``jax.debug.callback``.
    """

    def callback(step: Any, name: Any, policy: Any, critic: Any) -> None:
        save_fn(int(step), str(name), policy, critic)

    return callback


def _validate_manifest(manifest: object, archive_path: Path) -> None:
    """Validate a parsed model archive manifest."""
    if not isinstance(manifest, dict):
        msg = f"Model archive manifest must be a JSON object: {archive_path}"
        raise ValueError(msg)
    format_version = manifest.get("format_version")
    if format_version != MODEL_ARCHIVE_FORMAT_VERSION:
        msg = (
            f"Unsupported model archive format_version {format_version!r} in {archive_path}; "
            f"expected {MODEL_ARCHIVE_FORMAT_VERSION}."
        )
        raise ValueError(msg)
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        msg = f"Model archive manifest must list at least one component: {archive_path}"
        raise ValueError(msg)
    if not all(isinstance(name, str) for name in components):
        msg = f"Model archive component names must be strings: {archive_path}"
        raise ValueError(msg)
    for component_name in components:
        _validate_safe_path_segment(component_name, field="component")


def _validate_safe_path_segment(segment: str, *, field: str) -> str:
    """Reject path segments that could escape a single directory level."""
    if not segment or segment in {".", ".."}:
        msg = f"Model archive {field} must be a non-empty safe path segment, got {segment!r}"
        raise ValueError(msg)
    if segment != Path(segment).name:
        msg = f"Model archive {field} must be a single path segment, got {segment!r}"
        raise ValueError(msg)
    if "/" in segment or "\\" in segment:
        msg = f"Model archive {field} must not contain path separators, got {segment!r}"
        raise ValueError(msg)
    return segment


def _resolve_under_directory(root: Path, filename: str) -> Path:
    """Resolve a file name under ``root`` and reject directory escapes."""
    if Path(filename).name != filename:
        msg = f"Model archive filename must be a single path segment, got {filename!r}"
        raise ValueError(msg)
    resolved = (root / filename).resolve()
    if not resolved.is_relative_to(root):
        msg = f"Model archive path escapes destination directory: {filename!r}"
        raise ValueError(msg)
    return resolved


def _write_zip_from_directory(source_dir: Path, destination: Path) -> None:
    """Write a zip archive from ``source_dir`` without nesting the zip in itself."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            member = path.relative_to(source_dir).as_posix()
            archive.write(path, member)


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a zip archive while rejecting path traversal members."""
    dest_root = destination.resolve()
    for member in archive.namelist():
        member_path = dest_root / member
        resolved = member_path.resolve()
        if not resolved.is_relative_to(dest_root):
            msg = f"Unsafe path in model archive member: {member!r}"
            raise ValueError(msg)
        if member.endswith("/"):
            resolved.mkdir(parents=True, exist_ok=True)
            continue
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, resolved.open("wb") as target:
            shutil.copyfileobj(source, target)
