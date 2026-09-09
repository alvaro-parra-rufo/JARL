"""Workspace preset helpers for the Runner Lab sidebar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jarl.experiments.paths import resolve_cases_root, resolve_workspace_root

__all__ = [
    "WORKSPACE_PRESET_CASES",
    "WORKSPACE_PRESET_CUSTOM",
    "WORKSPACE_PRESET_DAGS",
    "CustomWorkspaceTarget",
    "infer_workspace_preset",
    "resolve_custom_workspace_target",
    "resolve_workspace_preset_root",
    "sticky_workspace_preset",
    "workspace_preset_labels",
]

WORKSPACE_PRESET_DAGS = "dags"
WORKSPACE_PRESET_CASES = "cases"
WORKSPACE_PRESET_CUSTOM = "custom"

_WORKSPACE_PRESETS: tuple[str, ...] = (
    WORKSPACE_PRESET_DAGS,
    WORKSPACE_PRESET_CASES,
    WORKSPACE_PRESET_CUSTOM,
)


@dataclass(frozen=True, slots=True)
class CustomWorkspaceTarget:
    """Resolved custom workspace: experiments parent, plus an experiment if the path was one."""

    root: Path
    experiment_dir: Path | None = None


def workspace_preset_labels() -> dict[str, str]:
    """Return selectbox labels for each workspace preset."""
    dags_root = resolve_workspace_root()
    cases_root = resolve_cases_root()
    return {
        WORKSPACE_PRESET_DAGS: f"Dags · {dags_root}",
        WORKSPACE_PRESET_CASES: f"Cases · {cases_root}",
        WORKSPACE_PRESET_CUSTOM: "Personalizado…",
    }


def infer_workspace_preset(workspace: Path) -> str:
    """Map the active workspace path to a preset id."""
    resolved = workspace.expanduser().resolve()
    if resolved == resolve_cases_root().resolve():
        return WORKSPACE_PRESET_CASES
    if resolved == resolve_workspace_root().resolve():
        return WORKSPACE_PRESET_DAGS
    return WORKSPACE_PRESET_CUSTOM


def sticky_workspace_preset(*, inferred: str, selected: str | None) -> str:
    """Return the sidebar preset without snapping Personalizado back to Dags/Cases.

    The applied workspace only seeds the widget on first load. After that the
    selectbox value wins so the user can type a custom path and apply it.
    """
    if selected is None:
        return inferred
    return selected


def resolve_workspace_preset_root(preset: str, *, custom_path: str | None = None) -> Path:
    """Return the workspace directory for one preset selection."""
    if preset == WORKSPACE_PRESET_CASES:
        return resolve_cases_root()
    if preset == WORKSPACE_PRESET_DAGS:
        return resolve_workspace_root()
    if preset != WORKSPACE_PRESET_CUSTOM:
        msg = f"Unknown workspace preset: {preset!r}"
        raise ValueError(msg)
    if custom_path is None or not custom_path.strip():
        msg = "Custom workspace preset requires a path."
        raise ValueError(msg)
    return resolve_custom_workspace_target(custom_path).root


def resolve_custom_workspace_target(custom_path: str) -> CustomWorkspaceTarget:
    """Resolve a custom path that may be an experiments parent or an experiment directory.

    The sidebar workspace is the parent of experiment folders. Pasting a folder that
    already contains `experiment.json` selects that experiment and uses its parent.
    """
    if not custom_path.strip():
        msg = "Custom workspace preset requires a path."
        raise ValueError(msg)
    resolved = resolve_workspace_root(override=custom_path.strip())
    if (resolved / "experiment.json").is_file():
        return CustomWorkspaceTarget(root=resolved.parent, experiment_dir=resolved)
    return CustomWorkspaceTarget(root=resolved)


def workspace_preset_options() -> tuple[str, ...]:
    """Return stable preset ids for sidebar selectboxes."""
    return _WORKSPACE_PRESETS
