"""Resolve experiment parent directories from CLI override or environment."""

from __future__ import annotations

import os
from pathlib import Path

JARL_EXPERIMENTS_ROOT_ENV = "JARL_EXPERIMENTS_ROOT"
"""Environment variable for the parent directory of experiment trees."""

JARL_CASES_ROOT_ENV = "JARL_CASES_ROOT"
"""Environment variable for the parent directory of materialized case runs."""

DEFAULT_EXPERIMENTS_ROOT = Path("results") / "dags"
"""Default experiment parent directory relative to the repository root."""

DEFAULT_CASES_ROOT = Path("results") / "cases"
"""Default parent directory for isolated case materializations."""

__all__ = [
    "DEFAULT_CASES_ROOT",
    "DEFAULT_EXPERIMENTS_ROOT",
    "JARL_CASES_ROOT_ENV",
    "JARL_EXPERIMENTS_ROOT_ENV",
    "resolve_cases_root",
    "resolve_workspace_root",
]


def resolve_workspace_root(*, override: str | Path | None = None) -> Path:
    """Return the directory that contains experiment folders.

    Precedence: explicit ``override`` → ``JARL_EXPERIMENTS_ROOT`` → ``<repo>/results/dags``.

    Relative values are anchored to the repository root (``pyproject.toml``), not ``cwd``.
    """
    if override is not None:
        return _anchor_to_repo(Path(override))
    raw = os.environ.get(JARL_EXPERIMENTS_ROOT_ENV, str(DEFAULT_EXPERIMENTS_ROOT))
    return _anchor_to_repo(Path(raw))


def resolve_cases_root(*, override: str | Path | None = None) -> Path:
    """Return the directory that contains isolated case experiment folders.

    Precedence: explicit ``override`` → ``JARL_CASES_ROOT`` → ``<repo>/results/cases``.
    """
    if override is not None:
        return _anchor_to_repo(Path(override))
    raw = os.environ.get(JARL_CASES_ROOT_ENV, str(DEFAULT_CASES_ROOT))
    return _anchor_to_repo(Path(raw))


def _anchor_to_repo(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file():
            return (candidate / expanded).resolve()
    return expanded.resolve()
