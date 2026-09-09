"""Runner Lab adapters for project environment loading."""

from __future__ import annotations

from pathlib import Path

from jarl.utils.env import find_project_root
from jarl.utils.env import load_project_env as load_core_project_env

__all__ = ["find_repo_root", "load_project_env"]


def find_repo_root(start: Path | None = None) -> Path | None:
    """Return the repository root (directory containing ``pyproject.toml``)."""
    return find_project_root(start)


def load_project_env(*, start: Path | None = None) -> Path | None:
    """Load the project `.env` without overriding existing values."""
    return load_core_project_env(start=start)
