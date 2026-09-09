"""Project environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["find_project_root", "load_project_env"]


def find_project_root(start: str | Path | None = None) -> Path | None:
    """Find the nearest parent directory containing `pyproject.toml`."""
    origin = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for path in (origin, *origin.parents):
        if (path / "pyproject.toml").is_file():
            return path

    package_path = Path(__file__).resolve()
    for path in package_path.parents:
        if (path / "pyproject.toml").is_file():
            return path
    return None


def load_project_env(
    *,
    start: str | Path | None = None,
    override: bool = False,
) -> Path | None:
    """Load the project `.env` file when present.

    Args:
        start: Optional path used to begin project-root discovery.
        override: Whether loaded values replace existing environment variables.

    Returns:
        Loaded `.env` path, or `None` when no project file exists.
    """
    root = find_project_root(start)
    if root is None:
        return None
    env_path = root / ".env"
    if not env_path.is_file():
        return None

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=override)
    except ImportError:
        pass

    _load_simple_env_lines(env_path, override=override)
    return env_path


def _load_simple_env_lines(env_path: Path, *, override: bool) -> None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"').strip("'")
        if normalized_key and (override or normalized_key not in os.environ):
            os.environ[normalized_key] = normalized_value
