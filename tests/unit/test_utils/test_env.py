"""Tests for project environment loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarl.utils.env import find_project_root, load_project_env


def test_find_project_root_walks_parent_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "package"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    root = find_project_root(nested)

    assert root == project


def test_load_project_env_preserves_existing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / ".env").write_text("JARL_ENV_TEST=file\n", encoding="utf-8")
    monkeypatch.setenv("JARL_ENV_TEST", "process")

    loaded = load_project_env(start=tmp_path)

    assert loaded == tmp_path / ".env"
    assert os.environ["JARL_ENV_TEST"] == "process"
