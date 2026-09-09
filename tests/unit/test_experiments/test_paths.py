"""Tests for experiment parent directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.experiments.paths import (
    DEFAULT_CASES_ROOT,
    DEFAULT_EXPERIMENTS_ROOT,
    JARL_CASES_ROOT_ENV,
    JARL_EXPERIMENTS_ROOT_ENV,
    resolve_cases_root,
    resolve_workspace_root,
)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "jarl"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='jarl'\n", encoding="utf-8")
    return repo


def test_resolve_workspace_root_default_under_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    app_dir = repo / "src" / "jarl" / "app"
    app_dir.mkdir(parents=True)
    monkeypatch.chdir(app_dir)
    monkeypatch.delenv(JARL_EXPERIMENTS_ROOT_ENV, raising=False)

    resolved = resolve_workspace_root()

    assert resolved == (repo / DEFAULT_EXPERIMENTS_ROOT).resolve()


def test_resolve_workspace_root_env_relative_anchored_to_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    notebooks = repo / "notebooks"
    notebooks.mkdir()
    monkeypatch.chdir(notebooks)
    monkeypatch.setenv(JARL_EXPERIMENTS_ROOT_ENV, "results/dags")

    resolved = resolve_workspace_root()

    assert resolved == (repo / "results" / "dags").resolve()


def test_resolve_workspace_root_env_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_repo(tmp_path)
    custom = tmp_path / "custom" / "runs"
    monkeypatch.setenv(JARL_EXPERIMENTS_ROOT_ENV, str(custom))

    resolved = resolve_workspace_root()

    assert resolved == custom.resolve()


def test_resolve_workspace_root_cli_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    override = repo / "from_cli"
    monkeypatch.setenv(JARL_EXPERIMENTS_ROOT_ENV, "ignored/relative")

    resolved = resolve_workspace_root(override=override)

    assert resolved == override.resolve()


def test_resolve_cases_root_default_under_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    app_dir = repo / "src" / "jarl" / "app"
    app_dir.mkdir(parents=True)
    monkeypatch.chdir(app_dir)
    monkeypatch.delenv(JARL_CASES_ROOT_ENV, raising=False)

    resolved = resolve_cases_root()

    assert resolved == (repo / DEFAULT_CASES_ROOT).resolve()


def test_resolve_cases_root_env_relative_anchored_to_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    notebooks = repo / "notebooks"
    notebooks.mkdir()
    monkeypatch.chdir(notebooks)
    monkeypatch.setenv(JARL_CASES_ROOT_ENV, "results/cases")

    resolved = resolve_cases_root()

    assert resolved == (repo / "results" / "cases").resolve()
