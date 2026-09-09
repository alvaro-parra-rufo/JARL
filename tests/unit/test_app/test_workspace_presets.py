"""Tests for Runner Lab workspace presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.app.lib.session import list_experiments
from jarl.app.lib.workspace_presets import (
    WORKSPACE_PRESET_CASES,
    WORKSPACE_PRESET_CUSTOM,
    WORKSPACE_PRESET_DAGS,
    infer_workspace_preset,
    resolve_custom_workspace_target,
    resolve_workspace_preset_root,
    sticky_workspace_preset,
)


def test_infer_workspace_preset_recognizes_dags_and_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'jarl'\n", encoding="utf-8")
    dags = repo / "results" / "dags"
    cases = repo / "results" / "cases"
    dags.mkdir(parents=True)
    cases.mkdir(parents=True)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("JARL_EXPERIMENTS_ROOT", raising=False)
    monkeypatch.delenv("JARL_CASES_ROOT", raising=False)

    assert infer_workspace_preset(dags) == WORKSPACE_PRESET_DAGS
    assert infer_workspace_preset(cases) == WORKSPACE_PRESET_CASES
    assert infer_workspace_preset(repo / "other") == WORKSPACE_PRESET_CUSTOM


def test_sticky_workspace_preset_keeps_custom_while_workspace_is_still_dags() -> None:
    assert (
        sticky_workspace_preset(inferred=WORKSPACE_PRESET_DAGS, selected=WORKSPACE_PRESET_CUSTOM)
        == WORKSPACE_PRESET_CUSTOM
    )


def test_sticky_workspace_preset_seeds_from_workspace_on_first_load() -> None:
    assert sticky_workspace_preset(inferred=WORKSPACE_PRESET_CASES, selected=None) == WORKSPACE_PRESET_CASES


def test_sticky_workspace_preset_does_not_override_dags_selection() -> None:
    assert (
        sticky_workspace_preset(inferred=WORKSPACE_PRESET_CUSTOM, selected=WORKSPACE_PRESET_DAGS)
        == WORKSPACE_PRESET_DAGS
    )


def test_resolve_workspace_preset_root_custom_anchors_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'jarl'\n", encoding="utf-8")
    custom = repo / "my" / "experiments"
    monkeypatch.chdir(repo)
    monkeypatch.delenv("JARL_EXPERIMENTS_ROOT", raising=False)

    resolved = resolve_workspace_preset_root(WORKSPACE_PRESET_CUSTOM, custom_path="my/experiments")

    assert resolved == custom.resolve()


def test_resolve_custom_workspace_target_parent_keeps_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'jarl'\n", encoding="utf-8")
    parent = repo / "results" / "cmp"
    parent.mkdir(parents=True)
    monkeypatch.chdir(repo)

    target = resolve_custom_workspace_target(str(parent))

    assert target.root == parent.resolve()
    assert target.experiment_dir is None


def test_resolve_custom_workspace_target_experiment_dir_uses_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'jarl'\n", encoding="utf-8")
    experiment = repo / "results" / "cmp" / "jarl"
    experiment.mkdir(parents=True)
    (experiment / "experiment.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(repo)

    target = resolve_custom_workspace_target(str(experiment))

    assert target.root == experiment.parent.resolve()
    assert target.experiment_dir == experiment.resolve()
    assert (
        resolve_workspace_preset_root(WORKSPACE_PRESET_CUSTOM, custom_path=str(experiment))
        == experiment.parent.resolve()
    )


def test_list_experiments_accepts_experiment_directory_as_root(tmp_path: Path) -> None:
    experiment = tmp_path / "jarl"
    experiment.mkdir()
    (experiment / "experiment.json").write_text("{}", encoding="utf-8")

    assert list_experiments(experiment) == [experiment]
