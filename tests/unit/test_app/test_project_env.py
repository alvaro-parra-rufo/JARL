"""Tests for repository ``.env`` loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_load_project_env_reads_export_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='jarl'\n", encoding="utf-8")
    (tmp_path / ".env").write_text('export WANDB_API_KEY="test-key-from-env"\n', encoding="utf-8")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    from jarl.app.lib.project_env import load_project_env

    loaded = load_project_env(start=tmp_path)
    assert loaded == tmp_path / ".env"
    assert os.environ.get("WANDB_API_KEY") == "test-key-from-env"
