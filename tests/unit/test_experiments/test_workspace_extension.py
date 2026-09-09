"""Tests for workspace subclass reconstruction helpers."""

from __future__ import annotations

import pytest

from jarl.experiments.node import NodeWorkspace
from jarl.experiments.workspace_extension import resolve_workspace_class


class TestResolveWorkspaceClass:
    """Tests for strict workspace class resolution."""

    def test_empty_import_path_returns_base(self) -> None:
        assert resolve_workspace_class("") is NodeWorkspace

    def test_invalid_import_path_raises(self) -> None:
        with pytest.raises(ImportError, match="No module named"):
            resolve_workspace_class("definitely.missing.module.Workspace")

    def test_allow_fallback_returns_base(self) -> None:
        resolved = resolve_workspace_class(
            "definitely.missing.module.Workspace",
            allow_fallback=True,
        )

        assert resolved is NodeWorkspace
