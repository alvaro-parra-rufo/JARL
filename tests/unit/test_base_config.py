"""Tests for BaseConfig immutable path helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest

from jarl.config import BaseConfig, collect_diff_paths


class InnerConfig(BaseConfig):
    """Nested config block for immutable path tests."""

    IMMUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset({"locked"})

    locked: int = 1
    mutable: int = 2


class SampleConfig(BaseConfig):
    """Top-level config with nested immutable declarations."""

    IMMUTABLE_PATHS: ClassVar[frozenset[str]] = frozenset({"root_locked"})

    root_locked: str = "fixed"
    root_mutable: str = "open"
    inner: InnerConfig = InnerConfig()


class TestImmutablePaths:
    """Tests for immutable path collection and diff validation."""

    def test_immutable_paths_composes_nested_fields(self) -> None:
        paths = SampleConfig.immutable_paths()

        assert "root_locked" in paths
        assert "inner.locked" in paths
        assert "root_mutable" not in paths
        assert "inner.mutable" not in paths

    @pytest.mark.parametrize(
        ("overrides", "should_fail"),
        [
            pytest.param({"root_mutable": "ok"}, False, id="mutable_root"),
            pytest.param({"inner.mutable": 9}, False, id="mutable_nested"),
            pytest.param({"root_locked": "nope"}, True, id="immutable_root"),
            pytest.param({"inner.locked": 9}, True, id="immutable_nested"),
        ],
    )
    def test_validate_diff_allowed(
        self,
        overrides: dict[str, object],
        should_fail: bool,
    ) -> None:
        base = SampleConfig()
        target = base.apply_overrides(overrides, nested=True)
        diff = base.diff(target, flatten=True)

        if should_fail:
            with pytest.raises(ValueError, match="immutable paths"):
                base.validate_diff_allowed(diff)
        else:
            base.validate_diff_allowed(diff)

    def test_collect_diff_paths_flattens_nested_dict_changes(self) -> None:
        base = SampleConfig()
        target = base.apply_overrides({"inner.mutable": 5}, nested=True)
        diff = base.diff(target)

        assert collect_diff_paths(diff) == frozenset({"inner.mutable"})
