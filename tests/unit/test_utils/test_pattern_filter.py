"""Tests for pattern matching helpers."""

from __future__ import annotations

import pytest

from jarl.utils.pattern_filter import jarl_pattern_match, match_patterns


class TestJarlPatternMatch:
    """Tests for the default pattern matcher."""

    @pytest.mark.parametrize(
        ("value", "pattern", "expected"),
        [
            pytest.param("graph", "graph", True, id="exact"),
            pytest.param("graph_fork", "graph_*", True, id="glob"),
            pytest.param("train", "graph", False, id="no-match"),
        ],
    )
    def test_match(self, value: str, pattern: str, expected: bool) -> None:
        assert jarl_pattern_match(value, pattern) is expected


class TestMatchPatterns:
    """Tests for include/exclude pattern filtering."""

    def test_include_none_passes_after_exclude(self) -> None:
        assert match_patterns({"graph", "read"}, include=None, exclude=None) is True

    def test_empty_include_never_matches(self) -> None:
        assert match_patterns({"graph"}, include=set()) is False

    def test_empty_exclude_excludes_nothing(self) -> None:
        assert match_patterns({"graph"}, include={"graph"}, exclude=set()) is True

    def test_exclude_disqualifies(self) -> None:
        assert (
            match_patterns(
                {"graph", "subagent"},
                include={"graph", "read"},
                exclude={"subagent"},
            )
            is False
        )

    def test_include_requires_any_label_match(self) -> None:
        assert match_patterns({"graph", "mutation"}, include={"read", "graph"}) is True

    def test_include_no_match_fails(self) -> None:
        assert match_patterns({"mutation"}, include={"read", "graph"}) is False

    def test_glob_include_matches_one_of_many_labels(self) -> None:
        assert match_patterns({"graph_fork"}, include={"graph_*"}) is True

    def test_custom_matcher(self) -> None:
        assert (
            match_patterns(
                {"abc"},
                include={"a"},
                matcher=lambda value, pattern: value.startswith(pattern),
            )
            is True
        )
