"""Tests for identifier helpers."""

from __future__ import annotations

import pytest

from jarl.utils.ids import short_uuid


class TestShortUuid:
    """Tests for short UUID generation."""

    def test_default_length(self) -> None:
        result = short_uuid()

        assert len(result) == 8

    @pytest.mark.parametrize(
        "length",
        [
            pytest.param(4, id="4-chars"),
            pytest.param(12, id="12-chars"),
            pytest.param(32, id="32-chars"),
        ],
    )
    def test_custom_length(self, length: int) -> None:
        result = short_uuid(length)

        assert len(result) == length

    def test_hex_characters_only(self) -> None:
        result = short_uuid(32)

        assert all(c in "0123456789abcdef" for c in result)

    def test_unique_across_calls(self) -> None:
        results = {short_uuid() for _ in range(100)}

        assert len(results) == 100
