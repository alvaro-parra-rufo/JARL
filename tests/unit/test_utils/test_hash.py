"""Tests for dictionary hashing."""

from __future__ import annotations

from jarl.utils.hash import dict_hash


class TestDictHash:
    """Tests for deterministic dictionary hashing."""

    def test_same_dict_same_hash(self) -> None:
        d = {"a": 1, "b": 2}

        assert dict_hash(d) == dict_hash(d)

    def test_order_independent(self) -> None:
        a = {"x": 1, "y": 2, "z": 3}
        b = {"z": 3, "x": 1, "y": 2}

        assert dict_hash(a) == dict_hash(b)

    def test_different_values_different_hash(self) -> None:
        a = {"key": "value_a"}
        b = {"key": "value_b"}

        assert dict_hash(a) != dict_hash(b)

    def test_nested_dicts_handled(self) -> None:
        d = {"outer": {"inner": 42}}

        result = dict_hash(d)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_returns_hex_string(self) -> None:
        result = dict_hash({"a": 1})

        assert all(c in "0123456789abcdef" for c in result)
