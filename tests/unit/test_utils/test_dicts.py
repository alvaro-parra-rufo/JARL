"""Tests for nested dictionary helpers."""

from __future__ import annotations

from jarl.utils.dicts import deep_merge_dicts, flatten_dict, unflatten_dict


class TestFlattenDict:
    """Tests for nested dictionary flattening."""

    def test_flat_dict_unchanged(self) -> None:
        d = {"a": 1, "b": 2}

        result = flatten_dict(d)

        assert result == {"a": 1, "b": 2}

    def test_one_level_nesting(self) -> None:
        d = {"a": {"b": 1, "c": 2}}

        result = flatten_dict(d)

        assert result == {"a.b": 1, "a.c": 2}

    def test_deep_nesting(self) -> None:
        d = {"a": {"b": {"c": {"d": 42}}}}

        result = flatten_dict(d)

        assert result == {"a.b.c.d": 42}

    def test_custom_separator(self) -> None:
        d = {"a": {"b": 898}}

        result = flatten_dict(d, sep="/")

        assert result == {"a/b": 898}

    def test_mixed_nesting(self) -> None:
        d = {"flat": 1, "nested": {"deep": 2}, "also_flat": 3}

        result = flatten_dict(d)

        assert result == {"flat": 1, "nested.deep": 2, "also_flat": 3}

    def test_empty_dict(self) -> None:
        result = flatten_dict({})

        assert result == {}

    def test_round_trip_with_unflatten(self) -> None:
        nested = {"flat": 1, "nested": {"deep": 2}, "also_flat": 3}

        result = unflatten_dict(flatten_dict(nested))

        assert result == nested


class TestUnflattenDict:
    """Tests for nested dictionary expansion."""

    def test_expands_dotted_keys(self) -> None:
        flat = {"a.b": 1, "a.c": 2, "flat": 3}

        result = unflatten_dict(flat)

        assert result == {"a": {"b": 1, "c": 2}, "flat": 3}

    def test_round_trip_with_flatten(self) -> None:
        nested = {"flat": 1, "nested": {"deep": 2}, "also_flat": 3}

        result = unflatten_dict(flatten_dict(nested))

        assert result == nested

    def test_round_trip_with_dotted_keys(self) -> None:
        flat = {"a.b.c": 1, "x": 2}

        result = unflatten_dict(flat)

        assert flatten_dict(result) == flat

    def test_preserves_nested_dict_values(self) -> None:
        flat = {"jax": {"precision": "float32"}}

        result = unflatten_dict(flat)

        assert result == {"jax": {"precision": "float32"}}


class TestDeepMergeDicts:
    """Tests for deep dictionary merging."""

    def test_merges_nested_keys(self) -> None:
        base = {"jax": {"precision": "bfloat16", "cache_dir": "var/cache"}}
        updates = {"jax": {"precision": "float32"}}

        result = deep_merge_dicts(base, updates)

        assert result == {"jax": {"precision": "float32", "cache_dir": "var/cache"}}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        updates = {"a": {"c": 2}}

        _ = deep_merge_dicts(base, updates)

        assert base == {"a": {"b": 1}}
