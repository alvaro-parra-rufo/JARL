"""Tests for jarl.utils package re-exports."""

from __future__ import annotations

import jarl.utils as utils_module


def test_public_exports_available() -> None:
    expected = {
        "deep_merge_dicts",
        "dict_hash",
        "find_project_root",
        "flatten_dict",
        "is_extra_available",
        "jarl_pattern_match",
        "load_project_env",
        "match_patterns",
        "OptionalExtra",
        "short_uuid",
        "unflatten_dict",
        "write_text_atomic",
    }

    assert set(utils_module.__all__) == expected
