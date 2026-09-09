"""Tests for flexible list Pydantic annotations."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from jarl.utils.pydantic import FlexibleList, parse_flexible_list


class _MetricKeysModel(BaseModel):
    metric_keys: FlexibleList[str] | None = None


class TestParseFlexibleList:
    def test_none_passthrough(self) -> None:
        assert parse_flexible_list(None) is None

    def test_list_and_tuple(self) -> None:
        assert parse_flexible_list(["a", "b"]) == ["a", "b"]
        assert parse_flexible_list(("a", "b")) == ["a", "b"]

    def test_comma_separated_string(self) -> None:
        assert parse_flexible_list("eval/episode_return,eval/episode_length") == [
            "eval/episode_return",
            "eval/episode_length",
        ]

    def test_preserves_empty_segments(self) -> None:
        assert parse_flexible_list(" a , , b ") == ["a", "", "b"]

    def test_blank_string_becomes_empty_list(self) -> None:
        assert parse_flexible_list("") == []
        assert parse_flexible_list("   ") == []

    def test_rejects_unsupported_shapes(self) -> None:
        with pytest.raises(ValueError, match="comma-separated"):
            parse_flexible_list(3)


class TestFlexibleListModel:
    def test_accepts_list(self) -> None:
        model = _MetricKeysModel.model_validate({"metric_keys": ["a", "b"]})

        assert model.metric_keys == ["a", "b"]

    def test_accepts_comma_separated_string(self) -> None:
        model = _MetricKeysModel.model_validate({"metric_keys": "eval/episode_return, eval/episode_length"})

        assert model.metric_keys == ["eval/episode_return", "eval/episode_length"]

    def test_optional_none(self) -> None:
        model = _MetricKeysModel.model_validate({})

        assert model.metric_keys is None

    def test_rejects_mapping_items(self) -> None:
        with pytest.raises(ValidationError):
            _MetricKeysModel.model_validate({"metric_keys": [{"a": 1}]})
