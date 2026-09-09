"""Tests for experiment case metadata."""

from __future__ import annotations

import pytest

from jarl.experiments.cases import CaseSpec


class TestCaseSpec:
    @pytest.mark.parametrize(
        "case_id",
        [
            pytest.param("", id="empty"),
            pytest.param("UpperCase", id="uppercase"),
            pytest.param("contains space", id="space"),
            pytest.param("-leading", id="leading-separator"),
        ],
    )
    def test_rejects_invalid_case_ids(self, case_id: str) -> None:
        with pytest.raises(ValueError, match="Invalid case id"):
            CaseSpec(id=case_id, title="Invalid")

    def test_rejects_empty_title(self) -> None:
        with pytest.raises(ValueError, match="title"):
            CaseSpec(id="valid", title=" ")

    @pytest.mark.parametrize(
        ("field_name", "field_value"),
        [
            pytest.param("tags", frozenset({""}), id="tags"),
            pytest.param("requirements", frozenset({" "}), id="requirements"),
        ],
    )
    def test_rejects_empty_filter_metadata(
        self,
        field_name: str,
        field_value: frozenset[str],
    ) -> None:
        kwargs = {field_name: field_value}

        with pytest.raises(ValueError, match=field_name):
            CaseSpec(id="valid", title="Valid", **kwargs)

    def test_resolved_objective_returns_explicit_objective(self) -> None:
        spec = CaseSpec(
            id="valid",
            title="Valid",
            description="Catalog blurb for testers.",
            objective="Summarize the experiment. Do not mutate or train.",
        )

        assert spec.resolved_objective == "Summarize the experiment. Do not mutate or train."

    def test_resolved_objective_ignores_description_when_empty(self) -> None:
        spec = CaseSpec(id="valid", title="Valid", description="Catalog blurb.")

        assert spec.resolved_objective is None
