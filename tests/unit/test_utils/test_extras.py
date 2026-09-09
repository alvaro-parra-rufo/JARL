"""Tests for optional dependency availability checks."""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from jarl.utils.extras import is_extra_available


@pytest.mark.parametrize(
    ("missing_module", "expected"),
    [
        pytest.param(None, True, id="all-present"),
        pytest.param("langgraph", False, id="missing-langgraph"),
        pytest.param("langchain_core", False, id="missing-langchain-core"),
    ],
)
def test_agentic_extra_checks_all_import_markers(
    missing_module: str | None,
    expected: bool,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "jarl.utils.extras.find_spec",
        side_effect=lambda module: None if module == missing_module else object(),
    )

    available = is_extra_available("agentic")

    assert available is expected
