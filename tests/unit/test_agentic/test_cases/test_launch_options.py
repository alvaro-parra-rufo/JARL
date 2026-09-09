"""Tests for declarative agentic case launch options."""

from __future__ import annotations

import pytest

from jarl.agentic.cases.launch_options import (
    CaseLaunchOption,
    parse_launch_option_value,
    resolve_tool_overrides,
)


class TestCaseLaunchOptions:
    def test_resolve_tool_overrides_uses_defaults(self) -> None:
        option = CaseLaunchOption(
            key="record_video",
            label="Grabar vídeo",
            tool="graph_checkpoint_rollout",
            field="record_video",
            default=False,
        )

        overrides = resolve_tool_overrides((option,), None)

        assert overrides == {"graph_checkpoint_rollout": {"record_video": False}}

    def test_resolve_tool_overrides_forces_selected_value(self) -> None:
        option = CaseLaunchOption(
            key="record_video",
            label="Grabar vídeo",
            tool="graph_checkpoint_rollout",
            field="record_video",
            default=False,
        )

        overrides = resolve_tool_overrides((option,), {"record_video": True})

        assert overrides == {"graph_checkpoint_rollout": {"record_video": True}}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param(True, True, id="bool"),
            pytest.param("true", True, id="string-true"),
            pytest.param("0", False, id="string-false"),
        ],
    )
    def test_parse_bool_launch_option(self, raw: object, expected: bool) -> None:
        option = CaseLaunchOption(
            key="record_video",
            label="Grabar vídeo",
            tool="graph_checkpoint_rollout",
            field="record_video",
            default=False,
        )

        assert parse_launch_option_value(option, raw) is expected
