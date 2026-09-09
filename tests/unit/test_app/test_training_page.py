"""Tests for Runner Lab training page helpers."""

from __future__ import annotations

from jarl.app.lib.training_page import training_launch_allowed


class TestTrainingLaunchAllowed:
    """Launch gating logic."""

    def test_blocks_on_validation_messages(self) -> None:
        assert (
            training_launch_allowed(
                validation_messages=["minibatch inválido"],
                config_ready=True,
            )
            is False
        )

    def test_allows_when_config_ready(self) -> None:
        assert training_launch_allowed(validation_messages=[], config_ready=True) is True

    def test_blocks_on_extra_blockers(self) -> None:
        assert (
            training_launch_allowed(
                validation_messages=[],
                config_ready=True,
                extra_blockers=["Nodo completado"],
            )
            is False
        )
