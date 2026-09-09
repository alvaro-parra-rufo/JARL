"""Tests for Runner Lab live UI helpers."""

from __future__ import annotations

from pytest_mock import MockerFixture

from jarl.app.lib.live_ui import (
    LAST_RUN_EXIT_CODE_KEY,
    RUN_FINISH_NOTIFIED_KEY,
    reset_run_finish_state,
)


class TestResetRunFinishState:
    """Tests for run-finish notification reset."""

    def test_reset_run_finish_state_clears_flags(self, mocker: MockerFixture) -> None:
        session: dict[str, object] = {
            RUN_FINISH_NOTIFIED_KEY: True,
            LAST_RUN_EXIT_CODE_KEY: 0,
        }
        mocker.patch("streamlit.session_state", session, create=True)

        reset_run_finish_state()

        assert session[RUN_FINISH_NOTIFIED_KEY] is False
        assert LAST_RUN_EXIT_CODE_KEY not in session
