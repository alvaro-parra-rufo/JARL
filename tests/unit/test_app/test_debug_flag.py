"""Tests for the model-debug sidebar flag."""

from __future__ import annotations

import inspect

from pytest_mock import MockerFixture

from jarl.app.lib import session as session_mod
from jarl.app.lib.debug.flag import MODEL_DEBUG_FLAG_KEY, is_debug_enabled


class TestIsDebugEnabled:
    def test_missing_key_is_off(self, mocker: MockerFixture) -> None:
        mocker.patch("streamlit.session_state", {}, create=True)

        assert is_debug_enabled() is False

    def test_reads_session_key(self, mocker: MockerFixture) -> None:
        mocker.patch("streamlit.session_state", {MODEL_DEBUG_FLAG_KEY: True}, create=True)

        assert is_debug_enabled() is True
        assert MODEL_DEBUG_FLAG_KEY not in inspect.getsource(session_mod)
