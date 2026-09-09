"""Tests for the session finish operation."""

from __future__ import annotations

from jarl.operations.session.finish import FinishRequest, finish


class TestFinish:
    def test_finish_declares_the_thread_complete(self) -> None:
        response = finish(FinishRequest(), thread_id="thread-abc")

        compact = response.to_compact_dict()

        assert response.finished is True
        assert response.thread_id == "thread-abc"
        assert compact == {"finished": True, "thread_id": "thread-abc"}
