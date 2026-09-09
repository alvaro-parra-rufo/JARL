"""App-specific preset integration tests (form state persistence)."""

from __future__ import annotations


def test_default_form_payload_save_model_is_true() -> None:
    from jarl.app.lib.form_state import default_form_payload

    payload = default_form_payload()
    assert payload["save_model"] is True
