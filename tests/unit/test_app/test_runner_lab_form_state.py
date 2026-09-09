"""Tests for Runner Lab persisted form session state."""

from __future__ import annotations

from typing import Any

from pytest_mock import MockerFixture

from jarl.app.lib.form_state import (
    CONTINUATION_SOURCE_KEY,
    FORM_DEFAULTS_REVISION,
    PERSISTED_FIELDS,
    RESET_REQUEST_KEY,
    REVISION_KEY,
    apply_pending_form_reset,
    ensure_widget_key,
    init_persisted_form,
    lab_key,
    load_run_form_values,
    request_persisted_form_reset,
    reset_persisted_form,
    sync_form_from_run_config_if_needed,
)


class _FakeSessionState(dict):
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def test_init_persisted_form_seeds_missing_keys_only(mocker: MockerFixture) -> None:
    session = _FakeSessionState({lab_key("learning_rate"): 0.42, REVISION_KEY: FORM_DEFAULTS_REVISION})
    mocker.patch("streamlit.session_state", session, create=True)

    init_persisted_form()

    assert session[lab_key("learning_rate")] == 0.42
    assert session[lab_key("algorithm")] == "ppo"
    assert session[lab_key("experiment_name")] == "navix_demo"


def test_init_persisted_form_migrates_stale_revision(mocker: MockerFixture) -> None:
    session = _FakeSessionState({lab_key("total_timesteps"): 512, lab_key("preset"): "fast"})
    mocker.patch("streamlit.session_state", session, create=True)

    init_persisted_form()

    assert session[REVISION_KEY] == FORM_DEFAULTS_REVISION
    assert session[lab_key("total_timesteps")] == 5_000_000
    assert session[lab_key("preset")] == "custom"


def test_request_persisted_form_reset_sets_flag(mocker: MockerFixture) -> None:
    session = _FakeSessionState()
    mocker.patch("streamlit.session_state", session, create=True)

    request_persisted_form_reset()

    assert session[RESET_REQUEST_KEY] is True


def test_apply_pending_form_reset_noop_without_flag(mocker: MockerFixture) -> None:
    session = _FakeSessionState({lab_key("total_timesteps"): 512})
    mocker.patch("streamlit.session_state", session, create=True)
    mocker.patch("streamlit.rerun")

    apply_pending_form_reset()

    assert session[lab_key("total_timesteps")] == 512


def test_reset_persisted_form_overwrites_all_fields(mocker: MockerFixture) -> None:
    session = _FakeSessionState(
        {
            lab_key("total_timesteps"): 512,
            lab_key("preset"): "fast",
            REVISION_KEY: FORM_DEFAULTS_REVISION,
        }
    )
    mocker.patch("streamlit.session_state", session, create=True)

    reset_persisted_form()

    assert session[lab_key("total_timesteps")] == 5_000_000
    assert session[lab_key("preset")] == "custom"
    assert session[lab_key("nr_envs")] == 64


def test_ensure_widget_key_does_not_overwrite_existing(mocker: MockerFixture) -> None:
    session = _FakeSessionState({lab_key("wandb_project"): "my-project"})
    mocker.patch("streamlit.session_state", session, create=True)

    ensure_widget_key("wandb_project", "default-project")

    assert session[lab_key("wandb_project")] == "my-project"


def test_load_run_form_values_reads_persisted_fields(mocker: MockerFixture) -> None:
    session = _FakeSessionState()
    mocker.patch("streamlit.session_state", session, create=True)
    init_persisted_form()
    session[lab_key("learning_rate")] = 0.001
    session[lab_key("track_wandb")] = False
    session[lab_key("video_frequency")] = 64
    session[lab_key("record_final_video")] = False

    values, video_frequency, record_final_video = load_run_form_values()

    assert values.learning_rate == 0.001
    assert values.track_wandb is False
    assert video_frequency == 64
    assert record_final_video is False


def test_persisted_fields_include_experiment_metadata() -> None:
    assert "experiment_name" in PERSISTED_FIELDS
    assert "root_label" in PERSISTED_FIELDS


def test_sync_form_from_run_config_applies_inline(mocker: MockerFixture) -> None:
    session = _FakeSessionState({CONTINUATION_SOURCE_KEY: "node_a"})
    mocker.patch("streamlit.session_state", session, create=True)
    rerun = mocker.patch("streamlit.rerun")
    config = mocker.Mock()
    apply_form = mocker.patch("jarl.app.lib.form_state.apply_form_from_run_config")

    sync_form_from_run_config_if_needed(source_key="node_b", config=config)

    apply_form.assert_called_once_with(config)
    assert session[CONTINUATION_SOURCE_KEY] == "node_b"
    rerun.assert_not_called()


def test_apply_pending_custom_base_does_not_rerun(mocker: MockerFixture) -> None:
    from jarl.app.lib.form_state import CUSTOM_BASE_REQUEST_KEY, apply_pending_custom_base

    session = _FakeSessionState({CUSTOM_BASE_REQUEST_KEY: "simple", lab_key("algorithm"): "ppo"})
    mocker.patch("streamlit.session_state", session, create=True)
    rerun = mocker.patch("streamlit.rerun")

    apply_pending_custom_base()

    assert session[lab_key("preset")] == "custom"
    rerun.assert_not_called()
