"""Optional functional tests that require a real LLM provider (``@pytest.mark.llm``)."""

from __future__ import annotations

import os

import pytest

from jarl.agentic.llm import (
    MAIN_COMPONENT_ID,
    create_chat_model,
    load_llm_catalog,
    load_llm_settings_from_env,
    resolve_llm_yaml_path,
)
from jarl.utils.env import load_project_env


def _require_configured_llm() -> None:
    load_project_env()
    if resolve_llm_yaml_path() is None and os.environ.get("JARL_LLM_PROVIDER") is None:
        pytest.skip(
            "Configure src/jarl/agentic/llm/profiles/custom.yaml (or default.yaml) "
            "or set JARL_LLM_PROVIDER (and provider-specific vars) to run LLM tests."
        )


@pytest.mark.llm
def test_load_llm_settings_from_configured_env() -> None:
    """Smoke test: catalog or env-backed settings resolve when configured."""
    _require_configured_llm()

    catalog_path = resolve_llm_yaml_path()
    if catalog_path is not None:
        settings = load_llm_catalog().resolve(MAIN_COMPONENT_ID).settings
    else:
        settings = load_llm_settings_from_env()
    assert settings.model
    if settings.provider == "openai_compatible":
        assert settings.base_url
        assert settings.api_key


@pytest.mark.llm
def test_create_chat_model_from_configured_env() -> None:
    """Smoke test: factory returns a chat model without invoking the agent graph."""
    _require_configured_llm()

    catalog_path = resolve_llm_yaml_path()
    if catalog_path is not None:
        llm = create_chat_model(load_llm_catalog().resolve(MAIN_COMPONENT_ID).settings)
    else:
        llm = create_chat_model()
    assert llm is not None
