"""Unit tests for jarl.agentic.llm factory and settings."""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from jarl.agentic.errors import LLMConfigurationError
from jarl.agentic.llm.config import (
    DEFAULT_OLLAMA_BASE_URL,
    LLMSettings,
    load_llm_settings_from_env,
)
from jarl.agentic.llm.factory import create_chat_model

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "api_key"),
    [
        ("ollama", "qwen2.5:7b", None, None),
        ("openai_compatible", "gpt-4o-mini", "https://api.example.com/v1", "test-key"),
    ],
)
def test_llm_settings_provider_defaults(
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> None:
    settings = LLMSettings(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    if provider == "ollama":
        assert settings.model == "qwen2.5:7b"
        assert settings.base_url == DEFAULT_OLLAMA_BASE_URL
    else:
        assert settings.model == "gpt-4o-mini"
        assert settings.base_url == "https://api.example.com/v1"


def test_load_llm_settings_from_env_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JARL_LLM_MODEL", "mistral:latest")
    monkeypatch.delenv("JARL_LLM_BASE_URL", raising=False)

    settings = load_llm_settings_from_env()
    assert settings.provider == "ollama"
    assert settings.model == "mistral:latest"
    assert settings.base_url == DEFAULT_OLLAMA_BASE_URL


def test_load_llm_settings_from_env_openai_compatible_deepseek_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("JARL_LLM_BASE_URL", DEEPSEEK_BASE_URL)
    monkeypatch.setenv("JARL_LLM_MODEL", DEEPSEEK_MODEL)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("JARL_LLM_API_KEY", raising=False)

    settings = load_llm_settings_from_env()
    assert settings.provider == "openai_compatible"
    assert settings.model == DEEPSEEK_MODEL
    assert settings.base_url == DEEPSEEK_BASE_URL
    assert settings.api_key == "secret"


def test_load_llm_settings_from_env_rejects_legacy_deepseek_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "deepseek")

    with pytest.raises(ValueError, match="openai_compatible"):
        load_llm_settings_from_env()


def test_load_llm_settings_from_env_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported JARL_LLM_PROVIDER"):
        load_llm_settings_from_env()


def test_create_chat_model_ollama(mocker: MockerFixture) -> None:
    chat_ollama = mocker.patch("langchain_ollama.ChatOllama")
    settings = LLMSettings(provider="ollama", model="llama3.1:latest")

    create_chat_model(settings, timeout=30)

    chat_ollama.assert_called_once_with(
        model="llama3.1:latest",
        base_url=DEFAULT_OLLAMA_BASE_URL,
        timeout=30,
    )


def test_create_chat_model_openai_compatible(mocker: MockerFixture) -> None:
    chat_openai = mocker.patch("langchain_openai.ChatOpenAI")
    settings = LLMSettings(
        provider="openai_compatible",
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key="secret",
        temperature=0.2,
    )

    create_chat_model(settings)

    chat_openai.assert_called_once_with(
        model=DEEPSEEK_MODEL,
        api_key="secret",
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2,
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_create_chat_model_openai_compatible_non_deepseek_skips_thinking(
    mocker: MockerFixture,
) -> None:
    """Non-DeepSeek OpenAI-compatible hosts should not get a thinking extra_body."""
    chat_openai = mocker.patch("langchain_openai.ChatOpenAI")
    settings = LLMSettings(
        provider="openai_compatible",
        model="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        api_key="secret",
    )

    create_chat_model(settings)

    chat_openai.assert_called_once_with(
        model="gpt-4o-mini",
        api_key="secret",
        base_url="https://api.example.com/v1",
    )


def test_create_chat_model_deepseek_keeps_explicit_thinking(mocker: MockerFixture) -> None:
    """An explicit thinking extra_body should not be overwritten for DeepSeek."""
    chat_openai = mocker.patch("langchain_openai.ChatOpenAI")
    settings = LLMSettings(
        provider="openai_compatible",
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key="secret",
    )

    create_chat_model(settings, extra_body={"thinking": {"type": "enabled"}})

    chat_openai.assert_called_once_with(
        model=DEEPSEEK_MODEL,
        api_key="secret",
        base_url=DEEPSEEK_BASE_URL,
        extra_body={"thinking": {"type": "enabled"}},
    )


def test_create_chat_model_wraps_settings_errors_missing_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("JARL_LLM_API_KEY", "secret")
    monkeypatch.setenv("JARL_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("JARL_LLM_BASE_URL", raising=False)

    with pytest.raises(LLMConfigurationError, match="JARL_LLM_BASE_URL"):
        create_chat_model()


def test_create_chat_model_wraps_settings_errors_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("JARL_LLM_API_KEY", "secret")
    monkeypatch.setenv("JARL_LLM_BASE_URL", DEEPSEEK_BASE_URL)
    monkeypatch.delenv("JARL_LLM_MODEL", raising=False)

    with pytest.raises(LLMConfigurationError, match="JARL_LLM_MODEL"):
        create_chat_model()


def test_create_chat_model_wraps_settings_errors_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARL_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("JARL_LLM_BASE_URL", DEEPSEEK_BASE_URL)
    monkeypatch.setenv("JARL_LLM_MODEL", DEEPSEEK_MODEL)
    monkeypatch.delenv("JARL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError, match="DEEPSEEK_API_KEY"):
        create_chat_model()
