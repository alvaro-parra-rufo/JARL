"""Environment-backed settings for agentic LLM providers."""

from __future__ import annotations

import os
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from jarl.config import BaseConfig

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "LLMProvider",
    "LLMSettings",
    "load_llm_settings_from_env",
]

LLMProvider = Literal["ollama", "openai_compatible"]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
_LEGACY_DEEPSEEK_PROVIDER = "deepseek"

_ENV_PROVIDER = "JARL_LLM_PROVIDER"
_ENV_MODEL = "JARL_LLM_MODEL"
_ENV_BASE_URL = "JARL_LLM_BASE_URL"
_ENV_API_KEY = "JARL_LLM_API_KEY"
_ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"


class LLMSettings(BaseConfig):
    """Resolved LLM provider configuration for ``create_chat_model``."""

    DEFAULT_PROVIDER: ClassVar[LLMProvider] = "ollama"

    provider: Annotated[
        LLMProvider,
        Field(
            description=(
                "Backend identifier: ``ollama`` for local Ollama or "
                "``openai_compatible`` for HTTP APIs with an OpenAI-style chat schema."
            ),
        ),
    ] = DEFAULT_PROVIDER
    model: Annotated[
        str,
        Field(description="Provider model name (for example ``qwen2.5:7b`` or ``deepseek-v4-flash``)."),
    ] = ""
    base_url: Annotated[
        str | None,
        Field(description="API base URL for HTTP providers; Ollama or OpenAI-compatible endpoints."),
    ] = None
    api_key: Annotated[
        str | None,
        Field(description="API key for OpenAI-compatible cloud providers."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(description="Optional sampling temperature forwarded to the chat model."),
    ] = None

    @model_validator(mode="before")
    @classmethod
    def _apply_provider_defaults(cls, data: object) -> object:
        """Fill provider-specific defaults before frozen model construction."""
        if not isinstance(data, dict):
            return data

        provider = data.get("provider", cls.DEFAULT_PROVIDER)
        model = data.get("model", "")
        if not model:
            if provider == "ollama":
                data["model"] = DEFAULT_OLLAMA_MODEL
            else:
                msg = f"{_ENV_MODEL} is required for provider {provider!r}."
                raise ValueError(msg)

        if data.get("base_url") is None and provider == "ollama":
            data["base_url"] = DEFAULT_OLLAMA_BASE_URL

        if data.get("api_key") is None and provider == "openai_compatible":
            data["api_key"] = os.environ.get(_ENV_API_KEY) or os.environ.get(_ENV_DEEPSEEK_API_KEY)

        return data

    @model_validator(mode="after")
    def _validate_cloud_provider(self) -> Self:
        """Require API keys and base URLs for OpenAI-compatible providers."""
        if self.provider == "openai_compatible" and not self.api_key:
            msg = (
                "An API key is required for provider 'openai_compatible'. "
                f"Set {_ENV_API_KEY} or {_ENV_DEEPSEEK_API_KEY}."
            )
            raise ValueError(msg)

        if self.provider == "openai_compatible" and not self.base_url:
            msg = f"{_ENV_BASE_URL} is required for provider 'openai_compatible'."
            raise ValueError(msg)

        return self


def load_llm_settings_from_env() -> LLMSettings:
    """Load ``LLMSettings`` from ``JARL_LLM_*`` environment variables."""
    provider_raw = os.environ.get(_ENV_PROVIDER, LLMSettings.DEFAULT_PROVIDER)
    if provider_raw == _LEGACY_DEEPSEEK_PROVIDER:
        msg = (
            f"{_ENV_PROVIDER}='deepseek' is not supported. "
            "Use 'openai_compatible' with "
            "JARL_LLM_BASE_URL=https://api.deepseek.com and "
            "JARL_LLM_MODEL set to your DeepSeek model name."
        )
        raise ValueError(msg)

    if provider_raw not in {"ollama", "openai_compatible"}:
        msg = f"Unsupported {_ENV_PROVIDER}={provider_raw!r}. Expected 'ollama' or 'openai_compatible'."
        raise ValueError(msg)

    model = os.environ.get(_ENV_MODEL, "")
    base_url = os.environ.get(_ENV_BASE_URL)
    api_key = os.environ.get(_ENV_API_KEY) or os.environ.get(_ENV_DEEPSEEK_API_KEY)
    temperature_raw = os.environ.get("JARL_LLM_TEMPERATURE")
    temperature = float(temperature_raw) if temperature_raw is not None else None

    return LLMSettings(
        provider=provider_raw,  # type: ignore[arg-type]
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
