"""Factory for LangChain chat models used by the agentic workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from jarl.agentic.errors import LLMConfigurationError
from jarl.agentic.llm.config import LLMSettings, load_llm_settings_from_env

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = ["create_chat_model"]


def create_chat_model(
    settings: LLMSettings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    Args:
        settings: Optional explicit settings. When omitted, values are loaded
            from ``JARL_LLM_*`` environment variables.
        **kwargs: Extra provider kwargs forwarded to the underlying chat model
            constructor (for example ``timeout``).

    Raises:
        LLMConfigurationError: If settings are invalid or provider deps are missing.
    """
    active_settings = settings or _load_settings_safe()
    if active_settings.provider == "ollama":
        return _create_ollama_model(active_settings, **kwargs)
    return _create_openai_compatible_model(active_settings, **kwargs)


def _load_settings_safe() -> LLMSettings:
    try:
        return load_llm_settings_from_env()
    except (ValueError, ValidationError) as exc:
        raise LLMConfigurationError(str(exc)) from exc


def _create_ollama_model(settings: LLMSettings, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        msg = "Install jarl[agentic] to use the Ollama LLM provider."
        raise LLMConfigurationError(msg) from exc

    model_kwargs: dict[str, Any] = {
        "model": settings.model,
        "base_url": settings.base_url,
        **kwargs,
    }
    if settings.temperature is not None:
        model_kwargs["temperature"] = settings.temperature
    return ChatOllama(**model_kwargs)


def _create_openai_compatible_model(settings: LLMSettings, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        msg = "Install jarl[agentic] with langchain-openai for cloud LLM providers."
        raise LLMConfigurationError(msg) from exc

    model_kwargs: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        **kwargs,
    }
    if settings.temperature is not None:
        model_kwargs["temperature"] = settings.temperature
    if _is_deepseek_endpoint(settings.base_url):
        extra_body = dict(model_kwargs.get("extra_body") or {})
        extra_body.setdefault("thinking", {"type": "disabled"})
        model_kwargs["extra_body"] = extra_body
    return ChatOpenAI(**model_kwargs)


def _is_deepseek_endpoint(base_url: str | None) -> bool:
    """Return whether ``base_url`` targets the DeepSeek HTTP API."""
    return "deepseek" in (base_url or "").lower()
