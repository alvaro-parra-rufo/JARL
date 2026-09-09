"""Structured-output mode selection for LangChain chat models."""

from __future__ import annotations

from typing import Literal

from jarl.agentic.llm.config import LLMSettings

StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]

__all__ = ["StructuredOutputMethod", "resolve_structured_output_method"]


def resolve_structured_output_method(settings: LLMSettings) -> StructuredOutputMethod:
    """Return a structured-output mode the provider is known to accept.

    DeepSeek and most OpenAI-compatible hosts only implement ``json_object``
    (LangChain ``json_mode``), not ``json_schema``. OpenAI's first-party API
    keeps ``json_schema`` for stricter enforcement.
    """
    if settings.provider == "ollama":
        return "json_mode"
    base_url = (settings.base_url or "").lower()
    if "deepseek" in base_url:
        return "json_mode"
    if "api.openai.com" in base_url:
        return "json_schema"
    return "json_mode"
