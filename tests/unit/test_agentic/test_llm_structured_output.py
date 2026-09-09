"""Tests for structured-output mode resolution."""

from __future__ import annotations

from jarl.agentic.llm.config import LLMSettings
from jarl.agentic.llm.structured_output import resolve_structured_output_method


def test_deepseek_openai_compatible_uses_json_mode() -> None:
    settings = LLMSettings(
        provider="openai_compatible",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="test-key",
    )
    assert resolve_structured_output_method(settings) == "json_mode"


def test_openai_first_party_uses_json_schema() -> None:
    settings = LLMSettings(
        provider="openai_compatible",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
    )
    assert resolve_structured_output_method(settings) == "json_schema"


def test_ollama_uses_json_mode() -> None:
    settings = LLMSettings(provider="ollama", model="qwen2.5:7b")
    assert resolve_structured_output_method(settings) == "json_mode"
