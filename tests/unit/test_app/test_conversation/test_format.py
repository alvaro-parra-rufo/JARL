"""Tests for shared conversation formatters."""

from __future__ import annotations

from jarl.agentic.run_events import LlmEndEvent, LlmUsage
from jarl.app.lib.conversation import format_llm_generation_chip, format_llm_usage


class TestFormatLlmUsage:
    def test_input_output_label(self) -> None:
        assert format_llm_usage(LlmUsage(input_tokens=10, output_tokens=32, total_tokens=42)) == "10 in · 32 out"


class TestFormatLlmGenerationChip:
    def test_duration_and_usage(self) -> None:
        chip = format_llm_generation_chip(
            LlmEndEvent(
                node="operate",
                duration_ms=1200.0,
                usage=LlmUsage(input_tokens=10, output_tokens=32, total_tokens=42),
            )
        )

        assert chip == "1.2s · 10 in · 32 out"

    def test_fallback_label_without_metrics(self) -> None:
        assert format_llm_generation_chip(LlmEndEvent(node="operate")) == "LLM"
