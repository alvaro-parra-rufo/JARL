"""Pure formatters for conversation chrome shared across Runner Lab views."""

from __future__ import annotations

from jarl.agentic.run_events import LlmEndEvent, LlmUsage

__all__ = ["format_llm_generation_chip", "format_llm_usage"]


def format_llm_usage(usage: LlmUsage) -> str:
    """Return a compact input/output token label for one generation."""
    return f"{usage.input_tokens} in · {usage.output_tokens} out"


def format_llm_generation_chip(generation: LlmEndEvent) -> str:
    """Return a compact duration and usage chip for one ``llm_end`` event."""
    parts: list[str] = []
    if generation.duration_ms is not None and generation.duration_ms > 0:
        parts.append(f"{generation.duration_ms / 1000:.1f}s")
    if generation.usage is not None:
        parts.append(format_llm_usage(generation.usage))
    return " · ".join(parts) if parts else "LLM"
