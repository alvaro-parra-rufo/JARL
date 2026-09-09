"""Tests for workflow LLM catalog binding, cache, and overrides."""

from __future__ import annotations

from pathlib import Path

from pytest_mock import MockerFixture

from jarl.agentic.llm import LLMCatalog, LLMSettings
from jarl.agentic.workflow import AgenticWorkflow


def _settings(model: str) -> LLMSettings:
    return LLMSettings(provider="ollama", model=model)


class TestWorkflowLLMCatalog:
    def test_set_llm_override_skips_catalog(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        fake = object()
        create = mocker.patch("jarl.agentic.workflow.create_chat_model")
        workflow.set_llm_catalog(
            LLMCatalog(
                profiles={"cheap": _settings("cheap")},
                assignments={},
                fallback="cheap",
            )
        )
        workflow.set_llm(fake)

        assert workflow.get_chat_model("main") is fake
        assert workflow.get_chat_model("subagents/metrics_analysis") is fake
        create.assert_not_called()

    def test_caches_clients_by_effective_settings(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        client_a = object()
        client_b = object()
        create = mocker.patch(
            "jarl.agentic.workflow.create_chat_model",
            side_effect=[client_a, client_b],
        )
        workflow.set_llm_catalog(
            LLMCatalog(
                profiles={
                    "powerful": _settings("big"),
                    "cheap": _settings("small"),
                    "alias_small": _settings("small"),
                },
                assignments={
                    "main": "powerful",
                    "subagents/*": "cheap",
                    "other": "alias_small",
                },
                fallback="cheap",
            )
        )

        main = workflow.get_chat_model("main")
        sub = workflow.get_chat_model("subagents/x")
        other = workflow.get_chat_model("other")

        assert main is client_a
        assert sub is client_b
        assert other is client_b
        assert create.call_count == 2

    def test_resolves_distinct_profiles(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        mocker.patch(
            "jarl.agentic.workflow.create_chat_model",
            side_effect=lambda settings, **_kwargs: f"client:{settings.model}",
        )
        workflow.set_llm_catalog(
            LLMCatalog(
                profiles={
                    "powerful": _settings("big"),
                    "cheap": _settings("small"),
                    "metrics_analysis": _settings("metrics"),
                },
                assignments={
                    "main": "powerful",
                    "subagents/*": "cheap",
                    "subagents/metrics_analysis": "metrics_analysis",
                },
                fallback="cheap",
            )
        )

        assert workflow.get_chat_model("main") == "client:big"
        assert workflow.last_resolved_llm is not None
        assert workflow.last_resolved_llm.profile_name == "powerful"
        assert workflow.get_chat_model("subagents/metrics_analysis") == "client:metrics"
        assert workflow.last_resolved_llm.profile_name == "metrics_analysis"
