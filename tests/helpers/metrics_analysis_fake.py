"""Fake LLM supporting tool binding and structured metrics-analysis output."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from jarl.agentic.langgraph.graphs.subagents.metrics_analysis import MetricsAnalysisOutput
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel

__all__ = ["MetricsAnalysisFake"]


class MetricsAnalysisFake(ToolBindingFakeChatModel):
    """Parent tool-call fake that also serves structured metrics-analysis output."""

    phase: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=lambda: ["improving"])
    summary: str = "Eval return improves from the initial to the final window."
    response_format_json: bool = False

    def bind(self, **kwargs: Any) -> MetricsAnalysisFake:
        """Return a copy configured for JSON-object responses."""
        bound = self.model_copy()
        response_format = kwargs.get("response_format")
        bound.response_format_json = isinstance(response_format, dict) and response_format.get("type") == "json_object"
        return bound

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Any:
        """Return a runnable that emits a valid ``MetricsAnalysisOutput``."""
        del kwargs
        evidence = list(self.evidence_ids) or [
            "eval_return.window_initial.mean",
            "eval_return.window_final.mean",
        ]
        labels = list(self.labels)
        summary = self.summary
        parent = self

        class _StructuredRunner:
            def invoke(self, _input: Any, config: Any | None = None, **kw: Any) -> BaseModel:
                del config, kw
                if schema is MetricsAnalysisOutput or getattr(schema, "__name__", "") == "MetricsAnalysisOutput":
                    return MetricsAnalysisOutput(
                        labels=labels,  # type: ignore[arg-type]
                        summary=summary,
                        evidence=evidence,
                    )
                msg = f"Unsupported structured schema: {schema!r}"
                raise TypeError(msg)

            def bind_tools(
                self,
                tools: Sequence[dict[str, Any] | type | callable | BaseTool],
                *,
                tool_choice: str | None = None,
                **bind_kwargs: Any,
            ) -> Any:
                return parent.bind_tools(tools, tool_choice=tool_choice, **bind_kwargs)

        return _StructuredRunner()

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        if self.response_format_json:
            evidence = list(self.evidence_ids) or [
                "eval_return.window_initial.mean",
                "eval_return.window_final.mean",
            ]
            payload = MetricsAnalysisOutput(
                labels=self.labels,  # type: ignore[arg-type]
                summary=self.summary,
                evidence=evidence,
            )
            message = AIMessage(content=json.dumps(payload.model_dump()))
            return ChatResult(generations=[ChatGeneration(message=message)])
        if self.phase == 0:
            self.phase = 1
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "subagent_metrics_analysis",
                        "args": {},
                        "id": "metrics_analysis",
                        "type": "tool_call",
                    }
                ],
            )
        elif self.phase == 1:
            self.phase = 2
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "session_finish",
                        "args": {},
                        "id": "finish",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="Metrics analysis complete.")
        return ChatResult(generations=[ChatGeneration(message=message)])
