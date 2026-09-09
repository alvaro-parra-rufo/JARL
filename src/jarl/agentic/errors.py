"""Errors raised by the agentic workflow runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from jarl.envs.navix.scenario_rewards import ScenarioRewardError

__all__ = [
    "AGENTIC_ENVIRONMENT_CONFIG_ERROR",
    "AgenticError",
    "ExperimentNotFoundError",
    "GraphMismatchError",
    "LLMConfigurationError",
    "LangGraphNotConfiguredError",
    "SessionError",
    "ToolError",
    "ToolErrorCode",
]

ToolErrorCode = Literal["validation", "domain", "internal"]

AGENTIC_ENVIRONMENT_CONFIG_ERROR = "Environment configuration is not valid for this run."
"""Neutral tool-facing message for invalid environment/overlay configuration."""


class AgenticError(Exception):
    """Base error for ``jarl.agentic``."""


class ToolError(AgenticError):
    """Structured error raised at the LangGraph tool boundary."""

    def __init__(
        self,
        *,
        tool: str,
        code: ToolErrorCode,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a tool boundary error."""
        super().__init__(message)
        self.tool = tool
        self.code = code
        self.message = message
        self.cause = cause

    @classmethod
    def from_exception(cls, tool: str, exc: Exception) -> ToolError:
        """Wrap ``exc`` as a tool boundary error for ``tool``."""
        if isinstance(exc, cls):
            return exc
        if isinstance(exc, ScenarioRewardError):
            return cls(
                tool=tool,
                code="domain",
                message=AGENTIC_ENVIRONMENT_CONFIG_ERROR,
                cause=exc,
            )
        return cls(
            tool=tool,
            code=cls.code_for_exception(exc),
            message=str(exc),
            cause=exc,
        )

    @staticmethod
    def code_for_exception(exc: Exception) -> ToolErrorCode:
        """Map an exception to a tool error code."""
        if isinstance(exc, ValidationError):
            return "validation"
        if isinstance(exc, ScenarioRewardError):
            return "domain"
        if isinstance(exc, AgenticError):
            return "domain"
        if isinstance(exc, (ValueError, KeyError, RuntimeError)):
            return "domain"
        return "internal"


class ExperimentNotFoundError(AgenticError):
    """Raised when an experiment directory lacks a valid manifest."""


class SessionError(AgenticError):
    """Raised when agent session metadata cannot be loaded or persisted."""


class GraphMismatchError(AgenticError):
    """Raised when an in-memory graph does not belong to the active experiment."""


class LangGraphNotConfiguredError(AgenticError):
    """Raised when LangGraph compile or invoke is not wired yet."""


class LLMConfigurationError(AgenticError):
    """Raised when LLM provider settings or dependencies are invalid."""
