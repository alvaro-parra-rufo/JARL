"""Experiment-level case snapshot persisted as ``case.json``."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from jarl.config import BaseConfig

__all__ = ["ExperimentCaseMetadata"]


class ExperimentCaseMetadata(BaseConfig):
    """Operator-facing snapshot of which experiment case materialized a graph.

    This file is for tests, CLI, and later validators. It is not an LLM tool
    payload.
    """

    case_id: Annotated[str, Field(description="Stable experiment case identifier.")]
    env_id: Annotated[str, Field(description="Gymnasium environment id of the materialized root.")]
    scenario_reward_id: Annotated[
        str,
        Field(description="Registered overlay identifier copied from the root config."),
    ]
    scenario_reward_version: Annotated[
        int,
        Field(description="Registered overlay version copied from the root config."),
    ]
