"""Shared fixtures for ``tests/llm_eval/test_agentic``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jarl.agentic.llm import (
    MAIN_COMPONENT_ID,
    create_chat_model,
    load_llm_catalog,
    resolve_llm_yaml_path,
)
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import EnvironmentConfig, RLRunConfig
from jarl.utils.env import load_project_env

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@pytest.fixture()
def navix_tool_context(tmp_path: Path) -> ToolContext:
    """Tool context with a Navix Empty root on ``main``."""
    exp_dir = tmp_path / "exp"
    config = RLRunConfig(environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0"))
    graph = ExperimentGraph(exp_dir, base_config=config)
    graph.create_root(config=config, branch="main", label="baseline", prepare=True)
    graph.save()
    workflow = AgenticWorkflow.from_experiment(exp_dir)
    return workflow.build_tool_context()


@pytest.fixture(scope="session")
def agentic_case_llm() -> BaseChatModel:
    """Create the configured real LLM or skip the opt-in case suite.

    Loads the project ``.env``, then prefers ``jarl/agentic/llm/profiles`` catalog
    settings for ``main``. Falls back to ``JARL_LLM_*`` env synthesis.
    """
    load_project_env()
    catalog_path = resolve_llm_yaml_path()
    has_env_provider = os.environ.get("JARL_LLM_PROVIDER") is not None
    if catalog_path is None and not has_env_provider:
        pytest.skip(
            "Configure src/jarl/agentic/llm/profiles/custom.yaml (or default.yaml) "
            "or set JARL_LLM_PROVIDER to run LLM eval scenarios."
        )
    if catalog_path is not None:
        resolved = load_llm_catalog().resolve(MAIN_COMPONENT_ID)
        return create_chat_model(resolved.settings)
    return create_chat_model()
