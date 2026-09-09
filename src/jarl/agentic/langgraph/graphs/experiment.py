"""Main experiment LangGraph with bootstrap, setup, and operate phases."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jarl.agentic.langgraph.state import AgentState
from jarl.agentic.langgraph.tools import build_langchain_tools
from jarl.agentic.prompts import (
    OPERATE_PHASE_RULES,
    SETUP_PHASE_RULES,
    build_phase_system_prompt,
)
from jarl.agentic.session import CompiledNodeInfo
from jarl.agentic.tools.specs import merge_tool_filters

if TYPE_CHECKING:
    from jarl.agentic.tools.context import ToolContext
    from jarl.agentic.tools.specs import ToolFilterBy, ToolRegistry
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "GRAPH_ID",
    "OPERATE_INCLUDE_LABELS",
    "SETUP_INCLUDE_LABELS",
    "build_graph",
    "resolve_experiment_phase",
    "route_after_setup",
]

GRAPH_ID = "experiment"

SETUP_INCLUDE_LABELS: frozenset[str] = frozenset({"setup"})
OPERATE_INCLUDE_LABELS: frozenset[str] = frozenset({"operate"})

PostSetupRoute = Literal["operate", "end"]
ExperimentPhase = Literal["setup", "operate"]


def resolve_experiment_phase(workflow: AgenticWorkflow) -> ExperimentPhase:
    """Return ``setup`` for empty experiments, otherwise ``operate``."""
    workflow.reload_graph()
    if workflow.graph.as_networkx().number_of_nodes() == 0:
        return "setup"
    return "operate"


def route_after_setup(workflow: AgenticWorkflow, _state: AgentState) -> PostSetupRoute:
    """Route to ``operate`` when setup created experiment nodes, otherwise end."""
    if resolve_experiment_phase(workflow) == "operate":
        return "operate"
    return "end"


def build_graph(
    ctx: ToolContext,
    registry: ToolRegistry,
    llm: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    filter_by: ToolFilterBy | None = None,
    objective: str | None = None,
) -> CompiledStateGraph:
    """Build the experiment graph with phase-specific tool visibility.

    ``filter_by`` is applied after the phase label filter. Drivers such as chat
    can hide finish tools with ``filter_by={"exclude_labels": {"finish"}}``.
    ``objective`` is appended to both phase system prompts when set; chat omits it.
    """
    workflow = ctx.workflow
    extra_filter = filter_by or {}
    setup_filter = merge_tool_filters({"include_labels": set(SETUP_INCLUDE_LABELS)}, extra_filter)
    operate_filter = merge_tool_filters({"include_labels": set(OPERATE_INCLUDE_LABELS)}, extra_filter)
    setup_tools = build_langchain_tools(registry.filter_by(**setup_filter), workflow)
    operate_tools = build_langchain_tools(registry.filter_by(**operate_filter), workflow)
    setup_system_prompt = build_phase_system_prompt(SETUP_PHASE_RULES, objective=objective)
    operate_system_prompt = build_phase_system_prompt(OPERATE_PHASE_RULES, objective=objective)
    setup_agent = create_agent(
        llm,
        setup_tools,
        system_prompt=setup_system_prompt,
        checkpointer=checkpointer,
        name="setup",
    )
    operate_agent = create_agent(
        llm,
        operate_tools,
        system_prompt=operate_system_prompt,
        checkpointer=checkpointer,
        name="operate",
    )
    workflow.replace_compiled_nodes(
        {
            "setup": CompiledNodeInfo.from_bind(
                system_prompt=setup_system_prompt,
                filter_by=setup_filter,
                tool_names=(tool.name for tool in setup_tools),
            ),
            "operate": CompiledNodeInfo.from_bind(
                system_prompt=operate_system_prompt,
                filter_by=operate_filter,
                tool_names=(tool.name for tool in operate_tools),
            ),
        }
    )

    def bootstrap(_state: AgentState) -> dict[str, object]:
        """Routing-only node; phase selection happens in conditional edges."""
        return {}

    def route_phase(_state: AgentState) -> ExperimentPhase:
        return resolve_experiment_phase(workflow)

    graph = StateGraph(AgentState)
    graph.add_node("bootstrap", bootstrap)
    graph.add_node("setup", setup_agent)
    graph.add_node("operate", operate_agent)
    graph.add_edge(START, "bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        route_phase,
        {"setup": "setup", "operate": "operate"},
    )
    graph.add_conditional_edges(
        "setup",
        partial(route_after_setup, workflow),
        {"operate": "operate", "end": END},
    )
    graph.add_edge("operate", END)
    return graph.compile(checkpointer=checkpointer)
