"""Functional scenarios: AgenticWorkflow effects on disk and audit trails."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from jarl.agentic.audit import audit_index_path, subagent_run_dir
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.subagent.metrics_analysis import ToolRequest as MetricsToolRequest
from jarl.agentic.tools.subagent.metrics_analysis import run_subagent_metrics_analysis
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel
from tests.helpers.metrics_analysis_fake import MetricsAnalysisFake


class TestWorkflowToolHandlersOnDisk:
    def test_create_root_handler_persists_manifest(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        handler = REGISTRY.get("graph_create_root").build_handler(workflow)

        payload = json.loads(handler({"label": "baseline", "branch": "main"}))

        reloaded = AgenticWorkflow.from_experiment(empty_experiment)
        assert reloaded.graph.as_networkx().number_of_nodes() == 1
        assert payload["node_id"] == reloaded.current_node_id

    def test_fork_extend_chain_updates_current_node_after_reload(self, demo_tree: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(demo_tree)
        graph = workflow.reload_graph()
        root_id = next(node_id for node_id in graph.as_networkx().nodes if graph.get_node(node_id).branch == "main")
        fork_handler = REGISTRY.get("graph_fork").build_handler(workflow)
        extend_handler = REGISTRY.get("graph_extend").build_handler(workflow)

        fork_payload = json.loads(
            fork_handler(
                {
                    "branch": "exp2",
                    "label": "child",
                    "prepare": True,
                    "from_node": root_id,
                }
            )
        )
        child_id = fork_payload["node_id"]

        extend_payload = json.loads(
            extend_handler(
                {
                    "branch": "exp2",
                    "label": "grandchild",
                    "prepare": True,
                }
            )
        )

        reloaded = AgenticWorkflow.from_experiment(demo_tree)
        assert extend_payload["parent_id"] == child_id
        assert reloaded.current_node_id == extend_payload["node_id"]

    def test_create_root_handler_accepts_max_episode_steps(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        handler = REGISTRY.get("graph_create_root").build_handler(workflow)

        payload = json.loads(
            handler(
                {
                    "label": "horizon",
                    "preset": "custom",
                    "form": {
                        "values": {
                            "env_id": "Navix-KeyCorridorS3R1-v0",
                            "max_episode_steps": 100,
                        }
                    },
                }
            )
        )

        reloaded = AgenticWorkflow.from_experiment(empty_experiment)
        config = reloaded.graph.resolve_config(reloaded.graph.get_node(payload["node_id"]))
        assert config.environment.env_id == "Navix-KeyCorridorS3R1-v0"
        assert config.environment.max_episode_steps == 100

    def test_subagent_metrics_analysis_writes_child_run(self, demo_tree: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(demo_tree)
        ctx = workflow.build_tool_context()
        workspace = ctx.graph.current_node
        workspace.log_scalar(0, "eval/episode_return", 1.0)
        workspace.log_scalar(1, "eval/episode_return", 3.0)
        workspace.log_scalar(2, "eval/episode_return", 5.0)
        ctx.graph.save()
        workflow.set_llm(
            MetricsAnalysisFake(
                messages=iter(()),
                evidence_ids=["eval_return.first.value", "eval_return.last.value"],
            )
        )

        run_subagent_metrics_analysis(ctx, MetricsToolRequest(focus="eval"))

        child_root = subagent_run_dir(demo_tree, workflow.thread_id, "metrics_analysis")
        assert (child_root / "manifest.json").is_file()


class TestWorkflowInvokeDiskEffects:
    def test_invoke_setup_create_root_persists_node(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "graph_create_root",
                                "args": {"label": "from_invoke"},
                                "id": "call_create_root",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="setup done"),
                    AIMessage(content="operate ok"),
                ]
            )
        )

        workflow.invoke(
            {"messages": [HumanMessage(content="crea el experimento")]},
            llm=llm,
        )

        reloaded = ExperimentGraph.from_directory(empty_experiment, config_cls=RLRunConfig)
        assert reloaded.as_networkx().number_of_nodes() == 1

    def test_mutations_record_audit_index(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment, audit_reads=False)
        handler = REGISTRY.get("graph_create_root").build_handler(workflow)
        handler({"label": "audited", "branch": "main"})

        audit_path = audit_index_path(empty_experiment)
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        kinds = {json.loads(line)["kind"] for line in lines}

        assert "mutation" in kinds
