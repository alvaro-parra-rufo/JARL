"""Contract tests for ``graph_extend`` (manual §3, §6)."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from jarl.agentic.langgraph.tools import build_langchain_tools
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.graph.extend import TOOL_SPEC, CheckpointRefRequest, ToolRequest
from jarl.agentic.tools.specs import ToolRegistry
from jarl.training.config import RLRunConfig


class TestGraphExtendContract:
    def test_tool_spec_description_structural_extend_vs_fork(self) -> None:
        description = TOOL_SPEC.description.lower()

        assert TOOL_SPEC.name == "graph_extend"
        assert description.strip()
        assert "same branch" in description
        assert "branch head" in description or "branch's head" in description
        assert "does not run" in description
        assert "new branch" in description
        assert "graph_fork" not in description
        assert "use `" not in description

    def test_tool_spec_description_avoids_override_type_heuristics(self) -> None:
        description = TOOL_SPEC.description.lower()

        assert "hyperparameter" not in description
        assert "more timesteps" not in description
        assert "major config" not in description
        assert "train_run" not in description
        assert "config_overrides" not in description

    def test_tool_request_fields_have_non_empty_descriptions(self) -> None:
        for field_name, field_info in ToolRequest.model_fields.items():
            assert field_info.description, f"Missing description for {field_name!r}"

    def test_config_overrides_field_describes_omission_without_schema_paths(self) -> None:
        description = (ToolRequest.model_fields["config_overrides"].description or "").lower()

        assert "configuration changes" in description
        assert "inherit" in description
        assert "branch head" in description
        assert "fork policy" not in description
        assert "algorithm.total_timesteps" not in description
        assert "environment.env_id" not in description

    def test_checkpoint_ref_fields_guide_value_choice(self) -> None:
        node_field = CheckpointRefRequest.model_fields["node_id"]
        step_field = CheckpointRefRequest.model_fields["checkpoint_step"]

        assert "restore" in (node_field.description or "").lower()
        assert "typically" in (node_field.description or "").lower()
        assert "restore" in (step_field.description or "").lower()
        assert "that node" in (step_field.description or "").lower()

    def test_config_overrides_schema_exposes_typed_fork_policy_paths(self) -> None:
        schema = ToolRequest.model_json_schema()
        config_schema = schema["properties"]["config_overrides"]
        model_ref = next(branch["$ref"] for branch in config_schema["anyOf"] if "$ref" in branch)
        override_schema = schema["$defs"][model_ref.rsplit("/", maxsplit=1)[-1]]
        expected_paths = RLRunConfig.RETRAIN_MUTABLE_PATHS | RLRunConfig.FORK_EXTRA_MUTABLE_PATHS

        assert frozenset(override_schema["properties"]) == expected_paths
        assert override_schema["additionalProperties"] is False
        assert override_schema["properties"]["algorithm.total_timesteps"]["type"] == "integer"
        assert override_schema["properties"]["algorithm.learning_rate"]["type"] == "number"


class TestGraphExtendLangChainExposure:
    def test_langchain_tool_exposes_spec_and_request_schema(self, navix_tool_context: ToolContext) -> None:
        workflow = navix_tool_context.workflow
        registry = ToolRegistry.from_module("jarl.agentic.tools").filter_by(
            include_names={"graph_extend"},
        )
        tools = build_langchain_tools(registry, workflow)

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, StructuredTool)
        assert tool.name == "graph_extend"
        assert tool.description == TOOL_SPEC.description
        assert tool.args_schema == ToolRequest.model_json_schema()
