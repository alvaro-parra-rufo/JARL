"""Contract tests for ``train_resume``."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from jarl.agentic.langgraph.tools import build_langchain_tools
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.specs import ToolRegistry
from jarl.agentic.tools.train.resume import TOOL_SPEC, ToolRequest
from jarl.training.config import RLRunConfig


class TestTrainResumeContract:
    def test_tool_spec_description_covers_resume_vs_run(self) -> None:
        description = TOOL_SPEC.description.lower()

        assert TOOL_SPEC.name == "train_resume"
        assert description.strip()
        assert "failed" in description or "interrupted" in description
        assert "checkpoint" in description
        assert "does not add" in description
        assert "not for first" in description or "not for" in description
        assert "train_run" not in description
        assert "use `" not in description

    def test_tool_spec_avoids_schema_prescription(self) -> None:
        description = TOOL_SPEC.description.lower()

        assert "config_overrides" not in description
        assert "retrain policy" not in description

    def test_tool_request_fields_have_non_empty_descriptions(self) -> None:
        for field_name, field_info in ToolRequest.model_fields.items():
            assert field_info.description, f"Missing description for {field_name!r}"

    def test_config_overrides_field_describes_omission_without_schema_paths(self) -> None:
        description = (ToolRequest.model_fields["config_overrides"].description or "").lower()

        assert "configuration changes" in description
        assert "inherit" in description
        assert "retrain policy" not in description
        assert "environment.env_id" not in description

    def test_config_overrides_schema_exposes_typed_retrain_policy_paths(self) -> None:
        schema = ToolRequest.model_json_schema()
        config_schema = schema["properties"]["config_overrides"]
        model_ref = next(branch["$ref"] for branch in config_schema["anyOf"] if "$ref" in branch)
        override_schema = schema["$defs"][model_ref.rsplit("/", maxsplit=1)[-1]]
        expected_paths = RLRunConfig.RETRAIN_MUTABLE_PATHS

        assert frozenset(override_schema["properties"]) == expected_paths
        assert override_schema["additionalProperties"] is False
        assert "environment.env_id" not in override_schema["properties"]


class TestTrainResumeLangChainExposure:
    def test_langchain_tool_exposes_spec_and_request_schema(self, navix_tool_context: ToolContext) -> None:
        workflow = navix_tool_context.workflow
        registry = ToolRegistry.from_module("jarl.agentic.tools").filter_by(
            include_names={"train_resume"},
        )
        tools = build_langchain_tools(registry, workflow)

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, StructuredTool)
        assert tool.name == "train_resume"
        assert tool.description == TOOL_SPEC.description
        assert tool.args_schema == ToolRequest.model_json_schema()
