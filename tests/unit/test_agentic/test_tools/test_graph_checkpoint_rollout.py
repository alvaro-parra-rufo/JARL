"""Tests for the graph checkpoint rollout agentic tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from jarl.agentic.audit import audit_index_path
from jarl.agentic.langgraph.graphs.experiment import (
    OPERATE_INCLUDE_LABELS,
    SETUP_INCLUDE_LABELS,
)
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.graph.checkpoint_rollout import (
    TOOL_SPEC,
    ToolRequest,
    run_graph_checkpoint_rollout,
)
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow
from jarl.operations.contracts.constants import CHECKPOINT_ROLLOUT_MAX_STEPS
from jarl.operations.graph.checkpoint_rollout import CheckpointRolloutRequest

_COMPACT_RESPONSE: dict[str, object] = {
    "identity": {
        "node_id": "main_root_ab12cd34",
        "checkpoint_step": 8,
        "seed": 17,
        "rollout_id": "0123456789abcdef",
    },
    "outcome": {
        "return_total": 1.0,
        "length": 3,
        "terminated": True,
        "truncated": False,
        "reason": "termination",
    },
    "artifact_paths": (
        "rollouts/0123456789abcdef/rollout.json",
        "rollouts/0123456789abcdef/trace.npz",
        "rollouts/0123456789abcdef/rollout.mp4",
    ),
    "cache_hit": False,
}
"""Compact operation payload used by tool adapter tests."""


class TestGraphCheckpointRolloutTool:
    """The adapter maps every LLM field to its dedicated operation."""

    def test_maps_request_and_returns_compact_json(
        self,
        mocker: MockerFixture,
        tool_context: ToolContext,
    ) -> None:
        response = mocker.Mock()
        response.to_compact_dict.return_value = _COMPACT_RESPONSE
        operation = mocker.patch(
            "jarl.agentic.tools.graph.checkpoint_rollout.checkpoint_rollout",
            return_value=response,
        )
        request = ToolRequest(
            node_id="main_root_ab12cd34",
            checkpoint_step=8,
            seed=17,
            env_id="Navix-DoorKey-5x5-v0",
            max_steps=64,
            record_video=True,
            video_view_mode="first_person",
        )

        payload = json.loads(
            run_graph_checkpoint_rollout(tool_context, request),
        )

        operation.assert_called_once_with(
            tool_context.graph,
            CheckpointRolloutRequest(
                node_id="main_root_ab12cd34",
                checkpoint_step=8,
                seed=17,
                env_id="Navix-DoorKey-5x5-v0",
                max_steps=64,
                record_video=True,
                video_view_mode="first_person",
            ),
        )
        assert payload == {
            **_COMPACT_RESPONSE,
            "artifact_paths": list(_COMPACT_RESPONSE["artifact_paths"]),
        }
        assert "trace" not in payload
        assert "rgb_frames" not in str(payload)


class TestGraphCheckpointRolloutSchema:
    """Pydantic exposes and enforces the bounded operation contract."""

    @pytest.mark.parametrize(
        ("payload", "field_name"),
        [
            pytest.param(
                {"checkpoint_step": -1, "seed": 0},
                "checkpoint_step",
                id="negative_checkpoint",
            ),
            pytest.param(
                {"checkpoint_step": 1, "seed": 0, "max_steps": 0},
                "max_steps",
                id="zero_horizon",
            ),
            pytest.param(
                {
                    "checkpoint_step": 1,
                    "seed": 0,
                    "max_steps": CHECKPOINT_ROLLOUT_MAX_STEPS + 1,
                },
                "max_steps",
                id="horizon_cap",
            ),
            pytest.param(
                {
                    "checkpoint_step": 1,
                    "seed": 0,
                    "video_view_mode": "side",
                },
                "video_view_mode",
                id="view_mode",
            ),
        ],
    )
    def test_rejects_invalid_payload(
        self,
        payload: dict[str, object],
        field_name: str,
    ) -> None:
        with pytest.raises(ValidationError, match=field_name):
            ToolRequest.model_validate(payload)

    def test_schema_exposes_horizon_cap_and_view_enum(self) -> None:
        properties = ToolRequest.model_json_schema()["properties"]
        max_steps_schema = next(
            branch for branch in properties["max_steps"]["anyOf"] if branch.get("type") == "integer"
        )

        assert max_steps_schema["minimum"] == 1
        assert max_steps_schema["maximum"] == CHECKPOINT_ROLLOUT_MAX_STEPS
        assert properties["video_view_mode"]["enum"] == [
            "full",
            "first_person",
        ]
        assert any(branch.get("type") == "string" for branch in properties["env_id"]["anyOf"])
        assert "compatibility" in properties["env_id"]["description"]

    def test_description_preserves_semantic_boundaries(self) -> None:
        description = TOOL_SPEC.description.lower()

        assert "exact checkpoint" in description
        assert "not for checkpoint selection" in description
        assert "does not train" in description
        assert "does not" in description and "mutate graph topology" in description


class TestGraphCheckpointRolloutHandler:
    """The shared handler validates, audits, and preserves graph state."""

    def test_success_audits_as_read_without_changing_node(
        self,
        mocker: MockerFixture,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(
            prepared_experiment,
            audit_reads=True,
        )
        response = mocker.Mock()
        response.to_compact_dict.return_value = _COMPACT_RESPONSE
        mocker.patch(
            "jarl.agentic.tools.graph.checkpoint_rollout.checkpoint_rollout",
            return_value=response,
        )
        handler = REGISTRY.get("graph_checkpoint_rollout").build_handler(
            workflow,
        )
        node_before = workflow.current_node_id

        payload = json.loads(
            handler(
                {
                    "checkpoint_step": 8,
                    "seed": 17,
                    "record_video": True,
                }
            )
        )

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert payload["identity"]["seed"] == 17
        assert event["tool"] == "graph_checkpoint_rollout"
        assert event["kind"] == "read"
        assert event["node_before"] == node_before
        assert event["node_after"] == node_before
        assert workflow.current_node_id == node_before

    def test_launch_option_overrides_llm_request_fields(
        self,
        mocker: MockerFixture,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(
            prepared_experiment,
            audit_reads=True,
            tool_overrides={"graph_checkpoint_rollout": {"record_video": True}},
        )
        response = mocker.Mock()
        response.to_compact_dict.return_value = _COMPACT_RESPONSE
        operation = mocker.patch(
            "jarl.agentic.tools.graph.checkpoint_rollout.checkpoint_rollout",
            return_value=response,
        )
        handler = REGISTRY.get("graph_checkpoint_rollout").build_handler(workflow)

        handler({"checkpoint_step": 8, "seed": 17, "record_video": False})

        request = operation.call_args.args[1]
        assert request.record_video is True

    def test_success_skips_audit_when_reads_are_disabled(
        self,
        mocker: MockerFixture,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(
            prepared_experiment,
            audit_reads=False,
        )
        response = mocker.Mock()
        response.to_compact_dict.return_value = _COMPACT_RESPONSE
        mocker.patch(
            "jarl.agentic.tools.graph.checkpoint_rollout.checkpoint_rollout",
            return_value=response,
        )
        handler = REGISTRY.get("graph_checkpoint_rollout").build_handler(
            workflow,
        )

        handler({"checkpoint_step": 8, "seed": 17})

        assert not audit_index_path(prepared_experiment).exists()

    def test_domain_error_is_serialized_and_audited(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(
            prepared_experiment,
            audit_reads=True,
        )
        handler = REGISTRY.get("graph_checkpoint_rollout").build_handler(
            workflow,
        )

        payload = json.loads(
            handler(
                {
                    "checkpoint_step": 999,
                    "seed": 0,
                    "max_steps": 3,
                }
            )
        )

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert payload["code"] == "domain"
        assert "step 999" in payload["error"]
        assert event["kind"] == "error"
        assert event["result"]["code"] == "domain"

    def test_validation_error_is_serialized_and_audited(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(
            prepared_experiment,
            audit_reads=False,
        )
        handler = REGISTRY.get("graph_checkpoint_rollout").build_handler(
            workflow,
        )

        payload = json.loads(
            handler(
                {
                    "checkpoint_step": 1,
                    "seed": 0,
                    "max_steps": 0,
                }
            )
        )

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert payload["code"] == "validation"
        assert "max_steps" in payload["error"]
        assert event["kind"] == "error"
        assert event["result"]["code"] == "validation"


class TestGraphCheckpointRolloutDiscovery:
    """Registry labels expose the tool only in the operate phase."""

    def test_visible_in_operate_and_hidden_from_setup(self) -> None:
        operate_tools = REGISTRY.filter_by(
            include_labels=set(OPERATE_INCLUDE_LABELS),
        )
        setup_tools = REGISTRY.filter_by(
            include_labels=set(SETUP_INCLUDE_LABELS),
        )

        assert "graph_checkpoint_rollout" in operate_tools.names()
        assert "graph_checkpoint_rollout" not in setup_tools.names()
