"""Tests for env read agentic tools."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.env.navix_maps import ToolRequest, run_env_navix_maps
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow


class TestEnvNavixMapsTool:
    def test_navix_maps_lists_catalog(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_env_navix_maps(tool_context, ToolRequest()))

        assert len(payload["maps"]) > 0
        assert payload["maps"][0]["env_id"].startswith("Navix-")

    def test_navix_maps_filters_by_categories(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_env_navix_maps(tool_context, ToolRequest(categories=["door_key", "empty"])))

        assert len(payload["maps"]) > 0
        assert {item["category"] for item in payload["maps"]} == {"door_key", "empty"}

    def test_navix_maps_filters_by_query(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_env_navix_maps(tool_context, ToolRequest(query="llave")))

        assert len(payload["maps"]) > 0
        assert {item["category"] for item in payload["maps"]} == {"door_key", "key_corridor"}
        assert all(1 <= item["difficulty"] <= 100 for item in payload["maps"])

    def test_navix_maps_unknown_category_lists_real_keys(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        handler = REGISTRY.get("env_navix_maps").build_handler(workflow)

        payload = json.loads(handler({"categories": ["puerta_llave"]}))

        assert "error" in payload
        assert "door_key" in payload["error"]
        assert "Look at the user's request context" in payload["error"]

    def test_navix_maps_accepts_comma_separated_categories(
        self,
        tool_context: ToolContext,
    ) -> None:
        payload = json.loads(
            run_env_navix_maps(tool_context, ToolRequest.model_validate({"categories": "door_key, empty"}))
        )

        assert len(payload["maps"]) > 0
        assert {item["category"] for item in payload["maps"]} == {"door_key", "empty"}

    def test_navix_maps_rejects_categories_object(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        handler = REGISTRY.get("env_navix_maps").build_handler(workflow)

        payload = json.loads(handler({"categories": {"door_key": True}}))

        assert "error" in payload
        assert "comma-separated" in payload["error"]


class TestEnvNavixMapsAudit:
    def test_env_navix_maps_audits_when_reads_enabled(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        handler = REGISTRY.get("env_navix_maps").build_handler(workflow)

        handler({"categories": ["lava_gap"]})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "env_navix_maps"
        assert event["kind"] == "read"
