"""Tests for the MCP tool factory and catalog registration."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

import pytest
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from jarl.agentic.tools.graph.checkpoints import ToolRequest as CheckpointsToolRequest
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.specs import ToolEntry, ToolRegistry, ToolSpec
from jarl.agentic.workflow import AgenticWorkflow
from jarl.mcp.bridge import mcp_callable_from_entry, uses_request_parameter
from jarl.mcp.server import create_server
from jarl.operations.train.run import RunResponse, ScheduleSummary
from jarl.training.launch import TRAIN_DETACH_ENV

_ALIASED_OVERRIDE_FIELDS = {
    ("graph_extend", "config_overrides", "ForkConfigOverrides"),
    ("graph_fork", "config_overrides", "ForkConfigOverrides"),
    ("train_resume", "config_overrides", "RetrainConfigOverrides"),
    ("train_run", "config_overrides", "RetrainConfigOverrides"),
}


def _handler_payload(payload: dict[str, object] | None = None, **kwargs: object) -> str:
    data = dict(payload or kwargs)
    return json.dumps({"got": data}, sort_keys=True)


def _discard_run(ctx: object, request: BaseModel) -> str:
    del ctx, request
    return ""


def _sample_entry(name: str, request_cls: type[BaseModel]) -> ToolEntry:
    return ToolEntry(
        spec=ToolSpec(name=name, description=f"{name} sample", labels=frozenset({"graph", "read"})),
        request_cls=request_cls,
        run_fn=_discard_run,
        build_handler=lambda _workflow: _handler_payload,
    )


class AlphaRequest(BaseModel):
    """Sample request with a required node id."""

    node_id: str = Field(description="Node to inspect.")
    checkpoint_step: int = Field(default=1, description="Checkpoint step.")


class BetaRequest(BaseModel):
    """Sample request with a different field."""

    branch: str = Field(description="Branch name.")


class TestMcpCallableFromEntry:
    def test_callables_bind_matching_request_cls_and_handler(self) -> None:
        alpha_calls: list[dict[str, object]] = []
        beta_calls: list[dict[str, object]] = []

        def alpha_handler(payload: dict[str, object] | None = None, **kwargs: object) -> str:
            alpha_calls.append(dict(payload or kwargs))
            return "alpha"

        def beta_handler(payload: dict[str, object] | None = None, **kwargs: object) -> str:
            beta_calls.append(dict(payload or kwargs))
            return "beta"

        alpha_entry = _sample_entry("alpha_tool", AlphaRequest)
        beta_entry = _sample_entry("beta_tool", BetaRequest)
        alpha_fn = mcp_callable_from_entry(alpha_entry, alpha_handler)
        beta_fn = mcp_callable_from_entry(beta_entry, beta_handler)

        alpha_result = _invoke_derived(alpha_fn, AlphaRequest, {"node_id": "n1", "checkpoint_step": 3})
        beta_result = _invoke_derived(beta_fn, BetaRequest, {"branch": "exp"})

        assert alpha_result == "alpha"
        assert beta_result == "beta"
        assert alpha_calls == [{"checkpoint_step": 3, "node_id": "n1"}]
        assert beta_calls == [{"branch": "exp"}]
        assert alpha_calls != beta_calls

    def test_loop_does_not_share_handler_model_or_metadata(self) -> None:
        captured: list[str] = []
        entries = [_sample_entry("alpha_tool", AlphaRequest), _sample_entry("beta_tool", BetaRequest)]
        fns: list[Callable[..., str]] = []
        for entry in entries:
            name = entry.spec.name

            def handler(
                payload: dict[str, object] | None = None,
                _name: str = name,
                **kwargs: object,
            ) -> str:
                del payload, kwargs
                captured.append(_name)
                return _name

            fns.append(mcp_callable_from_entry(entry, handler))

        alpha_fn, beta_fn = fns
        _invoke_derived(alpha_fn, AlphaRequest, {"node_id": "n1"})
        _invoke_derived(beta_fn, BetaRequest, {"branch": "exp"})

        assert captured == ["alpha_tool", "beta_tool"]
        assert alpha_fn is not beta_fn
        assert alpha_fn.__name__ == "alpha_tool"
        assert beta_fn.__name__ == "beta_tool"
        assert inspect.signature(alpha_fn) != inspect.signature(beta_fn)

    def test_direct_pydantic_annotation_or_flat_reconstruction(self) -> None:
        calls: list[dict[str, object]] = []

        def handler(payload: dict[str, object] | None = None, **kwargs: object) -> str:
            calls.append(dict(payload or kwargs))
            return "ok"

        entry = _sample_entry("alpha_tool", AlphaRequest)
        fn = mcp_callable_from_entry(entry, handler)
        payload = {"node_id": "n9", "checkpoint_step": 7}

        if uses_request_parameter(fn):
            parameter = inspect.signature(fn).parameters["request"]
            assert parameter.annotation is AlphaRequest
            result = fn(AlphaRequest.model_validate(payload))
        else:
            expected = AlphaRequest.model_validate(payload)
            result = fn(**payload)
            assert calls[-1] == expected.model_dump(mode="python")

        assert result == "ok"
        assert calls[-1]["node_id"] == "n9"
        assert calls[-1]["checkpoint_step"] == 7

    @pytest.mark.parametrize(
        ("tool_name", "payload"),
        [
            (
                "graph_fork",
                {
                    "branch": "alt",
                    "prepare": True,
                    "config_overrides": {"algorithm.learning_rate": 0.001},
                },
            ),
            (
                "graph_extend",
                {
                    "prepare": True,
                    "config_overrides": {"algorithm.total_timesteps": 512},
                },
            ),
            (
                "train_run",
                {
                    "node_id": "n1",
                    "config_overrides": {"algorithm.gamma": 0.97},
                },
            ),
            (
                "train_resume",
                {
                    "node_id": "n1",
                    "config_overrides": {"algorithm.nr_epochs": 2},
                },
            ),
        ],
    )
    def test_config_overrides_keep_schema_aliases(
        self,
        tool_name: str,
        payload: dict[str, object],
    ) -> None:
        """Keep dotted override keys when MCP dumps the ToolRequest for the handler."""
        captured: list[dict[str, object]] = []

        def handler(data: dict[str, object] | None = None, **kwargs: object) -> str:
            captured.append(dict(data or kwargs))
            return "ok"

        entry = REGISTRY.get(tool_name)
        fn = mcp_callable_from_entry(entry, handler)
        _invoke_derived(fn, entry.request_cls, payload)

        got = captured[0]["config_overrides"]
        expected = payload["config_overrides"]
        assert got == expected
        assert not any("__" in str(key) for key in got)
        entry.request_cls.model_validate(captured[0])

    def test_only_config_overrides_use_dotted_aliases(self) -> None:
        """Fail if another tool field starts using dotted aliases besides config_overrides."""
        found: set[tuple[str, str, str]] = set()
        for entry in REGISTRY:
            found.update(_aliased_nested_fields(entry.spec.name, entry.request_cls))
        assert found == _ALIASED_OVERRIDE_FIELDS

    @pytest.mark.parametrize(
        ("tool_name", "payload"),
        [
            (
                "train_run",
                {
                    "node_id": "n1",
                    "form": {"values": {"learning_rate": 0.003, "total_timesteps": 2048}},
                },
            ),
            (
                "train_resume",
                {
                    "node_id": "n1",
                    "form": {"values": {"nr_epochs": 2, "entropy_coef": 0.01}},
                },
            ),
        ],
    )
    def test_form_values_keep_submitted_keys(
        self,
        tool_name: str,
        payload: dict[str, object],
    ) -> None:
        """Keep preset-form hyperparameters when MCP dumps train requests."""
        captured = _capture_handler_payload(tool_name, payload)
        got = captured["form"]
        expected = payload["form"]
        assert got == expected
        REGISTRY.get(tool_name).request_cls.model_validate(captured)

    def test_set_reward_keeps_sparse_weight_keys(self) -> None:
        """Keep only the submitted reward weights when MCP dumps graph_set_reward."""
        payload: dict[str, object] = {"goal_reached": 1.0, "lava_clearance": 0.2}
        captured = _capture_handler_payload("graph_set_reward", payload)
        assert captured == payload
        REGISTRY.get("graph_set_reward").request_cls.model_validate(captured)


class TestRegisterCatalog:
    def test_registered_names_match_registry(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        server = create_server(workflow=workflow)
        names = {tool.name for tool in server._tool_manager.list_tools()}

        assert names == set(REGISTRY.names())

    def test_input_schema_matches_tool_request_without_request_wrapper(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        server = create_server(workflow=workflow)
        mcp_tool = server._tool_manager.get_tool("graph_checkpoints")
        assert mcp_tool is not None

        model_schema = CheckpointsToolRequest.model_json_schema()
        mcp_properties = mcp_tool.parameters.get("properties") or {}
        model_properties = model_schema.get("properties") or {}

        assert "request" not in mcp_properties
        assert set(mcp_properties) == set(model_properties)
        assert set(mcp_tool.parameters.get("required") or []) == set(model_schema.get("required") or [])
        for name, model_field in model_properties.items():
            assert mcp_properties[name].get("description") == model_field.get("description")
            assert _schema_type(mcp_properties[name]) == _schema_type(model_field)

    def test_read_and_mutation_via_registered_callables(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        server = create_server(workflow=workflow)
        summary_tool = server._tool_manager.get_tool("graph_summary")
        fork_tool = server._tool_manager.get_tool("graph_fork")
        checkout_tool = server._tool_manager.get_tool("graph_checkout")
        assert summary_tool is not None
        assert fork_tool is not None
        assert checkout_tool is not None

        summary = json.loads(summary_tool.fn())
        root_id = summary["nodes"][0]["node_id"]

        forked = json.loads(fork_tool.fn(branch="alt", label="child", prepare=True))
        checked_out = json.loads(checkout_tool.fn(node_id=root_id))

        assert summary["experiment"]["node_count"] == 1
        assert forked["branch"] == "alt"
        assert forked["node_id"] != root_id
        assert checked_out["node_id"] == root_id


class TestHelperRegistryRegistration:
    def test_can_register_sample_catalog(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        registry = ToolRegistry.from_module("tests.helpers.agentic_tools")

        server = create_server(workflow=workflow, registry=registry)
        names = {tool.name for tool in server._tool_manager.list_tools()}

        assert names == {"sample_ping", "sample_mutate"}


class TestTrainDetachRegistration:
    def test_train_run_enables_detach_env(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        seen: list[str | None] = []

        def fake_run(graph: object, request: object, **kwargs: object) -> RunResponse:
            del graph, request, kwargs
            seen.append(os.environ.get(TRAIN_DETACH_ENV))
            return RunResponse(
                node_id="n1",
                status="prepared",
                schedule=ScheduleSummary(
                    actual_rollout_updates=1,
                    actual_optimizer_updates=1,
                    actual_total_timesteps=1,
                ),
                detached=True,
                pid=1,
                log_path="runner_lab.log",
            )

        mocker.patch("jarl.agentic.tools.train.run.run", fake_run)
        previous = os.environ.get(TRAIN_DETACH_ENV)
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        server = create_server(workflow=workflow)
        tool = server._tool_manager.get_tool("train_run")
        assert tool is not None

        payload = json.loads(tool.fn())

        assert seen == ["1"]
        assert payload["detached"] is True
        assert os.environ.get(TRAIN_DETACH_ENV) == previous


def _capture_handler_payload(tool_name: str, payload: dict[str, object]) -> dict[str, object]:
    """Invoke the MCP callable and return the dict forwarded to the handler."""
    captured: list[dict[str, object]] = []

    def handler(data: dict[str, object] | None = None, **kwargs: object) -> str:
        captured.append(dict(data or kwargs))
        return "ok"

    entry = REGISTRY.get(tool_name)
    fn = mcp_callable_from_entry(entry, handler)
    _invoke_derived(fn, entry.request_cls, payload)
    return captured[0]


def _model_types(annotation: object) -> list[type[BaseModel]]:
    """Return Pydantic model classes inside a possibly optional annotation."""
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        return [
            argument
            for argument in get_args(annotation)
            if isinstance(argument, type) and issubclass(argument, BaseModel)
        ]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def _aliased_nested_fields(
    tool_name: str,
    model: type[BaseModel],
    prefix: str = "",
) -> set[tuple[str, str, str]]:
    """Return tool fields whose nested models expose JSON aliases distinct from Python names."""
    found: set[tuple[str, str, str]] = set()
    for name, field_info in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        for nested in _model_types(field_info.annotation):
            if any(info.alias is not None and info.alias != fname for fname, info in nested.model_fields.items()):
                found.add((tool_name, path, nested.__name__))
            found.update(_aliased_nested_fields(tool_name, nested, path))
    return found


def _invoke_derived(
    fn: Callable[..., str],
    request_cls: type[BaseModel],
    payload: dict[str, object],
) -> str:
    if uses_request_parameter(fn):
        return fn(request_cls.model_validate(payload))
    return fn(**payload)


def _schema_type(field_schema: dict[str, object]) -> object:
    if "anyOf" in field_schema:
        return field_schema["anyOf"]
    return field_schema.get("type")
