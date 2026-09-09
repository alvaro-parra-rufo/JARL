"""Factory that turns agentic `ToolEntry` objects into FastMCP callables."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool as FastMcpTool
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from jarl.agentic.tools.specs import ToolEntry, ToolRegistry
from jarl.agentic.workflow import AgenticWorkflow
from jarl.training.launch import TRAIN_DETACH_ENV

__all__ = [
    "mcp_callable_from_entry",
    "register_tools",
    "uses_request_parameter",
]

_DETACH_TRAIN_TOOLS = frozenset({"train_run", "train_resume"})


def register_tools(
    server: FastMCP,
    registry: ToolRegistry,
    workflow: AgenticWorkflow,
) -> None:
    """Register every catalog entry on `server` using handlers built once."""
    for entry in registry:
        handler = entry.build_handler(workflow)
        if entry.spec.name in _DETACH_TRAIN_TOOLS:
            handler = _enable_train_detach(handler)
        fn = mcp_callable_from_entry(entry, handler)
        server.add_tool(
            fn,
            name=entry.spec.name,
            description=entry.spec.description,
            structured_output=False,
        )


def _enable_train_detach(handler: Callable[..., str]) -> Callable[..., str]:
    """Force detached training for MCP ``train_run`` / ``train_resume`` calls."""

    def wrapped(payload: dict[str, object] | None = None, **kwargs: object) -> str:
        previous = os.environ.get(TRAIN_DETACH_ENV)
        os.environ[TRAIN_DETACH_ENV] = "1"
        try:
            if payload is not None:
                return handler(payload)
            return handler(**kwargs)
        finally:
            if previous is None:
                os.environ.pop(TRAIN_DETACH_ENV, None)
            else:
                os.environ[TRAIN_DETACH_ENV] = previous

    return wrapped


def mcp_callable_from_entry(
    entry: ToolEntry,
    handler: Callable[..., str],
) -> Callable[..., str]:
    """Build an MCP callable derived from `entry.request_cls`.

    Uses the Pydantic model as a single `request` parameter when FastMCP keeps
    fields at the schema root. Otherwise builds an equivalent keyword-only
    signature and reconstructs `request_cls` before calling `handler`.
    """
    nested = _make_request_param_callable(
        handler=handler,
        request_cls=entry.request_cls,
        name=entry.spec.name,
        description=entry.spec.description,
    )
    schema = FastMcpTool.from_function(
        nested,
        name=entry.spec.name,
        description=entry.spec.description,
        structured_output=False,
    ).parameters
    if _has_artificial_request_wrapper(schema, entry.request_cls):
        return _make_flat_fields_callable(
            handler=handler,
            request_cls=entry.request_cls,
            name=entry.spec.name,
            description=entry.spec.description,
        )
    return nested


def uses_request_parameter(fn: Callable[..., str]) -> bool:
    """Return whether `fn` takes a single `request` Pydantic parameter."""
    return list(inspect.signature(fn).parameters) == ["request"]


def _has_artificial_request_wrapper(
    parameters: dict[str, object],
    request_cls: type[BaseModel],
) -> bool:
    """Return whether FastMCP wrapped the whole model under `request`."""
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return False
    if set(properties) != {"request"}:
        return False
    return "request" not in request_cls.model_fields


def _make_request_param_callable(
    *,
    handler: Callable[..., str],
    request_cls: type[BaseModel],
    name: str,
    description: str,
) -> Callable[..., str]:
    """Wrap `handler` with a single `request: request_cls` parameter."""

    def invoke(request: BaseModel) -> str:
        return handler(_payload_for_handler(request))

    invoke.__name__ = name
    invoke.__qualname__ = name
    invoke.__doc__ = description
    invoke.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                "request",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=request_cls,
            )
        ],
        return_annotation=str,
    )
    invoke.__annotations__ = {"request": request_cls, "return": str}
    return invoke


def _make_flat_fields_callable(
    *,
    handler: Callable[..., str],
    request_cls: type[BaseModel],
    name: str,
    description: str,
) -> Callable[..., str]:
    """Wrap `handler` with keyword fields copied from `request_cls`."""

    def invoke(**kwargs: object) -> str:
        request = request_cls.model_validate(kwargs)
        return handler(_payload_for_handler(request))

    invoke.__name__ = name
    invoke.__qualname__ = name
    invoke.__doc__ = description
    signature = _signature_from_request(request_cls)
    invoke.__signature__ = signature
    invoke.__annotations__ = {
        **{param_name: param.annotation for param_name, param in signature.parameters.items()},
        "return": str,
    }
    return invoke


def _signature_from_request(request_cls: type[BaseModel]) -> inspect.Signature:
    """Build a keyword-only signature whose parameters match `request_cls` fields."""
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=_parameter_default(field_info),
            annotation=_parameter_annotation(field_info),
        )
        for name, field_info in request_cls.model_fields.items()
    ]
    return inspect.Signature(parameters, return_annotation=str)


def _parameter_default(field_info: FieldInfo) -> object:
    """Return the inspect default that matches whether `field_info` is required."""
    if field_info.is_required():
        return inspect.Parameter.empty
    if field_info.default is PydanticUndefined:
        return None
    return field_info.default


def _payload_for_handler(model: BaseModel) -> dict[str, object]:
    """Dump `model` with schema aliases so nested overrides round-trip."""
    return model.model_dump(mode="python", by_alias=True, exclude_unset=True)


def _parameter_annotation(field_info: FieldInfo) -> object:
    """Return a signature annotation that keeps the original Pydantic field info."""
    annotation: object = field_info.annotation if field_info.annotation is not None else object
    return Annotated[annotation, field_info]
