"""Shared wiring for LangGraph tool handlers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from jarl.agentic.audit import AuditIndexEvent, AuditKind
from jarl.agentic.errors import ToolError
from jarl.agentic.progress import append_progress_event
from jarl.agentic.tools._validation_messages import format_request_validation_error
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.specs import ToolSpec
from jarl.metadata import now_iso

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["wire_handler"]

_AUDIT_KIND_PRIORITY: tuple[str, ...] = (
    "train",
    "mutation",
    "subagent",
    "initialization",
    "read",
    "session",
)


def wire_handler(
    workflow: AgenticWorkflow,
    run_fn: Callable[[ToolContext, BaseModel], str],
    tool_spec: ToolSpec,
    *,
    request_cls: type[BaseModel],
) -> Callable[..., str]:
    """Adapt a ``run_*`` function to a LangGraph or MCP tool handler.

    Reloads the experiment graph from disk at the start of every call so
    out-of-process training status is visible to later reads.
    """

    def handler(payload: dict[str, Any] | None = None, **kwargs: Any) -> str:
        workflow.reload_graph()
        ctx = ToolContext.from_workflow(workflow)
        request_data = dict(payload or kwargs)
        forced = ctx.workflow.tool_overrides_for(tool_spec.name)
        if forced:
            request_data = {**request_data, **forced}
        node_before = ctx.try_current_node_id()
        try:
            request = request_cls.model_validate(request_data)
            result = run_fn(ctx, request)
        except Exception as exc:
            tool_error = ToolError.from_exception(tool_spec.name, exc)
            message = (
                format_request_validation_error(request_cls, exc)
                if isinstance(exc, ValidationError)
                else tool_error.message
            )
            error_payload = {"error": message, "code": tool_error.code}
            append_progress_event(
                ctx.workflow.exp_dir,
                "tool_error",
                f"Tool {tool_spec.name} falló: {message}",
                data={"tool": tool_spec.name, "code": tool_error.code},
            )
            _record_tool_audit(
                ctx,
                tool_spec=tool_spec,
                kind="error",
                request_data=request_data,
                result=error_payload,
                node_before=node_before,
            )
            return json.dumps(error_payload, ensure_ascii=False)

        if _should_audit(ctx, tool_spec):
            audit_kind = _audit_kind_from_labels(tool_spec.labels)
            _record_tool_audit(
                ctx,
                tool_spec=tool_spec,
                kind=audit_kind,
                request_data=request_data,
                result=_parse_result_summary(result),
                node_before=node_before,
                node_after=ctx.try_current_node_id(),
            )
            append_progress_event(
                ctx.workflow.exp_dir,
                "tool_call",
                f"Tool {tool_spec.name} ({audit_kind})",
                data={"tool": tool_spec.name, "kind": audit_kind},
            )
        return result

    return handler


def _audit_kind_from_labels(labels: frozenset[str]) -> AuditKind:
    # Train-category reads (e.g. train_recovery_status) must audit as read.
    effective = labels - {"train"} if ("train" in labels and "read" in labels) else labels
    for label in _AUDIT_KIND_PRIORITY:
        if label in effective:
            if label == "initialization":
                return "mutation"
            if label == "session":
                return "read"
            return label  # type: ignore[return-value]
    return "read"


def _should_audit(ctx: ToolContext, tool_spec: ToolSpec) -> bool:
    kind = _audit_kind_from_labels(tool_spec.labels)
    return not (kind == "read" and not ctx.workflow.session.audit_reads)


def _parse_result_summary(result: str) -> dict[str, object]:
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return {"raw": result}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _record_tool_audit(
    ctx: ToolContext,
    *,
    tool_spec: ToolSpec,
    kind: AuditKind,
    request_data: dict[str, Any],
    result: dict[str, object] | None,
    node_before: str | None = None,
    node_after: str | None = None,
) -> None:
    link = ctx.workflow.take_last_subgraph_link()
    attach_link = kind == "subagent" or (kind == "error" and link is not None)
    ctx.record_audit(
        AuditIndexEvent(
            ts=now_iso(),
            thread_id=ctx.thread_id,
            tool=tool_spec.name,
            kind=kind,
            request=request_data,
            result=result,
            node_before=node_before,
            node_after=node_after,
            sub_thread_id=link.sub_thread_id if attach_link and link is not None else None,
            parent_tool=((link.parent_tool or tool_spec.name) if attach_link and link is not None else None),
            run_dir=link.run_dir if attach_link and link is not None else None,
        )
    )
