"""Shared helpers for ``env_navix_maps`` agentic cases."""

from __future__ import annotations

from collections.abc import Mapping

from jarl.agentic.audit import AuditIndexEvent
from jarl.agentic.cases.base_agentic_case import AgenticCaseRun
from jarl.agentic.cases.validation import ValidationReport

__all__ = [
    "as_int",
    "assert_read_only_catalog_search",
    "difficulty_bounds_used",
    "message_contents",
    "navix_maps_events",
    "query_mentions",
    "request_categories",
    "request_truthy",
]

_EXPECTED_NODE_COUNT = 3
"""Node count materialized by `DemoTreeCase`."""


def navix_maps_events(run: AgenticCaseRun) -> list[AuditIndexEvent]:
    """Return audited ``env_navix_maps`` events."""
    return [event for event in run.audit_events if event.tool == "env_navix_maps"]


def message_contents(run: AgenticCaseRun) -> str:
    """Join textual AI/tool contents from the final LangGraph state."""
    parts: list[str] = []
    for message in run.final_state.messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
    return " ".join(parts)


def request_categories(run: AgenticCaseRun) -> frozenset[str]:
    """Return the union of category keys requested across ``env_navix_maps`` calls."""
    requested: set[str] = set()
    for event in navix_maps_events(run):
        categories = event.request.get("categories") or []
        if isinstance(categories, list):
            requested.update(str(item) for item in categories)
    return frozenset(requested)


def request_truthy(run: AgenticCaseRun, field: str) -> bool:
    """Return whether any ``env_navix_maps`` request sets ``field`` to a truthy value."""
    return any(bool(event.request.get(field)) for event in navix_maps_events(run))


def assert_read_only_catalog_search(
    run: AgenticCaseRun,
    report: ValidationReport,
) -> list[AuditIndexEvent]:
    """Validate read-only catalog usage and return ``env_navix_maps`` events."""
    map_events = navix_maps_events(run)
    report.check(map_events, "Expected at least one env_navix_maps search.")
    report.check(
        any(event.kind == "read" for event in run.audit_events),
        "Expected at least one audited read tool call.",
    )
    report.check(
        not any(event.kind in {"mutation", "train"} for event in run.audit_events),
        "Expected no mutation or training tool calls.",
        halted=True,
    )
    report.check(
        run.graph.as_networkx().number_of_nodes() == _EXPECTED_NODE_COUNT,
        "Expected the three-node demo tree to remain unchanged.",
    )
    return map_events


def as_int(value: object) -> int | None:
    """Parse an optional integer from a tool request field."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def query_mentions(run: AgenticCaseRun, token: str) -> bool:
    """Return whether any catalog query contains ``token`` (case-insensitive)."""
    needle = token.casefold()
    for event in navix_maps_events(run):
        query = event.request.get("query")
        if isinstance(query, str) and needle in query.casefold():
            return True
    return False


def difficulty_bounds_used(run: AgenticCaseRun) -> bool:
    """Return whether any request sets difficulty_min and/or difficulty_max."""
    for event in navix_maps_events(run):
        request: Mapping[str, object] = event.request
        if as_int(request.get("difficulty_min")) is not None:
            return True
        if as_int(request.get("difficulty_max")) is not None:
            return True
    return False
