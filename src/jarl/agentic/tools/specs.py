"""Tool metadata and registry for agentic tools."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel

from jarl.utils.pattern_filter import match_patterns

if TYPE_CHECKING:
    from jarl.agentic.tools.context import ToolContext
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "BoundTool",
    "ToolEntry",
    "ToolFilterBy",
    "ToolRegistry",
    "ToolSpec",
    "merge_tool_filters",
]

_REGISTRY_SKIP_MODULES = frozenset(
    {
        "jarl.agentic.tools.context",
        "jarl.agentic.tools.specs",
        "jarl.agentic.tools.handler",
        "jarl.agentic.tools.registry",
    }
)


class ToolFilterBy(TypedDict, total=False):
    """Keyword arguments accepted by ``ToolRegistry.filter_by``."""

    include_names: set[str] | None
    exclude_names: set[str] | None
    include_labels: set[str] | None
    exclude_labels: set[str] | None


_TOOL_FILTER_KEYS: tuple[str, ...] = (
    "include_names",
    "exclude_names",
    "include_labels",
    "exclude_labels",
)
"""Keys merged and forwarded to ``ToolRegistry.filter_by``."""


def merge_tool_filters(*filters: ToolFilterBy | None) -> ToolFilterBy:
    """Union ``ToolFilterBy`` mappings, dropping empty and unset keys.

    ``include_*`` and ``exclude_*`` sets are unioned. Callers typically pass
    phase labels as one mapping and the driver extra as another.
    """
    merged: dict[str, set[str]] = {}
    for filt in filters:
        if not filt:
            continue
        for key in _TOOL_FILTER_KEYS:
            raw = filt.get(key)  # type: ignore[literal-required]
            if not raw:
                continue
            merged.setdefault(key, set()).update(str(item) for item in raw)
    return merged  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Metadata exposed to LangGraph and filtering."""

    name: str
    description: str
    labels: frozenset[str]
    module: str = ""

    def with_module(self, module: str) -> ToolSpec:
        """Return a copy with the discovered module path set."""
        return replace(self, module=module)


@dataclass(frozen=True, slots=True)
class BoundTool:
    """LLM-facing tool payload after the same ``StructuredTool`` bind as the graph."""

    name: str
    description: str
    schema: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """Discovered tool module exports."""

    spec: ToolSpec
    request_cls: type[BaseModel]
    run_fn: Callable[[ToolContext, BaseModel], str]
    build_handler: Callable[[AgenticWorkflow], Callable[..., str]]


class ToolRegistry:
    """Index of agentic tools discovered under ``jarl.agentic.tools``."""

    def __init__(self, entries: Mapping[str, ToolEntry] | None = None) -> None:
        """Initialize a registry from optional pre-built entries."""
        self._entries: dict[str, ToolEntry] = dict(entries or {})

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._entries)

    def __iter__(self) -> Iterator[ToolEntry]:
        """Iterate registered tool entries."""
        return iter(self._entries.values())

    def names(self) -> frozenset[str]:
        """Return registered tool names."""
        return frozenset(self._entries)

    def get(self, name: str) -> ToolEntry:
        """Return the entry for ``name``.

        Raises:
            KeyError: If the tool is not registered.
        """
        return self._entries[name]

    @classmethod
    def from_module(cls, package: str) -> ToolRegistry:
        """Discover tools by walking ``package`` submodules."""
        package_module = importlib.import_module(package)
        package_path = getattr(package_module, "__path__", None)
        if package_path is None:
            msg = f"Package {package!r} has no __path__ for discovery."
            raise ValueError(msg)

        entries: dict[str, ToolEntry] = {}
        prefix = f"{package}."
        for module_info in pkgutil.walk_packages(package_path, prefix=prefix):
            if module_info.name in _REGISTRY_SKIP_MODULES:
                continue
            if module_info.ispkg:
                continue
            module_basename = module_info.name.rsplit(".", 1)[-1]
            if module_basename.startswith("_"):
                continue
            module = importlib.import_module(module_info.name)
            entry = _entry_from_module(module)
            if entry is None:
                if module_info.name.endswith(".__init__"):
                    continue
                msg = f"Tool module {module_info.name!r} is missing TOOL_SPEC."
                raise ValueError(msg)
            if entry.spec.name in entries:
                msg = f"Duplicate tool name {entry.spec.name!r} in {module_info.name}"
                raise ValueError(msg)
            entries[entry.spec.name] = entry
        return cls(entries)

    def filter_by(
        self,
        *,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
        include_labels: set[str] | None = None,
        exclude_labels: set[str] | None = None,
    ) -> ToolRegistry:
        """Return a registry view filtered by tool names and labels."""
        filtered: dict[str, ToolEntry] = {}
        for name, entry in self._entries.items():
            if not match_patterns([name], include=include_names, exclude=exclude_names):
                continue
            if not match_patterns(entry.spec.labels, include=include_labels, exclude=exclude_labels):
                continue
            filtered[name] = entry
        return ToolRegistry(filtered)

    def bound_catalog(
        self,
        *,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
        include_labels: set[str] | None = None,
        exclude_labels: set[str] | None = None,
    ) -> tuple[BoundTool, ...]:
        """Return LLM-facing tool payloads after the same bind as the graph.

        Filter kwargs match ``filter_by``. Construction lives in
        ``jarl.agentic.langgraph.tools`` so this catalog dumps the resulting
        ``StructuredTool``, not a parallel schema.
        """
        from jarl.agentic.langgraph.tools import bound_tools_from_registry

        filtered = self.filter_by(
            include_names=include_names,
            exclude_names=exclude_names,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
        )
        return bound_tools_from_registry(filtered)

    def build_handlers(self, workflow: AgenticWorkflow) -> list[Callable[..., str]]:
        """Build LangGraph-ready handlers for every registered tool."""
        return [entry.build_handler(workflow) for entry in self._entries.values()]


def _entry_from_module(module: object) -> ToolEntry | None:
    """Parse a tool module exports into a registry entry."""
    spec = getattr(module, "TOOL_SPEC", None)
    if spec is None:
        return None
    if not isinstance(spec, ToolSpec):
        msg = f"TOOL_SPEC in {module!r} must be a ToolSpec instance."
        raise TypeError(msg)

    request_cls = getattr(module, "ToolRequest", None)
    build_handler = getattr(module, "build_handler", None)
    run_fn = _find_run_fn(module)
    if request_cls is None or build_handler is None or run_fn is None:
        msg = f"Tool module {getattr(module, '__name__', module)!r} is missing required exports."
        raise ValueError(msg)

    module_name = str(getattr(module, "__name__", ""))
    bound_spec = spec.with_module(module_name)
    return ToolEntry(
        spec=bound_spec,
        request_cls=request_cls,
        run_fn=run_fn,
        build_handler=build_handler,
    )


def _find_run_fn(module: object) -> Callable[[ToolContext, BaseModel], str] | None:
    """Return the first ``run_*`` callable exported by a tool module."""
    for attr_name in dir(module):
        if not attr_name.startswith("run_"):
            continue
        candidate = getattr(module, attr_name, None)
        if callable(candidate):
            return candidate  # type: ignore[return-value]
    return None
