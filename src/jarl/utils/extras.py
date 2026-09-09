"""Optional dependency availability checks."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

OptionalExtra = Literal["agentic", "app"]

_EXTRA_MODULES: dict[OptionalExtra, tuple[str, ...]] = {
    "agentic": ("langchain_core", "langgraph"),
    "app": ("streamlit",),
}

__all__ = ["OptionalExtra", "is_extra_available"]


def is_extra_available(extra: OptionalExtra) -> bool:
    """Return whether the import markers for an optional JARL extra exist."""
    return all(find_spec(module) is not None for module in _EXTRA_MODULES[extra])
