"""Streamlit custom component for the Runner Lab experiment tree explorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_NAME = "jarl_tree_explorer"
_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

_tree_explorer_component = components.declare_component(_COMPONENT_NAME, path=str(_FRONTEND_DIR))

__all__ = ["tree_explorer"]


def tree_explorer(
    *,
    payload: dict[str, Any],
    key: str | None = None,
    default: dict[str, Any] | None = None,
    height: int = 640,
) -> dict[str, Any] | None:
    """Render the interactive experiment tree explorer.

    Args:
        payload: JSON-serializable subtree snapshot from ``graph_feed``.
        key: Optional Streamlit widget key.
        default: Default component return value.
        height: Iframe height in pixels.

    Returns:
        Component event payload when the operator interacts with the tree.
    """
    return _tree_explorer_component(
        payload=payload,
        height=height,
        key=key,
        default=default,
    )
