"""Sidebar flag for the model-debug timeline."""

from __future__ import annotations

MODEL_DEBUG_FLAG_KEY = "jarl_model_debug_enabled"
"""Streamlit session key for the model-debug toggle. Not stored in ``session.py``."""

__all__ = ["MODEL_DEBUG_FLAG_KEY", "is_debug_enabled"]


def is_debug_enabled() -> bool:
    """Return whether the model-debug timeline is on.

    Missing key is off. Lazy-imports Streamlit like ``jarl.app.lib.session``.
    """
    import streamlit as st

    return bool(st.session_state.get(MODEL_DEBUG_FLAG_KEY, False))
