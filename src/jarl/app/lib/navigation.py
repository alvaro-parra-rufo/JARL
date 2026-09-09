"""Streamlit navigation shell for Runner Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from jarl.app.lib.layout import APP_PAGES

APP_DIR = Path(__file__).resolve().parents[1]

REGISTERED_NAV_PAGES = frozenset({"inicio", "metricas", "entrenar", "arbol", "inferencias", "testing"})

__all__ = [
    "APP_DIR",
    "REGISTERED_NAV_PAGES",
    "build_navigation",
    "nav_page_path",
    "run_runner_lab",
]


def nav_page_path(key: str) -> Path:
    """Return the path of a page registered in ``st.navigation``."""
    if key not in REGISTERED_NAV_PAGES:
        msg = f"Page {key!r} is not registered in Runner Lab navigation."
        raise ValueError(msg)
    return APP_DIR / APP_PAGES[key]


def build_navigation() -> Any:
    """Build the Runner Lab navigation tree."""
    return st.navigation(
        {
            "": [
                st.Page(
                    APP_DIR / APP_PAGES["inicio"],
                    title="Inicio",
                    icon=":material/home:",
                ),
            ],
            "Métricas": [
                st.Page(
                    APP_DIR / APP_PAGES["metricas"],
                    title="Métricas",
                    icon=":material/show_chart:",
                ),
            ],
            "Entrenamiento": [
                st.Page(
                    APP_DIR / APP_PAGES["entrenar"],
                    title="Entrenamiento",
                    icon=":material/play_arrow:",
                    default=True,
                ),
            ],
            "Árbol": [
                st.Page(
                    APP_DIR / APP_PAGES["arbol"],
                    title="Árbol",
                    icon=":material/account_tree:",
                ),
            ],
            "Inferencias": [
                st.Page(
                    APP_DIR / APP_PAGES["inferencias"],
                    title="Lanzar",
                    icon=":material/movie:",
                ),
            ],
            "Evaluación": [
                st.Page(
                    APP_DIR / APP_PAGES["testing"],
                    title="Testing",
                    icon=":material/science:",
                ),
            ],
        },
        position="sidebar",
        expanded=True,
    )


def run_runner_lab() -> None:
    """Initialize the app shell and run the active navigation page."""
    from jarl.app.lib.ui import init_app_shell

    init_app_shell()
    build_navigation().run()
