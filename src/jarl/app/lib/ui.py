"""Visual helpers for the Streamlit Runner Lab."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

STATUS_TONES: Mapping[str, tuple[str, str]] = {
    "created": ("#51606f", "#eef2f7"),
    "prepared": ("#2563eb", "#dbeafe"),
    "training": ("#0f766e", "#ccfbf1"),
    "paused": ("#7c3aed", "#ede9fe"),
    "interrupted": ("#b45309", "#fef3c7"),
    "completed": ("#15803d", "#dcfce7"),
    "failed": ("#b91c1c", "#fee2e2"),
}


def init_app_shell(*, page_title: str = "JARL Runner Lab") -> None:
    """Configure Streamlit once from the app entrypoint."""
    if not st.session_state.get("_jarl_app_shell_ready"):
        st.set_page_config(page_title=page_title, page_icon="J", layout="wide")
        st.session_state["_jarl_app_shell_ready"] = True
    apply_theme()


def configure_page(title: str) -> None:
    """Apply Runner Lab theme on navigation subpages."""
    _ = title
    apply_theme()


def apply_theme() -> None:
    """Inject lightweight CSS for a focused local operator UI."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block"
        />
        <style>
        :root {
          --jarl-bg: #f7f8fa;
          --jarl-panel: #ffffff;
          --jarl-border: #d9dee7;
          --jarl-text: #18212f;
          --jarl-muted: #647084;
          --jarl-blue: #2563eb;
          --jarl-teal: #0f766e;
          --jarl-amber: #b45309;
          --jarl-red: #b91c1c;
        }
        .stApp {
          background:
            linear-gradient(180deg, #f9fafb 0%, var(--jarl-bg) 36%, #f3f5f8 100%);
          color: var(--jarl-text);
        }
        .block-container {
          padding-top: 1.65rem;
          padding-bottom: 3rem;
          max-width: 1380px;
        }
        section[data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--jarl-border);
        }
        h1, h2, h3 {
          letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
          background: var(--jarl-panel);
          border: 1px solid var(--jarl-border);
          border-radius: 8px;
          padding: 0.9rem 1rem;
          min-height: 95px;
        }
        div[data-testid="stMetric"] label {
          color: var(--jarl-muted);
        }
        .jarl-page-kicker {
          color: var(--jarl-muted);
          font-size: 0.82rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
          margin: 0 0 .2rem;
        }
        .jarl-page-title {
          margin: 0;
          line-height: 1.05;
        }
        .jarl-page-subtitle {
          color: var(--jarl-muted);
          font-size: 1rem;
          max-width: 860px;
          margin-top: .45rem;
        }
        .jarl-status {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          font-size: .78rem;
          font-weight: 700;
          line-height: 1;
          padding: .34rem .52rem;
          white-space: nowrap;
        }
        .jarl-path {
          color: #334155;
          background: #f1f5f9;
          border: 1px solid #d9e2ee;
          border-radius: 6px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: .84rem;
          overflow-wrap: anywhere;
          padding: .55rem .65rem;
        }
        .jarl-kv {
          display: grid;
          grid-template-columns: minmax(120px, 180px) 1fr;
          gap: .5rem .8rem;
          align-items: start;
        }
        .jarl-kv dt {
          color: var(--jarl-muted);
          font-weight: 700;
        }
        .jarl-kv dd {
          margin: 0;
          overflow-wrap: anywhere;
        }
        .jarl-empty {
          border: 1px dashed #b8c3d4;
          border-radius: 8px;
          padding: 1.1rem 1.2rem;
          background: rgba(255,255,255,.68);
        }
        .jarl-empty strong {
          color: var(--jarl-text);
        }
        .jarl-muted {
          color: var(--jarl-muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, *, kicker: str = "JARL Runner Lab") -> None:
    """Render a compact page header."""
    st.markdown(
        f"""
        <div>
          <p class="jarl-page-kicker">{escape(kicker)}</p>
          <h1 class="jarl-page-title">{escape(title)}</h1>
          <p class="jarl-page-subtitle">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Return an HTML badge for a node lifecycle status."""
    fg, bg = STATUS_TONES.get(status, ("#334155", "#e2e8f0"))
    label = status.replace("_", " ").title()
    return f'<span class="jarl-status" style="color:{fg};background:{bg};">{escape(label)}</span>'


def render_status_badge(status: str) -> None:
    """Render a lifecycle status badge."""
    st.markdown(status_badge(status), unsafe_allow_html=True)


def render_parent_checkpoint_banner(workspace: object) -> None:
    """Show whether training will restore parent checkpoint weights."""
    parent_step = getattr(workspace, "parent_checkpoint_step", None)
    parent_id = getattr(getattr(workspace, "node_metadata", None), "parent_id", None)
    if parent_step is None:
        return
    parent_label = parent_id or "padre"
    st.info(f"Este nodo cargará el checkpoint **{parent_step}** del nodo **{parent_label}** al entrenar.")


def path_block(path: str | Path) -> None:
    """Render a filesystem path as a compact block."""
    st.markdown(f'<div class="jarl-path">{escape(str(path))}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    """Render a neutral empty-state panel."""
    st.markdown(
        f"""
        <div class="jarl-empty">
          <strong>{escape(title)}</strong><br>
          <span class="jarl-muted">{escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def key_value(items: Mapping[str, Any]) -> None:
    """Render key-value metadata without a heavy table."""
    rows = "\n".join(
        f"<dt>{escape(str(key))}</dt><dd>{escape(_format_value(value))}</dd>" for key, value in items.items()
    )
    st.markdown(f'<dl class="jarl-kv">{rows}</dl>', unsafe_allow_html=True)


def format_compact_number(value: int | float | None) -> str:
    """Format large counters for metric cards."""
    if value is None:
        return "-"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.3g}"
    return f"{int(value):,}"


def format_float(value: float | None) -> str:
    """Format scalar metric values for operator display."""
    if value is None:
        return "-"
    return f"{value:.5g}"


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return format_float(value)
    return str(value)
