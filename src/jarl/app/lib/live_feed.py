"""Incremental file tail readers for Runner Lab live UI fragments."""

from __future__ import annotations

from pathlib import Path

from jarl.io.tail import merge_lines_into_buffer, read_log_bytes_delta

DEFAULT_TERMINAL_MAX_LINES = 500

__all__ = [
    "DEFAULT_TERMINAL_MAX_LINES",
    "cursor_key_for_path",
    "read_log_delta",
    "reset_log_feed",
    "reset_tail_cursor",
    "reset_terminal_buffer",
    "tail_cursor_offset",
    "update_tail_cursor",
    "update_terminal_buffer",
]


def cursor_key_for_path(path: Path, *, prefix: str = "jarl_tail") -> str:
    """Build a stable session key for a tail cursor on ``path``."""
    return f"{prefix}_{path.resolve()}"


def tail_cursor_offset(session_key: str, *, default: int = 0) -> int:
    """Return the stored byte offset for a tail cursor."""
    import streamlit as st

    return int(st.session_state.get(session_key, default))


def update_tail_cursor(session_key: str, byte_offset: int) -> None:
    """Persist a tail cursor byte offset."""
    import streamlit as st

    st.session_state[session_key] = int(byte_offset)


def reset_tail_cursor(session_key: str) -> None:
    """Reset a tail cursor to the start of the file."""
    update_tail_cursor(session_key, 0)


def read_log_delta(path: Path, *, cursor_key: str) -> list[str]:
    """Append-only log delta using a Streamlit session cursor."""
    offset = tail_cursor_offset(cursor_key)
    lines, next_offset = read_log_bytes_delta(path, byte_offset=offset)
    update_tail_cursor(cursor_key, next_offset)
    return lines


def update_terminal_buffer(
    buffer_key: str,
    new_lines: list[str],
    *,
    max_lines: int = DEFAULT_TERMINAL_MAX_LINES,
) -> str:
    """Append lines to a session terminal buffer and return rendered text."""
    import streamlit as st

    existing = list(st.session_state.get(buffer_key, []))
    buffer = merge_lines_into_buffer(existing, new_lines, max_lines=max_lines)
    st.session_state[buffer_key] = buffer
    return "\n".join(buffer)


def reset_log_feed(path: Path, buffer_key: str) -> None:
    """Reset the tail cursor and terminal buffer for one log file."""
    reset_tail_cursor(f"{buffer_key}__{cursor_key_for_path(path)}")
    reset_terminal_buffer(buffer_key)


def reset_terminal_buffer(buffer_key: str) -> None:
    """Clear a session terminal buffer."""
    import streamlit as st

    st.session_state[buffer_key] = []
