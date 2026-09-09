"""Streamlit helpers for live conversation polling from disk revision."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from jarl.agentic.errors import SessionError
from jarl.agentic.transcript import CheckpointTranscriptReader, ConversationTranscript

CONVERSATION_CACHE_DIR_KEY = "jarl_conversation_cached_dir"
CONVERSATION_CACHE_TRANSCRIPT_KEY = "jarl_conversation_cached_transcript"
CONVERSATION_CACHE_REVISION_KEY = "jarl_conversation_cached_revision"

__all__ = [
    "CONVERSATION_CACHE_DIR_KEY",
    "CONVERSATION_CACHE_REVISION_KEY",
    "CONVERSATION_CACHE_TRANSCRIPT_KEY",
    "poll_conversation_transcript",
    "reset_conversation_live_cache",
    "track_conversation_experiment_dir",
]


def reset_conversation_live_cache() -> None:
    """Drop cached transcript and revision so the next poll reloads from disk."""
    st.session_state.pop(CONVERSATION_CACHE_TRANSCRIPT_KEY, None)
    st.session_state.pop(CONVERSATION_CACHE_DIR_KEY, None)
    st.session_state.pop(CONVERSATION_CACHE_REVISION_KEY, None)


def track_conversation_experiment_dir(exp_dir: Path) -> None:
    """Reset conversation cache when the active experiment directory changes."""
    resolved = str(exp_dir.resolve())
    cached = st.session_state.get(CONVERSATION_CACHE_DIR_KEY)
    if cached != resolved:
        reset_conversation_live_cache()
        st.session_state[CONVERSATION_CACHE_DIR_KEY] = resolved


def poll_conversation_transcript(
    exp_dir: Path,
    *,
    force: bool = False,
) -> ConversationTranscript | None:
    """Return the cached transcript, reloading when disk revision changes.

    Uses ``CheckpointTranscriptReader.revision`` (sqlite + pointer) as the
    liveness clock. Replaces the cache entirely; never appends rows.
    """
    reader = CheckpointTranscriptReader.for_experiment(exp_dir)
    if reader is None:
        reset_conversation_live_cache()
        return None

    revision = reader.revision()
    cached_revision = st.session_state.get(CONVERSATION_CACHE_REVISION_KEY)
    cached_transcript = st.session_state.get(CONVERSATION_CACHE_TRANSCRIPT_KEY)
    if (
        not force
        and cached_revision == revision.token
        and cached_transcript is not None
        and isinstance(cached_transcript, ConversationTranscript)
    ):
        return cached_transcript

    try:
        transcript = reader.load()
    except (SessionError, OSError, ValueError, TypeError):
        raise
    st.session_state[CONVERSATION_CACHE_TRANSCRIPT_KEY] = transcript
    st.session_state[CONVERSATION_CACHE_REVISION_KEY] = revision.token
    st.session_state[CONVERSATION_CACHE_DIR_KEY] = str(exp_dir.resolve())
    return transcript
