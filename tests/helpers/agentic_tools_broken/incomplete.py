"""Incomplete tool module used to test registry discovery failures."""

from __future__ import annotations

from pydantic import BaseModel


class ToolRequest(BaseModel):
    """Payload without a matching ``TOOL_SPEC`` export."""

    value: str = "broken"
