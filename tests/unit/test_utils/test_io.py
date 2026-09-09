"""Tests for atomic text writes."""

from __future__ import annotations

from pathlib import Path

from jarl.utils.io import write_text_atomic


class TestWriteTextAtomic:
    """Tests for atomic file persistence."""

    def test_writes_content(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "file.txt"

        result = write_text_atomic(target, "hello")

        assert result == target
        assert target.read_text(encoding="utf-8") == "hello"
