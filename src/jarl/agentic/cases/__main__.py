"""CLI entrypoint for `python -m jarl.agentic.cases`."""

from __future__ import annotations

import sys

from jarl.agentic.cases.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
