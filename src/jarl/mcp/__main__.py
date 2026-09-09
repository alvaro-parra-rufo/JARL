"""CLI entrypoint for ``python -m jarl.mcp``."""

from __future__ import annotations

import sys

from jarl.mcp.server import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
