"""CLI entrypoint for ``python -m jarl.showcase``."""

from __future__ import annotations

import os
import sys

# Training subprocesses set their own JAX platforms; prefer GPU when both are listed.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

from jarl.showcase.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
