"""CLI entrypoint for `python -m jarl.experiments.cases`."""

from __future__ import annotations

import sys

from jarl.experiments.cases.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
