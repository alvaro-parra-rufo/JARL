"""CLI entrypoint for ``python -m jarl.training.run``."""

import sys

from jarl.training.cli import main

if __name__ == "__main__":
    sys.exit(main())
