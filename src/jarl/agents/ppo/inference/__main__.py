"""CLI entrypoint for ``python -m jarl.agents.ppo.inference``."""

import sys

from jarl.agents.ppo.inference.cli import main

if __name__ == "__main__":
    sys.exit(main())
