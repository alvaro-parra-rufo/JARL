# Showcase launcher

Notebook para lanzar el pipeline [`jarl.showcase`](../../src/jarl/showcase/README.md) desde Jupyter.

```bash
uv sync
jupyter lab examples/showcase/showcase_launcher.ipynb
```

CLI directo:

```bash
uv run python -m jarl.showcase --dry-run
```
