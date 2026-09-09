# JARL Showcase Pipeline

CLI reproducible que genera un experimento Navix completo (árbol de 11 nodos en escala `medium`) para explorar en Runner Lab.

## Requisitos

```bash
uv sync
```

## Uso

Desde Jupyter: `examples/showcase/showcase_launcher.ipynb` (configura `SCALE`, `NAME` y flags en la primera celda de código).

```bash
# Plan sin entrenar (validación)
uv run python -m jarl.showcase --dry-run

# Smoke (2–3 nodos, ~5–15 min CPU)
uv run python -m jarl.showcase --scale smoke --name jarl-showcase-smoke

# Showcase por defecto (11 nodos, GPU recomendada)
uv run python -m jarl.showcase --name jarl-showcase-demo

# Producción / TFM (W&B online por defecto)
uv run python -m jarl.showcase --scale large --name jarl-showcase-tfm
```

El CLI carga `.env` del repo vía `jarl.utils.env.load_project_env`.

### Flags útiles

| Flag | Descripción |
|------|-------------|
| `--workspace` | Carpeta padre de experimentos (default: `JARL_EXPERIMENTS_ROOT` o `results/dags`) |
| `--algorithm` | `ppo` o `ppo_gru` (sobreescribe el YAML del escenario) |
| `--scenario` | YAML de escenario (default: `scenarios/default.yaml`) |
| `--reuse` | Saltar nodos `completed` en `showcase_run.json` |
| `--from-node` | Reanudar desde un nodo lógico |
| `--wandb-online` | Forzar W&B online (requiere `WANDB_API_KEY`) |
| `--wandb-project` | Nombre del proyecto W&B (default: `jarl-showcase`) |
| `--force` | Permitir directorio existente sin `--reuse` |

## Escalas

| Escala | Nodos | Timesteps root | W&B |
|--------|-------|----------------|-----|
| `smoke` | 3 | 512 | offline |
| `medium` | 11 | 100k | offline |
| `large` | 11 | 2M | online |

Valores en `scenarios/scales/{smoke,medium,large}.yaml`.

## Artefactos

- `showcase_run.json` — estado incremental por nodo
- `showcase_summary.json` — resumen para **Inicio** en Runner Lab
- Vídeos por rol en `video_metrics.jsonl`: `source_checkpoint`, `child_start`, `final`

## Explorar resultado

```bash
uv run streamlit run src/jarl/app/app.py
```

Selecciona el experimento en la barra lateral. **Inicio** muestra el panel showcase si existe `showcase_summary.json`. **Inspección** agrupa vídeos por rol.

## Tests

```bash
uv run pytest tests/unit/test_showcase/test_showcase_pipeline.py -q
```
