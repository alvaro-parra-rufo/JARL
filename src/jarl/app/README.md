# JARL Runner Lab

UI Streamlit empaquetada en `jarl.app` para operar experimentos Navix con la API pública de JARL.

## Requisitos

```bash
uv sync --extra app
```

## Arranque

```bash
uv run streamlit run src/jarl/app/app.py
```

Desde Jupyter: `src/jarl/app/runner_lab_launcher.ipynb` (UI) · `examples/showcase/showcase_launcher.ipynb` (pipeline showcase)

Si la página no carga o la sidebar muestra rutas antiguas, para cualquier instancia previa y vuelve a arrancar (el launcher del notebook lo hace automáticamente). Un servidor viejo en el puerto 8501 suele quedar colgado tras mover `pages/` → `views/`.

## Navegación (tres módulos)

La barra lateral agrupa **Inicio**, **Métricas**, **Entrenamiento**, **Árbol** e **Inferencias**. El entrypoint (`app.py`) usa `st.navigation` sobre scripts en `views/` (no hay carpeta `pages/` multipage). La selección de workspace y experimento sigue en la sidebar.

| Módulo | Sub-pestañas | Función |
|--------|--------------|---------|
| **Inicio** | — | Resumen del experimento activo, nodos, checkpoints, linaje |
| **Métricas** | Explorador · Comparar · Linaje continuo · Inspección | Curvas `metrics.jsonl`, comparación multi-nodo, concatena extends, diff, TB |
| **Entrenamiento** | Nuevo · Lanzar · Continuar · Debug | Crear root, train/resume, wizard extend/fork, depuración |
| **Árbol** | — | Explorador live del DAG (zoom, búsqueda, checkout, export SVG) |
| **Inferencias** | Lanzar | Greedy: checkpoint + mapa Navix + seed + horizon/episodios → MP4 y returns |

### Shell compartido

- `jarl.app.lib.navigation` — árbol `st.navigation` (iconos Material)
- `jarl.app.lib.layout` — `render_context_bar`, `select_node_workspace`, `require_experiment_dir`
- `jarl.app.lib.sidebar` — workspace, experimento activo, chip de run + enlace al monitor en **Lanzar**

### Inferencias (defaults UI)

- **Mapa Navix:** selector del catálogo (`jarl.envs.navix.catalog`); default = mapa del nodo.
- **Horizon:** 100 pasos si el nodo no fija `max_episode_steps`; `0` = sin override en el job.
- **Episodios:** 5 greedy rollouts por defecto (`--final-video-episodes`).

## Workspace

Default: `results/dags/` (override con `JARL_EXPERIMENTS_ROOT` en `.env`). Configurable en la barra lateral de Runner Lab.

## Formulario compartido

`lab_form_*` en sesión: persiste entre páginas (entorno, PPO, vídeo, tracking, avanzado).

En **Entrenar** el entorno es inmutable; solo overrides de run.

Defaults actuales: preset **Personalizado (producción)** con `5_000_000` timesteps,
`64` envs, `128` rollout steps y `2048` minibatch. El preset **Rápido** queda
separado para smoke local (`512` timesteps) y no pisa los valores custom.

Si una sesión vieja conserva valores cortos, pulsa **Defaults producción** o
recarga tras la migración de `FORM_DEFAULTS_REVISION`.

## Métricas largas

La página **Métricas** limita puntos de gráfico y filas raw por defecto para que
`metrics.jsonl` grandes sigan siendo explorables. Ajusta `Puntos gráfico` o
`Filas raw` si necesitas más detalle.

## W&B

Activa en el formulario. Opcional en `.streamlit/secrets.toml`:

```toml
WANDB_API_KEY = "..."
WANDB_MODE = "offline"
```

## Subprocess

Entrenamientos vía `python -m jarl.training.run` (JAX aislado del proceso Streamlit). Inferencia vídeo: `python -m jarl.agents.ppo.inference`.

## Feeds en vivo

`app/lib/live_feed.py` lee deltas de log y `metrics.jsonl` con cursores en sesión.
`app/lib/live_ui.py` expone fragments (`st.fragment(run_every=…)`) para:

- terminal live en **Entrenar** e **Inferencias** (`train.log`, `runner_lab.log`, `inference_runner_lab.log`)
- KPIs live desde `metrics.jsonl` (Métricas / Explorador)
- estado del subprocess en la sidebar **sin** `st.rerun()` al terminar (toast + mensaje en fragment)

| Feed | Intervalo default | Límite / notas |
|------|-------------------|----------------|
| Logs (`LiveTerminalFragment`) | 2 s | Buffer máx. 500 líneas en sesión (`DEFAULT_TERMINAL_MAX_LINES`) |
| Métricas (`LiveMetricsFragment`) | 5 s | Última fila por poll; lectura incremental por cursor |
| Estado run (`RunStatusFragment`) | 3 s | Toast una vez al terminar train o inferencia |
| Árbol (`TreeExplorerFragment`) | 5 s (2.5 s con run activo) | Solo recalcula si cambia la revisión disco/sesión o la vista |

Solo el fragment se re-ejecuta; el resto del formulario no parpadea.

## Árbol live

Módulo dedicado **Árbol** en la navegación lateral (`views/5_Arbol.py`, `tree_page.py`).

- HTML embebido vía `st.components.v1.html` (sin CDN ni `declare_component` frágil).
- Feed en `graph_feed.py`; poll en fragment (5 s / 2.5 s con run activo).
- Solo contenido del árbol: explorador, acciones de checkout/navegación, export SVG, tabla opcional.
- Forks y reconcile viven en **Entrenamiento → Continuar**, no aquí.

## Showcase pipeline

Genera un experimento demo completo (árbol YAML, vídeos por rol, TB, W&B):

```bash
uv run python -m jarl.showcase --dry-run
uv run python -m jarl.showcase --scale smoke --name jarl-showcase-smoke
```

Ver `src/jarl/showcase/README.md` y `examples/showcase/showcase_launcher.ipynb`.

## Tests

```bash
uv run pytest tests/unit/test_app/ -q
```
