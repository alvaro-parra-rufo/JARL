# AGENTS.md

## Architecture Router

- `src/jarl/`: Core library (production; must not import `src/jarl/app/`)
  - `config.py`: `BaseConfig`, `ConfigDiff`, overrides/diffs — shared by graph, runner, future tools
  - `env_setup.py`: `setup_jax(JaxConfig)`, device logging
  - `io/`: generic I/O helpers
    - `tail.py`: incremental log/`jsonl` tail readers (no session state)
  - `showcase/`: reproducible Navix demo pipeline (`python -m jarl.showcase`); YAML scenarios, executor, CLI
  - `experiments/`: experiment tree (`ExperimentGraph`), `NodeWorkspace`, tracking (TensorBoard, W&B)
    - `cases/`: reusable `ExperimentCase` scenarios, catalog services, materialization CLI (`python -m jarl.experiments.cases`); `metadata.py` (`ExperimentCaseMetadata` → `case.json`)
    - `paths.py`: `resolve_workspace_root`, `JARL_EXPERIMENTS_ROOT`
    - `feed.py`: graph subtree payloads (`build_visible_subtree`, `read_graph_revision`, JSON DTOs)
    - `summaries.py`: `ExperimentSummary`, `load_experiment_summary`, `config_highlights`
    - `metrics/`: `MetricsSnapshot`, `build_metrics_snapshot`, lineage concat helpers
    - `io/layout.py`: `ExperimentLayout`, `NodeLayout`, path constants
    - `io/checkpoints.py`: `CheckpointRef`, registry, aliases (`checkpoint_ref_from_alias`)
    - `io/rollouts.py`: versioned rollout artifacts (`rollout.json`, `trace.npz`, optional MP4)
    - `io/model_archive.py`: `.model` zip export/load (inference/eval)
    - `manifest.py`: `experiment.json` tree manifest
    - `run_config.py`: `RunConfig`, `JaxConfig`, `CheckpointConfig`, `TrackingConfig`
  - `training/`: RL orchestration — `RLRunConfig`, `schedule.py`, `runner.py`, `registry.py`, `env_factory.py`
    - `presets.py`: form values → `RLRunConfig`, validation, production presets
    - `launch.py`: `write_run_config`, `build_*_train_command` argv builders
    - `cli.py` / `run/`: `python -m jarl.training.run` (train, resume, `--config-json`)
    - `trainer.py`: `Trainer` protocol; real trainers live in `agents/`
  - `agents/ppo/`: PPO and PPO-GRU full-JAX trainers, networks, JIT loops, metrics, video
    - `inference/`: greedy checkpoint video CLI (`python -m jarl.agents.ppo.inference`), `render.py`, `requests.py`, `backend.py` (PPO/PPO-GRU policy loaders)
  - `inference/`: reusable policy runtime — `policy.py`, `registry.py`, `run.py` (`run_checkpoint_inference`)
  - `envs/navix/`: full-JIT adapter (`full_jit.py`); configurable rewards (`reward_config.py`, `rewards.py`, `scenario_rewards.py`); JARL maps (`custom/`, p. ej. `empty_variant.py` id `Navix-EmptyVariant-5x5-v0`, `rgb_overlay.py`); rollout telemetry (`telemetry/` capture, postprocess, `NavixRolloutSummary.as_text`)
  - `utils/`: shared helpers (`dicts`, `hash`, `ids`, `io`, `pattern_filter`, `pydantic`); import via `jarl.utils`
  - `operations/`: domain operations shared by app, CLI, and agentic tools (no LangChain)
    - `contracts/`: `dataclass_to_compact_dict`, payload limits, downsampling
    - `graph/`: `create_root`, `checkout`, `fork`, `extend`, `summary`, `reward`, `set_reward`, `checkpoints`, `checkpoint_rollout`, `subtree`, `diff`, `pin`, `promote`, `metrics_series`, `checkpoint_events`
    - `train/`: `run`, `resume`, `recovery_status`
    - `subagent/`: `metrics_analysis` (+ preprocess objetivo de features)
    - `env/`: `navix_maps` (catálogo Navix por categoría)
  - `agentic/`: session, audit, `AgenticWorkflow`; LangGraph/tools (extra `jarl[agentic]`)
    - `tools/`: discovery por módulos hoja (`graph/`, `train/`, `session/`, `env/`, `subagent/`)
      - Lecturas clave: `graph_checkpoints`, `graph_checkpoint_rollout`, `graph_reward`, `graph_set_reward`, `train_recovery_status`, `subagent_metrics_analysis`
      - `session_finish`: declara fin de turno (`finish` + fase); el chat futuro la oculta
    - `debug.py`: `ModelDebugView.from_workflow` / `from_experiment` (timeline de lectura)
    - `transcript/`: `ConversationTranscript.load`, `CheckpointTranscriptReader`, `ConversationRevision`, `ActiveCheckpointPointer`
    - `run_events.py`: `RunEvent` Pydantic, `LlmUsage`, `RunWindow` / `RunWindows.from_events`
    - `langgraph/trace.py`: `LlmGenerationCallback` (metadatos `llm_*` + `usage` desde `AIMessage.usage_metadata`)
    - `langgraph/graphs/subagents/`: subgrafos (p. ej. `metrics_analysis` con LLM)
    - `cases/`: `BaseAgenticCase`, deterministic reports, execution/launch services and CLI (`python -m jarl.agentic.cases`); `list_agentic_case_runs`; builtin p. ej. `navix_empty_variant_spec`
    - `llm/`: `LLMSettings`, `LLMCatalog`, `load_llm_catalog`, `create_chat_model` (`ollama`, `openai_compatible`)
    - `llm/profiles/`: catálogo local (`custom.yaml` > `default.yaml`, gitignored) + plantillas `*.example.yaml` (docs, no se cargan)
  - `mcp/`: puente MCP stdio sobre `REGISTRY` (`python -m jarl.mcp`; extra `jarl[mcp]`). Sin lógica de host; Cursor y Codex solo se distinguen por su archivo de registro
  - Entry: `app.py`, `views/*` (page shells), `runner_lab_launcher.ipynb`
  - **Stays in app (domain facades / UI):** `graph_ops.py`, `continue_flow.py`, `layout.py`, `session.py`, `navigation.py`, `sidebar.py`, `project_env.py`, `tree_explorer` + `components/tree_explorer/`
  - **Partial splits (core + UI residue):** `graph_feed.py` (session revision + tree defaults), `metrics_view.py` (pandas/plotly presets), `summaries.py` (`node_table` only), `live_feed.py` (Streamlit tail cursors), `inference_view.py` (pandas timing), `runner_bridge.py` (`Popen` subprocess bridge), `inspect_view.py` / `checkpoints_view.py` (pandas tables)
  - **Orquestación páginas:** `training_page`, `metrics_page`, `tree_page`, `inference_page`, `testing_page`, `forms`, `form_state`, `live_ui`, `inspection_panel`
  - **Conversación (UI compartida):** `lib/conversation/` (`live.py`, `messages.py`, `format.py` chips); Testing y futuro chatbot
  - **Debug del modelo (UI):** `lib/debug/` (`flag.py`, `panel.py`); toggle en `sidebar.py` (no en `session.py`); filas/chips vía `conversation/`
- `tests/`: pytest suite mirroring `src/jarl`; reusable cases run with fake LLM under `functional/` and real providers under opt-in `llm_eval/`
- `docs/`: mkdocs; `docs/development/ppo-collected-data.md` = PPO artefact/metric inventory; `docs/development/agentic-workflow.md` = workflow + debug del modelo; `docs/development/agentic-cases.md` = case authoring/execution (protocolo batch)
- `examples/notebooks/`: experiment tree demos (`experiments_mnist_*`)
- `examples/showcase/`: `showcase_launcher.ipynb` for the showcase pipeline
- `.agents/`: skills, plans (`.agents/local/plans/`), feedback, reports

### App vs core (post-migración)

| Necesidad sin Streamlit | Módulo core |
|-------------------------|-------------|
| Entrenar / reanudar | `jarl.training.run`, `jarl.training.presets`, `jarl.training.launch` |
| Inferencia vídeo | `jarl.agents.ppo.inference` |
| Rollout analizado desde checkpoint | `jarl.operations.graph.checkpoint_rollout`, `jarl.inference.run` |
| Telemetría / resumen Navix | `jarl.envs.navix.telemetry` (`as_text`, `show`) |
| Grafo / nodos / checkpoints | `jarl.experiments.graph`, `NodeWorkspace`, `io/checkpoints` |
| Resumen / feed JSON árbol | `jarl.experiments.summaries`, `jarl.experiments.feed` |
| Series métricas | `jarl.experiments.metrics` |
| Tail logs/jsonl | `jarl.io.tail` |
| Workspace root | `jarl.experiments.paths` |
| Pipeline demo | `jarl.showcase` |
| Materializar escenarios | `jarl.experiments.cases` |
| Evaluar tools con casos | `jarl.agentic.cases` |
| Listar runs de casos | `jarl.agentic.cases` (`list_agentic_case_runs`, `load_favorite_run_names`) |
| Cerrar caso batch | `session_finish` + `BaseAgenticCase.run` (`max_continuations`) |
| Timeline debug del modelo | `jarl.agentic.debug` (`ModelDebugView`) |
| Conversación (checkpointer) | `jarl.agentic.transcript` (`ConversationTranscript.load`, `revision`) |
| Eventos de run / ventanas | `jarl.agentic.run_events`, `audit.load_run_events` |
| Metadatos LLM en el run | `jarl.agentic.langgraph.trace` |
| Lanzar casos en subprocess | `jarl.agentic.cases.launch` |
| Elegir checkpoint (aliases/search) | `jarl.operations.graph.checkpoints` |
| Ejecutar rollout greedy + artefactos | `jarl.operations.graph.checkpoint_rollout` |
| Diagnóstico recovery train | `jarl.operations.train.recovery_status` |
| Features métricas (preprocess) | `jarl.operations.subagent.metrics_analysis` |
| Tools JARL vía MCP stdio | `jarl.mcp` (`python -m jarl.mcp`) |

## Validation

Run with `uv run` (Python 3.12):

- `uv run ruff check src/ tests/` — lint
- `uv run ruff format --check src/ tests/` — format check
- `uv run pytest tests/` — test suite
- `uv run zensical build --strict` — docs build (not setup yet)

Run the relevant validation commands after changes (e.g., `ruff` after code edits, `zensical build` after docs, `pytest` after tests).

## Available Skills

Skills live in `.agents/skills/`, prefixed `project-`.

- `project-jarl-mcp`: Operate JARL MCP tools with LangGraph usage rules (user directs; no extra work)
