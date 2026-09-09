#!/usr/bin/env python3
r"""Goal-driven RL demo with a real LLM and the JARL agentic workflow.

This script is **not** a test. It wires a natural-language RL objective through
``AgenticWorkflow.invoke`` → LangGraph → registered tools → ``jarl.operations``
(graph mutations, ``train_run``, inspection subagents, etc.).

**Warning:** Running this demo may start **real PPO training** and mutate nodes
under ``examples/agentic/_output/``. Configure ``JARL_LLM_*`` (and provider
credentials) before executing.

Example::

    export JARL_LLM_PROVIDER=ollama
    uv run python examples/agentic/goal_driven_training.py doorkey-demo --max-turns 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from jarl.agentic.llm import create_chat_model, load_llm_settings_from_env
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.manifest import MANIFEST_FILENAME, ExperimentManifest
from jarl.experiments.summaries import config_highlights
from jarl.training.config import RLRunConfig
from jarl.utils.io import write_text_atomic

DEMO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DEMO_ROOT / "_output"
DEFAULT_EXPERIMENT_NAME = "goal-driven-demo"

DEFAULT_ENV_ID = "Navix-DoorKey-5x5-v0"
DEFAULT_GOAL_TEMPLATE = (
    "Quiero que entrenes una política PPO para completar el mapa {env_id} con el id igual a como se llama"
    "con el menor número de pasos posible. Crea un baseline razonable, entrena, "
    "inspecciona el resultado y decide si conviene crear una variante, solo puedes usar ppo"
    "En cada paso razona brevemente qué acción tomas y por qué."
)
CONTINUATION_GOAL = (
    "Continúa con el objetivo anterior: si aún falta baseline o entrenamiento, "
    "hazlo; si ya hay resultados, inspecciónalos y crea o entrena una variante "
    "solo si aporta valor. Razona brevemente cada acción y compreuba el estado del experimento despues de cada acción."
)
_TOOL_PREVIEW_MAX = 10000


def _find_repo_root(start: Path | None = None) -> Path | None:
    for path in ((start or Path.cwd()), *(start or Path.cwd()).parents):
        if (path / "pyproject.toml").is_file():
            return path
    return None


def _load_repo_dotenv() -> None:
    root = _find_repo_root()
    if root is None:
        return
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def _ensure_experiment_dir(exp_dir: Path) -> Path:
    """Return an experiment directory, creating an empty manifest when missing."""
    resolved = exp_dir.resolve()
    manifest_path = resolved / MANIFEST_FILENAME
    if manifest_path.is_file():
        return resolved

    resolved.mkdir(parents=True, exist_ok=True)
    config = RLRunConfig()
    graph = ExperimentGraph(resolved, base_config=config)
    config.save(graph.layout.config_path)
    write_text_atomic(
        manifest_path,
        ExperimentManifest().model_dump_json(indent=2),
    )
    return resolved


def _format_message(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        content = message.content if isinstance(message.content, str) else json.dumps(message.content)
        return f"[usuario] {content}"
    if isinstance(message, AIMessage):
        if message.tool_calls:
            names = [str(call.get("name", "?")) for call in message.tool_calls]
            return f"[asistente → tools] {names}"
        content = message.content if isinstance(message.content, str) else json.dumps(message.content)
        return f"[asistente] {content}"
    if isinstance(message, ToolMessage):
        preview = message.content if isinstance(message.content, str) else json.dumps(message.content)
        if len(preview) > _TOOL_PREVIEW_MAX:
            preview = preview[:_TOOL_PREVIEW_MAX] + "…"
        name = message.name or "tool"
        return f"[tool:{name}] {preview}"
    return f"[{message.__class__.__name__}] {message!r}"


def _print_message_trace(messages: list[BaseMessage], *, tail: int = 24) -> None:
    print("--- Traza (últimos mensajes) ---")
    for line in (_format_message(message) for message in messages[-tail:]):
        print(line)


def _print_experiment_summary(workflow: AgenticWorkflow) -> None:
    graph = workflow.reload_graph()
    nx_graph = graph.as_networkx()
    print("--- Resumen del experimento ---")
    print(f"  directorio: {workflow.exp_dir}")
    print(f"  thread_id:  {workflow.thread_id}")
    print(f"  nodos:      {nx_graph.number_of_nodes()}")
    current_id = workflow.try_current_node_id()
    print(f"  nodo actual:{current_id or '(ninguno)'}")
    for branch, head in sorted(graph.get_branches().items()):
        highlights = config_highlights(graph.resolve_config(head))
        latest = head.latest_metrics().get("eval/episode_return")
        eval_text = f"{latest:.3f}" if isinstance(latest, (int, float)) else "n/a"
        print(f"  rama {branch:12} id={head.id} status={head.status.value} env={highlights['env']} eval={eval_text}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Demo agentic real: objetivo RL en lenguaje natural → LLM → LangGraph → tools JARL."),
    )
    parser.add_argument(
        "experiment_name",
        nargs="?",
        default=DEFAULT_EXPERIMENT_NAME,
        help=(
            f"Nombre del subdirectorio bajo {DEFAULT_OUTPUT_ROOT.relative_to(DEMO_ROOT)!s}/ "
            f"(default: {DEFAULT_EXPERIMENT_NAME})."
        ),
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Objetivo en lenguaje natural (por defecto: plantilla orientada a PPO + mapa).",
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help=f"Navix env id insertado en la plantilla de objetivo (default: {DEFAULT_ENV_ID}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1,
        metavar="N",
        help="Número de invocaciones consecutivas de workflow.invoke (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperatura del LLM (si se omite, usa JARL_LLM_* / default del provider).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the agentic workflow, and print traces."""
    args = _build_parser().parse_args(argv)
    if args.max_turns < 1:
        print("error: --max-turns debe ser >= 1", file=sys.stderr)
        return 2

    _load_repo_dotenv()
    try:
        llm_settings = load_llm_settings_from_env()
    except ValueError as exc:
        print(f"error: configuración LLM inválida: {exc}", file=sys.stderr)
        return 1

    exp_dir = _ensure_experiment_dir(DEFAULT_OUTPUT_ROOT / args.experiment_name)
    workflow = AgenticWorkflow.from_experiment(exp_dir, audit_reads=True)

    llm = create_chat_model(temperature=args.temperature) if args.temperature is not None else create_chat_model()

    primary_goal = args.goal or DEFAULT_GOAL_TEMPLATE.format(env_id=args.env_id)

    print("=== JARL goal-driven agentic demo ===")
    print(f"Experimento: {exp_dir}")
    print(f"LLM: provider={llm_settings.provider} model={llm_settings.model}")
    print(f"Turnos: {args.max_turns}")
    print(f"Objetivo:\n{primary_goal}\n")

    last_messages: list[BaseMessage] = []
    for turn in range(1, args.max_turns + 1):
        user_text = primary_goal if turn == 1 else CONTINUATION_GOAL
        print(f"=== Turno {turn}/{args.max_turns} ===")
        print(f"[usuario] {user_text}\n")

        result = workflow.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            llm=llm,
        )
        messages = list(result.messages)
        if messages:
            last_messages = [message for message in messages if isinstance(message, BaseMessage)]
            _print_message_trace(last_messages)
        print()

    if last_messages:
        final_text = last_messages[-1].content
        if isinstance(final_text, str) and final_text.strip():
            print("=== Respuesta final del agente ===")
            print(final_text)
            print()

    _print_experiment_summary(workflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
