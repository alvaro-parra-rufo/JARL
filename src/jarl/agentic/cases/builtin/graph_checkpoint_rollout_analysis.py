"""Agentic case: analyze one greedy checkpoint rollout from a trained root."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._graph_checkpoint_rollout_analysis import (
    RECORD_VIDEO_LAUNCH_OPTION,
    validate_checkpoint_rollout_analysis,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_ppo_rollout_trained_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP,
    ROLLOUT_ENV_ID,
    ROLLOUT_SEED,
    ROLLOUT_TRAINING_TIMESTEPS,
    NavixPpoRolloutTrainedCheckpointCase,
)

__all__ = ["CASE", "GraphCheckpointRolloutAnalysisCase"]


class GraphCheckpointRolloutAnalysisCase(BaseAgenticCase):
    """Validate rollout execution and bounded summary from a trained checkpoint."""

    def __init__(self) -> None:
        """Initialize the checkpoint rollout analysis case."""
        super().__init__(
            CaseSpec(
                id="graph_checkpoint_rollout_analysis",
                title="Analyze rollout and plan larger-map transfer",
                description=(
                    "Execute one reproducible greedy rollout from a trained PPO checkpoint "
                    f"on {ROLLOUT_ENV_ID} ({ROLLOUT_TRAINING_TIMESTEPS} timesteps) and summarize "
                    "what to improve next for a harder, larger map."
                ),
                tags=frozenset({"graph", "operate", "read", "inference", "rollout", "trained"}),
                requirements=frozenset({"llm", "navix"}),
            ),
            NavixPpoRolloutTrainedCheckpointCase(),
            (
                AgenticTurn(
                    f"Necesito un diagnóstico de la política entrenada en {ROLLOUT_ENV_ID} "
                    f"mediante un rollout greedy reproducible con el checkpoint del nodo root "
                    f"en el step {ROLLOUT_CHECKPOINT_STEP} y seed {ROLLOUT_SEED}. "
                    "Devuélveme un resumen estructurado con `checkpoint_eval.episode_return`, "
                    "`checkpoint_eval.episode_length`, el return y la longitud del rollout greedy, "
                    "y la razón de terminación. "
                    "En este mapa trivial no tiene sentido comparar checkpoints entre sí ni "
                    "elegir cuál es 'mejor'. "
                    "En su lugar, interpreta qué limitaciones muestra la política actual y qué "
                    "mejorarías (timesteps, eval frequency, hiperparámetros, arquitectura, seed, "
                    "max_episode_steps, etc.) para adaptar el modelo a un mapa Navix más grande "
                    "o más difícil (p. ej. 16x16, DoorKey, obstáculos). "
                    "Propón pasos concretos siguientes en el grafo JARL (fork/extend con qué "
                    "cambios de config) sin entrenar ni modificar el grafo en esta sesión."
                ),
            ),
            launch_options=(RECORD_VIDEO_LAUNCH_OPTION,),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate rollout read audit, summary payload, and persisted artifacts."""
        validate_checkpoint_rollout_analysis(
            run,
            report,
            checkpoint_step=ROLLOUT_CHECKPOINT_STEP,
            seed=ROLLOUT_SEED,
        )


CASE = GraphCheckpointRolloutAnalysisCase()
"""Built-in checkpoint rollout analysis case."""
