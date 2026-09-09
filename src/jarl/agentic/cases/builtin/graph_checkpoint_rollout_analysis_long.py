"""Agentic case: analyze one greedy rollout from a longer trained checkpoint."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._graph_checkpoint_rollout_analysis import (
    RECORD_VIDEO_LAUNCH_OPTION,
    validate_checkpoint_rollout_analysis,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_ppo_rollout_long_trained_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP,
    ROLLOUT_ENV_ID,
    ROLLOUT_SEED,
    ROLLOUT_TRAINING_TIMESTEPS,
    NavixPpoRolloutLongTrainedCheckpointCase,
)

__all__ = ["CASE", "GraphCheckpointRolloutAnalysisLongCase"]


class GraphCheckpointRolloutAnalysisLongCase(BaseAgenticCase):
    """Validate rollout execution from a longer trained checkpoint."""

    def __init__(self) -> None:
        """Initialize the longer trained checkpoint rollout analysis case."""
        super().__init__(
            CaseSpec(
                id="graph_checkpoint_rollout_analysis_long",
                title="Analyze rollout and plan larger-map transfer (dense checkpoints)",
                description=(
                    "Execute one reproducible greedy rollout from a denser-checkpoint "
                    f"trained PPO root on {ROLLOUT_ENV_ID} ({ROLLOUT_TRAINING_TIMESTEPS} timesteps, "
                    "checkpoints every 2048) and summarize next steps for a harder map."
                ),
                tags=frozenset({"graph", "operate", "read", "inference", "rollout", "trained"}),
                requirements=frozenset({"llm", "navix"}),
            ),
            NavixPpoRolloutLongTrainedCheckpointCase(),
            (
                AgenticTurn(
                    f"Necesito un diagnóstico de la política entrenada en {ROLLOUT_ENV_ID} "
                    f"mediante un rollout greedy reproducible con el checkpoint del nodo root "
                    f"en el step {ROLLOUT_CHECKPOINT_STEP} y seed {ROLLOUT_SEED}. "
                    "Este root guardó checkpoints más densos durante el entrenamiento. "
                    "Devuélveme un resumen estructurado con `checkpoint_eval.episode_return`, "
                    "`checkpoint_eval.episode_length`, el return y la longitud del rollout greedy, "
                    "y la razón de terminación. "
                    "No compares checkpoints para decidir cuál es mejor en este mapa fácil. "
                    "Usa el contexto del entrenamiento solo para inferir si la curva parece estable "
                    "o si harían falta más timesteps antes de transferir. "
                    "Propón qué mejorarías y los pasos concretos siguientes para adaptar el modelo "
                    "a un mapa Navix más grande o más difícil (env, budget, eval, hiperparámetros, "
                    "fork/extend en el grafo). No entrenes ni modifiques el grafo en esta sesión."
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


CASE = GraphCheckpointRolloutAnalysisLongCase()
"""Built-in longer trained checkpoint rollout analysis case."""
