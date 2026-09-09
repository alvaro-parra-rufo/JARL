"""Public checkpoint-inference contracts and backend discovery."""

from typing import TYPE_CHECKING

from jarl.inference.backend import InferenceBackend, InferenceEnvironment
from jarl.inference.policy import (
    GreedyPolicyStepFn,
    InferencePolicyRuntime,
    InitialPolicyStateFn,
    LoadedInferencePolicy,
    PolicyState,
    PreprocessObservationFn,
)
from jarl.inference.registry import (
    INFERENCE_BACKEND_REGISTRY,
    list_inference_backend_names,
    resolve_inference_backend_from_name,
)

if TYPE_CHECKING:
    from jarl.inference.run import (
        CheckpointInferenceRequest,
        CheckpointInferenceResult,
        resolve_inference_horizon,
        run_checkpoint_inference,
    )

__all__ = [
    "INFERENCE_BACKEND_REGISTRY",
    "CheckpointInferenceRequest",
    "CheckpointInferenceResult",
    "GreedyPolicyStepFn",
    "InferenceBackend",
    "InferenceEnvironment",
    "InferencePolicyRuntime",
    "InitialPolicyStateFn",
    "LoadedInferencePolicy",
    "PolicyState",
    "PreprocessObservationFn",
    "list_inference_backend_names",
    "resolve_inference_backend_from_name",
    "resolve_inference_horizon",
    "run_checkpoint_inference",
]


def __getattr__(name: str) -> object:
    """Lazily expose inference orchestration without coupling policy imports."""
    if name not in {
        "CheckpointInferenceRequest",
        "CheckpointInferenceResult",
        "resolve_inference_horizon",
        "run_checkpoint_inference",
    }:
        raise AttributeError(name)
    from jarl.inference.run import (
        CheckpointInferenceRequest,
        CheckpointInferenceResult,
        resolve_inference_horizon,
        run_checkpoint_inference,
    )

    exports = {
        "CheckpointInferenceRequest": CheckpointInferenceRequest,
        "CheckpointInferenceResult": CheckpointInferenceResult,
        "resolve_inference_horizon": resolve_inference_horizon,
        "run_checkpoint_inference": run_checkpoint_inference,
    }
    return exports[name]
