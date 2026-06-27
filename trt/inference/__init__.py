"""VLA TensorRT inference pipeline (runtime tests + parity)."""

from trt.inference.backends import (
    EagerBackend,
    SerializedModuleBackend,
    StageHandles,
    TrtModuleBackend,
    stage_handles_from_modules,
)
from trt.inference.context import InferenceContext, InferenceResult, LanguageOutputs
from trt.inference.hooks import VLAInferenceHooks
from trt.inference.language_prefill import (
    LanguagePrefillInputs,
    build_language_prefill_inputs,
    run_language_prefill,
)
from trt.inference.mode import InferenceBackendKind, InferenceMode
from trt.inference.pipeline import VLAInferencePipeline

__all__ = [
    "EagerBackend",
    "InferenceBackendKind",
    "InferenceContext",
    "InferenceMode",
    "InferenceResult",
    "LanguageOutputs",
    "LanguagePrefillInputs",
    "SerializedModuleBackend",
    "StageHandles",
    "TrtModuleBackend",
    "VLAInferenceHooks",
    "VLAInferencePipeline",
    "build_language_prefill_inputs",
    "run_language_prefill",
    "stage_handles_from_modules",
]
