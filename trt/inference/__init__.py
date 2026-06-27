"""VLA TensorRT inference pipeline (runtime tests + parity)."""

from trt.inference.backends import (
    EagerBackend,
    SerializedModuleBackend,
    StageHandles,
    TrtModuleBackend,
    stage_handles_from_modules,
)
from trt.inference.context import InferenceContext, InferenceResult, LanguageOutputs
from trt.inference.groot import (
    GrootInferenceHooks,
    compare_edge_pipeline_to_eager,
    compute_groot_policy_action_metrics,
    run_inference_pytorch_groot,
    run_inference_trt_plugin,
    run_serialized_action_context,
    run_serialized_language,
)
from trt.inference.molmoact2 import (
    MolmoAct2InferenceHooks,
    run_inference_molmoact2_engines,
    run_inference_pytorch_molmoact2,
    run_inference_trt_molmoact2,
)
from trt.inference.pi05 import (
    Pi05InferenceHooks,
    run_inference_pi05_engines,
    run_inference_pytorch_pi05,
    run_inference_trt_plugin as run_inference_trt_plugin_pi05,
)
from trt.inference.smolvla import (
    SmolVLAInferenceHooks,
    run_inference_pytorch_smolvla,
    run_inference_smolvla_engines,
)
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
    "GrootInferenceHooks",
    "InferenceBackendKind",
    "InferenceContext",
    "InferenceMode",
    "InferenceResult",
    "LanguageOutputs",
    "LanguagePrefillInputs",
    "MolmoAct2InferenceHooks",
    "run_inference_molmoact2_engines",
    "run_inference_pytorch_molmoact2",
    "run_inference_trt_molmoact2",
    "Pi05InferenceHooks",
    "SerializedModuleBackend",
    "SmolVLAInferenceHooks",
    "StageHandles",
    "TrtModuleBackend",
    "VLAInferenceHooks",
    "VLAInferencePipeline",
    "build_language_prefill_inputs",
    "compare_edge_pipeline_to_eager",
    "compute_groot_policy_action_metrics",
    "run_inference_pi05_engines",
    "run_inference_pytorch_groot",
    "run_inference_pytorch_pi05",
    "run_inference_pytorch_smolvla",
    "run_inference_smolvla_engines",
    "run_inference_trt_plugin",
    "run_inference_trt_plugin_pi05",
    "run_language_prefill",
    "run_serialized_action_context",
    "run_serialized_language",
    "stage_handles_from_modules",
]
