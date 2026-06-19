import ctypes
import json
import os
import pathlib

import torch
import torch_tensorrt

def flatten_tensors(x):
    if isinstance(x, torch.Tensor):
        return [x]

    if isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.extend(flatten_tensors(item))
        return out

    return []


def tensor_meta(t):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
    }

def _make_input_spec(x):
    if isinstance(x, torch.Tensor):
        return torch_tensorrt.Input(
            shape=tuple(x.shape),
            dtype=x.dtype,
        )

    if isinstance(x, (list, tuple)):
        return type(x)(_make_input_spec(v) for v in x)

    raise TypeError(f"Unsupported TRT input type: {type(x)}")

def compile_trt_module(module, sample_inputs, settings):
    module = module.eval()

    exported = torch.export.export(
        module,
        args=sample_inputs,
        strict=False,
    )

    input_specs = _make_input_spec(sample_inputs)

    return torch_tensorrt.dynamo.compile(
        exported,
        inputs=input_specs,
        **settings,
    )

def export_trt_module(module, sample_inputs):
    try:
        return torch.export.export(
            module,
            args=sample_inputs,
            strict=False,
        )
    except Exception:
        return torch.export._trace._export(
            module,
            args=sample_inputs,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def save_trt_engine_module(
    module,
    sample_inputs,
    engine_dir,
    *,
    engine_file,
    model_type,
    component,
    input_names,
    output_names=None,
    example_output=None,
    extra_config=None,
    trt_settings=None,
    dual_optimization_profiles=False,
):
    module = module.eval()
    sample_inputs = tuple(sample_inputs)

    engine_dir = pathlib.Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    engine_path = engine_dir / engine_file
    config_path = engine_dir / "config.json"

    if example_output is None:
        with torch.no_grad():
            example_output = module(*sample_inputs)

    exported = export_trt_module(module, sample_inputs)

    input_specs = tuple(
        torch_tensorrt.Input(
            shape=tuple(t.shape),
            dtype=t.dtype,
            format=torch.contiguous_format,
            name=name,
        )
        for name, t in zip(input_names, sample_inputs)
    )

    settings = {
        "disable_tf32": True,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "truncate_double": True,
        "immutable_weights": True,
        "require_full_compilation": True,
        "min_block_size": 1,
        "workspace_size": 1 << 34,
    }
    if trt_settings:
        settings.update(trt_settings)

    from torch_tensorrt.dynamo.conversion import _TRTInterpreter as trt_interpreter

    prev_output_names = trt_interpreter.OUTPUT_NAMES_OVERRIDE
    prev_dual_profiles = trt_interpreter.FORCE_DUAL_OPTIMIZATION_PROFILES
    try:
        trt_interpreter.OUTPUT_NAMES_OVERRIDE = (
            list(output_names) if output_names else None
        )
        trt_interpreter.FORCE_DUAL_OPTIMIZATION_PROFILES = dual_optimization_profiles
        engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported,
            inputs=input_specs,
            **settings,
        )
    finally:
        trt_interpreter.OUTPUT_NAMES_OVERRIDE = prev_output_names
        trt_interpreter.FORCE_DUAL_OPTIMIZATION_PROFILES = prev_dual_profiles

    engine_path.write_bytes(engine_bytes)

    config = {
        "model_type": model_type,
        "component": component,
        "engine_file": engine_file,
        "precision": "FP16",
        "input_names": list(input_names),
        "inputs": {
            name: tensor_meta(t)
            for name, t in zip(input_names, sample_inputs)
        },
        "outputs": [
            tensor_meta(t)
            for t in flatten_tensors(example_output)
        ],
    }

    if output_names:
        config["output_names"] = list(output_names)

    if extra_config:
        config.update(extra_config)

    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return engine_path


def dump_tensor_bin(path: pathlib.Path, tensor: torch.Tensor) -> None:
    """Write a contiguous CPU tensor as a raw binary blob."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensor = tensor.detach().contiguous().cpu()
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
    path.write_bytes(ctypes.string_at(cpu_tensor.data_ptr(), nbytes))


def _fixture_filename(name: str) -> str:
    return name if name.endswith(".bin") else f"{name}.bin"


def dump_edge_fixture(
    engine_root: str | pathlib.Path,
    tensors: dict[str, torch.Tensor],
    *,
    fixture_subdir: str | None = None,
) -> pathlib.Path:
    """Write named tensor blobs for generic_run_inference under engine_root/fixtures/."""
    subdir = fixture_subdir or f"pid_{os.getpid()}"
    fixture_dir = pathlib.Path(engine_root) / "fixtures" / subdir
    for name, tensor in tensors.items():
        dump_tensor_bin(fixture_dir / _fixture_filename(name), tensor)
    return fixture_dir