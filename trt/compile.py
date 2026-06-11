import json
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

    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        inputs=input_specs,
        **settings,
    )

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