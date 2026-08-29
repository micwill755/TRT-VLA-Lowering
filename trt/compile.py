import json
import pathlib
from contextlib import contextmanager
from inspect import Parameter, signature

import torch
import torch.nn as nn
import torch_tensorrt

from torch.export import export
from torch_tensorrt.dynamo.utils import default_device, to_torch_device

try:
    # Newer torch-tensorrt exposes a shared dim registry so dynamic dims with the
    # same name are the same symbolic Dim across inputs.
    from torch_tensorrt.dynamo._tracer import build_dim_registry, get_dynamic_shapes
except ImportError:
    # torch-tensorrt 2.10 (TensorRT 10.14 / DriveOS 7.0.5 on Thor) does not ship
    # build_dim_registry and uses a single-arg get_dynamic_shapes. Provide shims
    # with the two-arg signature this module expects.
    from torch.export import Dim
    from torch_tensorrt._Input import Input as _TRTInput

    def _iter_input_specs(specs):
        if isinstance(specs, _TRTInput):
            yield specs
        elif isinstance(specs, (list, tuple)):
            for item in specs:
                yield from _iter_input_specs(item)

    def _dynamic_dim_bounds(spec):
        min_shape = spec.shape["min_shape"]
        opt_shape = spec.shape["opt_shape"]
        max_shape = spec.shape["max_shape"]
        for dim in range(len(min_shape)):
            if min_shape[dim] == opt_shape[dim] == max_shape[dim]:
                continue
            yield dim, min_shape[dim], max_shape[dim]

    def build_dim_registry(specs, registry=None):
        registry = dict(registry or {})
        for spec in _iter_input_specs(specs):
            if getattr(spec, "shape_mode", None) != _TRTInput._ShapeMode.DYNAMIC:
                continue
            for dim, min_v, max_v in _dynamic_dim_bounds(spec):
                key = f"{spec.name}_{dim}"
                if key not in registry:
                    registry[key] = Dim(key, min=min_v, max=max_v)
        return registry

    def get_dynamic_shapes(spec, registry=None):
        registry = registry if registry is not None else {}
        if not isinstance(spec, _TRTInput):
            return {}
        if getattr(spec, "shape_mode", None) != _TRTInput._ShapeMode.DYNAMIC:
            return {}
        dynamic_dims = {}
        for dim, min_v, max_v in _dynamic_dim_bounds(spec):
            key = f"{spec.name}_{dim}"
            dynamic_dims[dim] = registry.get(key) or Dim(key, min=min_v, max=max_v)
        return dynamic_dims

from trt.io_spec import VLA_LANGUAGE_LEADING_INPUT_COUNT

LANGUAGE_EDGE_LEADING_INPUT_COUNT = VLA_LANGUAGE_LEADING_INPUT_COUNT


def count_leading_language_inputs(input_names: list[str] | tuple[str, ...]) -> int:
    """Count fixed language bindings before ``past_key_values_*`` tensors."""
    count = 0
    for name in input_names:
        if name.startswith("past_key_values"):
            break
        count += 1
    return count


def trace_language_for_edge_llm(
    module: nn.Module,
    flat_trace_tensors: tuple,
    input_specs: tuple,
    *,
    device=None,
    leading_input_count: int | None = None,
):
    """Export Edge-LLM language module with multi-profile ``Input`` specs."""
    if leading_input_count is None:
        leading_input_count = LANGUAGE_EDGE_LEADING_INPUT_COUNT
    leading_input_count = int(leading_input_count)

    if len(flat_trace_tensors) < leading_input_count + 1:
        raise ValueError(
            "flat_trace_tensors must include leading bindings plus at least one KV cache"
        )
    if len(input_specs) < leading_input_count:
        raise ValueError(
            f"Expected at least {leading_input_count} flat input specs"
        )

    leading_specs = tuple(input_specs[:leading_input_count])
    kv_tensors = flat_trace_tensors[leading_input_count:]

    resolved_device = to_torch_device(device or default_device())
    torch_arg_inputs = tuple(
        t.to(resolved_device) if isinstance(t, torch.Tensor) else t
        for t in flat_trace_tensors
    )

    dim_registry = build_dim_registry(leading_specs, {})
    dynamic_shapes = {}

    positional_names: list[str] = []
    var_pos_name: str | None = None
    for param in signature(module.forward).parameters.values():
        if param.kind == Parameter.VAR_POSITIONAL:
            var_pos_name = param.name
            break
        if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            positional_names.append(param.name)

    for spec, name in zip(leading_specs, positional_names[:leading_input_count]):
        if name == "inputs_embeds":
            dynamic_shapes[name] = get_dynamic_shapes(spec, dim_registry)
        else:
            dynamic_shapes[name] = {}

    if var_pos_name is not None:
        dynamic_shapes[var_pos_name] = tuple({} for _ in range(len(kv_tensors)))

    export_kwargs = dict(
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    try:
        return export(module, torch_arg_inputs, **export_kwargs)
    except Exception:
        from torch.export._trace import _export

        return _export(
            module,
            torch_arg_inputs,
            prefer_deferred_runtime_asserts_over_guards=True,
            **export_kwargs,
        )


@contextmanager
def patch_trt_interpreter_output_names(output_names: list[str] | tuple[str, ...] | None):
    """Patch TRTInterpreter to use Edge-LLM output binding names (logits, context_embs)."""
    if not output_names:
        yield
        return

    import tensorrt as trt
    from torch_tensorrt._enums import dtype
    import torch_tensorrt.dynamo.conversion._TRTInterpreter as tri_module

    original_output = tri_module.TRTInterpreter.output

    def output(self, target, args, kwargs):
        assert len(args) == 1
        if isinstance(args[0], tuple):
            outputs = args[0]
        elif isinstance(args[0], list):
            outputs = tuple(args[0])
        else:
            outputs = (args[0],)

        for output_idx in range(len(outputs)):
            output = outputs[output_idx]
            if not isinstance(output, trt.ITensor):
                from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

                new_output = get_trt_tensor(self.ctx, output, target)
                outputs = (
                    outputs[:output_idx] + (new_output,) + outputs[output_idx + 1 :]
                )

        if not all(isinstance(output, trt.ITensor) for output in outputs):
            raise RuntimeError("TensorRT requires all outputs to be Tensor!")

        if self.output_dtypes is not None and len(self.output_dtypes) != len(outputs):
            raise RuntimeError(
                f"Specified output dtypes ({len(self.output_dtypes)}) differ from number of outputs ({len(outputs)})"
            )

        marked_outputs_ids = []
        for i, output in enumerate(outputs):
            if id(output) in marked_outputs_ids:
                continue
            marked_outputs_ids.append(id(output))

            name = output_names[i] if i < len(output_names) else f"output{i}"

            if any(
                op_name in output.name.split("_")
                for op_name in (
                    "eq",
                    "gt",
                    "lt",
                    "or",
                    "xor",
                    "and",
                    "not",
                    "ne",
                    "isinf",
                    "isnan",
                    "any",
                )
            ):
                output_dtype = dtype.b
            elif self.output_dtypes is not None:
                output_dtype = self.output_dtypes[i]
                if output_dtype == dtype.i64 and not hasattr(self, "_cast_output_dtype"):
                    output = self.ctx.net.add_cast(
                        output, dtype.i64.to(trt.DataType)
                    ).get_output(0)
            else:
                output_dtype = dtype.unknown

            if output_dtype is not dtype.unknown:
                trt_dtype = output_dtype.to(trt.DataType, use_default=True)
                if hasattr(self, "_cast_output_dtype"):
                    output = self._cast_output_dtype(output, trt_dtype, name)

            output.name = name
            outputs = outputs[:i] + (output,) + outputs[i + 1 :]
            self.ctx.net.mark_output(output)
            if (
                output_dtype is not dtype.unknown
                and not hasattr(self, "_cast_output_dtype")
                and output_dtype != dtype.i64
            ):
                output.dtype = output_dtype.to(trt.DataType, use_default=True)
            self._output_names.append(name)

        return list(outputs)

    tri_module.TRTInterpreter.output = output
    try:
        yield
    finally:
        tri_module.TRTInterpreter.output = original_output


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

def make_input_spec(x):
    if isinstance(x, torch.Tensor):
        return torch_tensorrt.Input(
            shape=tuple(x.shape),
            dtype=x.dtype,
        )

    if isinstance(x, (list, tuple)):
        return type(x)(make_input_spec(v) for v in x)

    raise TypeError(f"Unsupported TRT input type: {type(x)}")

def compile_trt_module(module, sample_inputs, settings):
    module = module.eval()

    exported = torch.export.export(
        module,
        args=sample_inputs,
        strict=False,
    )

    input_specs = make_input_spec(sample_inputs)

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


def _infer_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    extra_config=None,
    trt_settings=None,
    input_specs=None,
    flat_tensors=None,
    trace_tensors=None,
):
    module = module.eval()
    sample_inputs = tuple(sample_inputs)
    flat_tensors = tuple(flat_tensors if flat_tensors is not None else sample_inputs)
    # Dual-profile language traces below capacity so ``inputs_embeds`` stays
    # dynamic; ``flat_tensors`` still records full-capacity config metadata.
    trace_for_export = tuple(trace_tensors) if trace_tensors is not None else flat_tensors

    engine_dir = pathlib.Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    engine_path = engine_dir / engine_file
    config_path = engine_dir / "config.json"

    with torch.no_grad():
        example_output = module(*sample_inputs)

    if input_specs is not None:
        exported = trace_language_for_edge_llm(
            module,
            trace_for_export,
            input_specs,
            device=_infer_module_device(module),
            leading_input_count=count_leading_language_inputs(input_names),
        )
    else:
        exported = export_trt_module(module, sample_inputs)
        input_specs = tuple(
            torch_tensorrt.Input(
                shape=tuple(t.shape),
                dtype=t.dtype,
                format=torch.contiguous_format,
                name=name,
            )
            for name, t in zip(input_names, flat_tensors)
        )

    settings = dict(trt_settings or {})
    module_device = _infer_module_device(module)

    with patch_trt_interpreter_output_names(output_names):
        engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported,
            inputs=input_specs,
            **settings,
        )

    if settings.get("offload_module_to_cpu"):
        module.to(module_device)

    engine_path.write_bytes(engine_bytes)

    config = {
        "model_type": model_type,
        "component": component,
        "engine_file": engine_file,
        "precision": "FP16",
        "input_names": list(input_names),
        "inputs": {
            name: tensor_meta(t)
            for name, t in zip(input_names, flat_tensors)
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