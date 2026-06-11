from __future__ import annotations

import json
import pathlib
import torch 

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SerializedModuleSpec:
    name: str
    engine_subdir: str
    wrapper_cls: type

class SerializedTRTEngine:
    def __init__(self, engine_dir: str | pathlib.Path):
        import tensorrt as trt

        self.engine_dir = pathlib.Path(engine_dir)
        self.config = load_engine_config(self.engine_dir.parent, self.engine_dir.name)
        self.engine_path = self.engine_dir / self.config["engine_file"]
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Missing serialized TensorRT engine: {self.engine_path}")

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.input_tensor_names: list[str] = []
        self.output_tensor_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_tensor_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_tensor_names.append(name)

        self.config_input_names = list(self.config.get("input_names", self.config.get("inputs", {}).keys()))
        if len(self.config_input_names) != len(self.input_tensor_names):
            raise RuntimeError(
                f"Input count mismatch for {self.engine_path}: "
                f"config={self.config_input_names}, engine={self.input_tensor_names}"
            )

        self.input_name_map: dict[str, str] = {}
        for index, config_name in enumerate(self.config_input_names):
            if config_name in self.input_tensor_names:
                self.input_name_map[config_name] = config_name
            else:
                self.input_name_map[config_name] = self.input_tensor_names[index]

    def __call__(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if not inputs:
            raise ValueError("SerializedTRTEngine requires at least one input tensor")

        first = next(iter(inputs.values()))
        if first.device.type != "cuda":
            raise RuntimeError("Serialized TensorRT engines require CUDA tensors")
        device = first.device

        bound_inputs: dict[str, torch.Tensor] = {}
        for config_name in self.config_input_names:
            if config_name not in inputs:
                raise KeyError(f"Missing TensorRT input {config_name!r} for {self.engine_path}")
            actual_name = self.input_name_map[config_name]
            tensor = inputs[config_name].to(device=device).contiguous()
            ok = self.context.set_input_shape(actual_name, tuple(tensor.shape))
            if ok is False:
                raise RuntimeError(
                    f"Failed to set input shape for {actual_name}: {tuple(tensor.shape)}"
                )
            bound_inputs[actual_name] = tensor

        outputs: list[torch.Tensor] = []
        for output_index, actual_name in enumerate(self.output_tensor_names):
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(actual_name))
            if any(dim < 0 for dim in shape):
                shape = tuple(int(dim) for dim in self.config["outputs"][output_index]["shape"])
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(actual_name))
            outputs.append(torch.empty(shape, device=device, dtype=dtype))

        for actual_name, tensor in bound_inputs.items():
            ok = self.context.set_tensor_address(actual_name, tensor.data_ptr())
            if ok is False:
                raise RuntimeError(f"Failed to bind TensorRT input tensor: {actual_name}")

        for actual_name, tensor in zip(self.output_tensor_names, outputs):
            ok = self.context.set_tensor_address(actual_name, tensor.data_ptr())
            if ok is False:
                raise RuntimeError(f"Failed to bind TensorRT output tensor: {actual_name}")

        stream = torch.cuda.current_stream(device).cuda_stream
        ok = self.context.execute_async_v3(stream_handle=stream)
        if ok is False:
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")

        return tuple(outputs)

def _trt_dtype_to_torch(dtype) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.BF16:
        return torch.bfloat16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT64:
        return torch.int64
    if dtype == trt.DataType.INT8:
        return torch.int8
    if dtype == trt.DataType.UINT8:
        return torch.uint8
    if dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")

def _is_plugin_info_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))

def _add_config_to_plugin_info(plugin_info: dict, component_name: str, config: dict) -> None:
    plugin_info[f"{component_name}_config"] = config
    plugin_info[f"{component_name}_engine_file"] = config.get("engine_file")

    for key, value in config.items():
        if not _is_plugin_info_value(value):
            continue

        plugin_info[f"{component_name}_{key}"] = value

        if key not in plugin_info:
            plugin_info[key] = value


def _apply_plugin_info_aliases(
    plugin_info: dict,
    engines: dict[str, SerializedTRTEngine],
    aliases: dict[str, tuple[str, str]] | None,
) -> None:
    if not aliases:
        return

    for output_key, source in aliases.items():
        component_name, config_key = source
        plugin_info[output_key] = engines[component_name].config[config_key]

def load_engine_config(engine_root: pathlib.Path, component: str) -> dict:
    with (engine_root / component / "config.json").open() as f:
        return json.load(f)

def load_serialized_modules(
    engine_root,
    *,
    specs: tuple[SerializedModuleSpec, ...],
    plugin_info_aliases: dict[str, tuple[str, str]] | None = None,
):
    engine_root = pathlib.Path(engine_root)

    engines: dict[str, SerializedTRTEngine] = {}
    modules = []
    plugin_info = {
        "engine_root": str(engine_root),
    }

    for spec in specs:
        engine_dir = engine_root / spec.engine_subdir
        engine = SerializedTRTEngine(engine_dir)

        engines[spec.name] = engine
        modules.append(spec.wrapper_cls(engine))

        plugin_info[f"{spec.name}_engine_dir"] = str(engine_dir)
        plugin_info[f"{spec.name}_engine_path"] = str(engine.engine_path)

        _add_config_to_plugin_info(
            plugin_info,
            spec.name,
            engine.config,
        )

    _apply_plugin_info_aliases(
        plugin_info,
        engines,
        plugin_info_aliases,
    )

    return (*modules, plugin_info)

class SerializedGrootVision:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, pixel_values):
        return self.engine({
            "pixel_values": pixel_values,
        })[0]

class SerializedGrootLanguage:
    def __init__(self, engine):
        self.engine = engine
        self.max_seq_len = int(engine.config["max_seq_len"])

    def __call__(self, input_embs, kv_caches, ctx_len):
        inputs = {
            "inputs_embeds": input_embs,
            "ctx_len": ctx_len,
        }

        for i, kv_cache in enumerate(kv_caches):
            inputs[f"kv_cache_{i}"] = kv_cache

        return self.engine(inputs)[0]

class SerializedGrootAction:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, actions, timestep, context_embs, state, embodiment_id):
        return self.engine({
            "actions": actions,
            "timestep": timestep,
            "context_embs": context_embs,
            "state": state,
            "embodiment_id": embodiment_id,
        })[0]