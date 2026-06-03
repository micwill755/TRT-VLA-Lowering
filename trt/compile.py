import torch
import torch_tensorrt

def compile_trt_module(module, sample_inputs, trt_settings):
    input_specs = [
        torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
        for t in sample_inputs
    ]

    with torch.no_grad():
        exported = torch.export.export(module, sample_inputs, strict=False)

    return torch_tensorrt.dynamo.compile(
        exported,
        inputs=input_specs,
        **trt_settings,
    )