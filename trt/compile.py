import torch
import torch_tensorrt

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