import torch
import torch_tensorrt


'''def run_vlm_preprocessing(
    model: nn.Module,
    model_inputs: dict[str, Any],
    trt_vision: nn.Module | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = torch.device(device)
    tokenized_data = copy.deepcopy(model_inputs["tokenzied_data"])
    pass'''

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