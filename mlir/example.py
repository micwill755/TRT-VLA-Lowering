import torch, torch_bear.frontends

@torch.compile(backend="torch-bear", dynamic=True)
def f(x):
    return torch.nn.functional.gelu(torch.mm(x, x))

f(torch.randn(4, 4))
