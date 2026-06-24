from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from trt.utils import make_runner_inputs


@dataclass
class ActionRolloutContext:
    noise: torch.Tensor
    device: torch.device

    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None
    prefix_pad_mask: torch.Tensor | None = None

    context_embs: torch.Tensor | None = None
    state: torch.Tensor | None = None
    embodiment_id: torch.Tensor | None = None


class ActionRolloutAdapter(Protocol):
    def initial_actions(self, context: ActionRolloutContext) -> torch.Tensor:
        ...

    def num_steps(self, context: ActionRolloutContext) -> int:
        ...

    def make_timestep(
        self,
        step: int,
        actions: torch.Tensor,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        ...

    def make_runner_inputs(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        context: ActionRolloutContext,
    ) -> tuple:
        ...

    def update(
        self,
        actions: torch.Tensor,
        model_output: torch.Tensor,
        step: int,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        ...

    def finalize(
        self,
        actions: torch.Tensor,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        return actions


@torch.no_grad()
def sample_actions_raw(
    action_runner,
    context: ActionRolloutContext,
    adapter: ActionRolloutAdapter,
) -> torch.Tensor:
    actions = adapter.initial_actions(context)

    for step in range(adapter.num_steps(context)):
        timestep = adapter.make_timestep(step, actions, context)
        runner_inputs = adapter.make_runner_inputs(actions, timestep, context)

        model_output = action_runner(*runner_inputs)
        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]

        actions = adapter.update(actions, model_output, step, context)

    finalize = getattr(adapter, "finalize", None)
    if finalize is None:
        return actions
    return finalize(actions, context)


@dataclass
class PrefixKVFlowActionAdapter:
    """Flow-matching action rollout with prefix K/V (PI0.5, SmolVLA, etc.)."""

    core: object
    num_steps_value: int
    runner_inputs_fn: Callable[..., tuple] = make_runner_inputs

    def initial_actions(self, context: ActionRolloutContext) -> torch.Tensor:
        return context.noise.clone().to(device=context.device)

    def num_steps(self, context: ActionRolloutContext) -> int:
        return int(self.num_steps_value)

    def make_timestep(
        self,
        step: int,
        actions: torch.Tensor,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        dt = -1.0 / self.num_steps(context)
        return torch.full(
            (actions.shape[0],),
            1.0 + step * dt,
            dtype=torch.float32,
            device=actions.device,
        )

    def make_runner_inputs(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        context: ActionRolloutContext,
    ) -> tuple:
        return self.runner_inputs_fn(
            self.core,
            context.prefix_pad_mask,
            context.prefix_k,
            context.prefix_v,
            actions,
            timestep,
            context.device,
        )

    def update(
        self,
        actions: torch.Tensor,
        model_output: torch.Tensor,
        step: int,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        dt = -1.0 / self.num_steps(context)
        return actions + dt * model_output.float()


# Backward-compatible alias; prefer PrefixKVFlowActionAdapter.
PI05ActionAdapter = PrefixKVFlowActionAdapter


@dataclass
class GROOTActionAdapter:
    action_head: object

    def initial_actions(self, context: ActionRolloutContext) -> torch.Tensor:
        dtype = context.context_embs.dtype
        return context.noise.clone().to(device=context.device, dtype=dtype)

    def num_steps(self, context: ActionRolloutContext) -> int:
        return int(self.action_head.num_inference_timesteps)

    def make_timestep(
        self,
        step: int,
        actions: torch.Tensor,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        num_steps = self.num_steps(context)
        t_cont = step / float(num_steps)
        timestep_bucket = int(t_cont * self.action_head.num_timestep_buckets)

        return torch.full(
            (actions.shape[0],),
            timestep_bucket,
            device=actions.device,
            dtype=torch.long,
        )

    def make_runner_inputs(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        context: ActionRolloutContext,
    ) -> tuple:
        dtype = context.context_embs.dtype
        return (
            actions.to(device=context.device, dtype=dtype),
            timestep.to(device=context.device),
            context.context_embs.to(device=context.device, dtype=dtype),
            context.state.to(device=context.device, dtype=dtype),
            context.embodiment_id.to(device=context.device),
        )

    def update(
        self,
        actions: torch.Tensor,
        model_output: torch.Tensor,
        step: int,
        context: ActionRolloutContext,
    ) -> torch.Tensor:
        dt = 1.0 / self.num_steps(context)
        return actions + dt * model_output.float()
