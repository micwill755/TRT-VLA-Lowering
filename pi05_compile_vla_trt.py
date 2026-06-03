# Test/pi05_compile_vla_trt.py
import torch
import torch.nn as nn
import torch_tensorrt

from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION

from trt.compile import compile_trt_module
from trt.measure import compare_full_vla_to_eager_actions, compare_language, compare_vision
from trt.diffusion import PI05StaticKVDiffusionStep
from trt.utils import load_policy, build_prefix_inputs, sample_actions_eager, prepare_policy_inputs, make_suffix_position_and_mask
from trt.data import make_batch
from trt.vision import PI05VisualEmbed
from trt.language import PI05PrefixLanguagePrefill

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "min_block_size": 1,
    "use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
}

ACTION_TRT_SETTINGS = {
    **TRT_SETTINGS,
    "offload_module_to_cpu": True,
}

MODEL_ID = "lerobot/pi05_libero"
SEED = 42

def make_compile_inputs(core, action_step, batch_size, prefix_len, device):
    chunk_size = core.config.chunk_size
    action_dim = core.config.max_action_dim
    expert_cfg = core.paligemma_with_expert.gemma_expert.model.config
    dtype = next(action_step.parameters()).dtype

    x_t = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=dtype)
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    prefix_k = torch.zeros(
        expert_cfg.num_hidden_layers,
        batch_size,
        expert_cfg.num_key_value_heads,
        prefix_len,
        expert_cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    prefix_v = torch.zeros_like(prefix_k)
    prefix_pad_masks = torch.ones(batch_size, prefix_len, dtype=torch.bool, device=device)

    position_ids, attention_mask = make_suffix_position_and_mask(core, prefix_pad_masks, x_t, device)
    return x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask

def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(PI05Policy, MODEL_ID, device, False).to(device).eval()
    batch = make_batch(policy, MODEL_ID, device, fill_missing=True)
    core = policy.model.to(device).eval()

    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch)

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")
    trt_visual = compile_trt_module(
        PI05VisualEmbed(core).eval().to(device),
        (images[0],),
        TRT_SETTINGS,
    )

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_prefix_inputs(
        core,
        images,
        img_masks,
        tokens,
        masks,
        visual_runner=trt_visual,
    )

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")
    core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

    trt_language = compile_trt_module(
        PI05PrefixLanguagePrefill(core).eval().to(device),
        (prefix_embs, prefix_attention_mask, prefix_position_ids),
        TRT_SETTINGS,
    )

    # -------------------------
    # Eager baseline before action compile/offload
    # -------------------------
    num_steps = core.config.num_inference_steps
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    noise = core.sample_noise(
        (tokens.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )

    eager_actions = sample_actions_eager(policy, batch, noise, num_steps)

    # -------------------------
    # Action engine
    # -------------------------
    print("compiling action")
    action_module = PI05StaticKVDiffusionStep(core).eval().to(device)

    sample_inputs = make_compile_inputs(
        core,
        action_module,
        batch_size=tokens.shape[0],
        prefix_len=prefix_pad_masks.shape[1],
        device=device,
    )

    trt_action = compile_trt_module(
        action_module,
        sample_inputs,
        ACTION_TRT_SETTINGS,
    )

    # -------------------------
    # Metrics
    # -------------------------
    print("metrics")
    compare_vision(core, images, trt_visual)

    eager_prefix_embs, _, eager_prefix_attention_mask, eager_prefix_position_ids = build_prefix_inputs(
        core,
        images,
        img_masks,
        tokens,
        masks,
        visual_runner=None,
    )

    compare_language(
        core.paligemma_with_expert.paligemma.model.language_model,
        trt_language,
        eager_prefix_embs,
        eager_prefix_attention_mask,
        eager_prefix_position_ids,
    )

    print("full action metrics")
    compare_full_vla_to_eager_actions(
        policy,
        batch,
        trt_visual,
        trt_language,
        trt_action,
        eager_actions,
        noise,
        num_steps,
        device=device,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())