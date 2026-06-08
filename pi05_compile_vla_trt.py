# Test/pi05_compile_vla_trt.py
import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt

from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION

from trt.compile import compile_trt_module
from trt.measure import (
    compare_action_step,
    compare_full_vla_to_eager_actions,
    compare_language,
    compare_vision,
)
from trt.diffusion import PI05StaticKVDiffusionStep
from trt.utils import load_policy, build_prefix_inputs, sample_actions_eager, prepare_policy_inputs, make_suffix_position_and_mask
from trt.data import make_batch
from trt.vision import PI05VisualEmbed
from trt.language import (
    compact_prefix_inputs,
    compile_lm_trt_with_plugin,
    pi05_plugin_lm_smoke_check,
    run_pi05_plugin_language,
)
from trt.attention import PluginAttention, ViTPluginAttention
from trt.plugin_utils import (
    load_plugin,
    patch_vision_attention,  
    restore_attention,
    infer_siglip_seq_len,
)

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "min_block_size": 1,
    "use_python_runtime": True,
    "immutable_weights": True,
    "decompose_attention": True,
    "require_full_compilation": True,
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

def debug_prefix_eager_contract(prefix_embs, prefix_pad_masks, attention_mask, position_ids, name="prefix"):
    # attention_mask is 0 for allowed positions and large negative for blocked positions.
    allowed = attention_mask == 0
    if allowed.ndim != 4:
        raise ValueError(f"expected 4D attention_mask, got {allowed.shape}")

    bsz, _, q_len, kv_len = allowed.shape
    causal = torch.tril(
        torch.ones(q_len, kv_len, dtype=torch.bool, device=allowed.device)
    )[None, None, :, :]

    valid = prefix_pad_masks.to(torch.bool)
    valid_2d = valid[:, None, :, None] & valid[:, None, None, :]

    allowed_valid = allowed & valid_2d
    allowed_above_diag = allowed_valid & ~causal
    blocked_valid = valid_2d & ~allowed

    print(f"[{name}] prefix_embs:", tuple(prefix_embs.shape), prefix_embs.dtype)
    print(f"[{name}] prefix_pad_masks:", tuple(prefix_pad_masks.shape))
    print(f"[{name}] attention_mask:", tuple(attention_mask.shape), attention_mask.dtype)
    print(f"[{name}] position_ids:", tuple(position_ids.shape), position_ids.dtype)

    print(f"[{name}] valid tokens per batch:", valid.sum(dim=1).detach().cpu().tolist())
    print(f"[{name}] total seq len:", prefix_embs.shape[1])
    print(f"[{name}] position min/max:", int(position_ids.min()), int(position_ids.max()))

    print(f"[{name}] allowed valid entries:", int(allowed_valid.sum()))
    print(f"[{name}] allowed above diagonal:", int(allowed_above_diag.sum()))
    print(f"[{name}] blocked valid entries:", int(blocked_valid.sum()))

    is_plain_causal = torch.equal(allowed_valid, valid_2d & causal)
    is_full_valid_prefix = torch.equal(allowed_valid, valid_2d)

    print(f"[{name}] mask == causal valid mask:", is_plain_causal)
    print(f"[{name}] mask == full valid prefix mask:", is_full_valid_prefix)

def main() -> int:
    # We disabled TF32 so eager and TRT comparisons use stricter, more reproducible math while debugging accuracy.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_policy(PI05Policy, MODEL_ID, device, False).to(device).eval()
    batch = make_batch(policy, MODEL_ID, device, fill_missing=True)
    core = policy.model.to(device).eval()

    # images here are raw pixels
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)
    pixel_values = images[0]

    load_plugin()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")
    
    vision_model = core.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    batch_size, seq_len = infer_siglip_seq_len(vision_model, images[0])
    eager_model = PI05VisualEmbed(core).eval().to(device=device)

    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    trt_vision_model = compile_trt_module(
        eager_model,
        (images[0],),
        TRT_SETTINGS,
    )

    restore_attention(patched)
    
    compare_vision(core, images, trt_vision_model)
    
    # -------------------------
    # Prefix preprocessing
    # -------------------------

    # image embeddings are patch tokens after the vision tower
    trt_image_embs = [trt_vision_model(image) for image in images]

    eager_image_embs = [
        core.paligemma_with_expert.embed_image(image)
        for image in images
    ]

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_prefix_inputs(
        core,
        trt_image_embs,
        img_masks,
        tokens,
        masks,
    )
    prefix_embs = prefix_embs.to(torch.float16)

    eager_prefix_embs, eager_prefix_pad_masks, eager_prefix_attention_mask, eager_prefix_position_ids = build_prefix_inputs(
        core,
        eager_image_embs,
        img_masks,
        tokens,
        masks,
    )

    compact_prefix_embs, compact_prefix_pad_masks, compact_prefix_attention_mask, compact_prefix_position_ids = compact_prefix_inputs(
        prefix_embs,
        prefix_pad_masks,
        prefix_position_ids,
    )

    compact_eager_prefix_embs, compact_eager_prefix_pad_masks, compact_eager_prefix_attention_mask, compact_eager_prefix_position_ids = compact_prefix_inputs(
        eager_prefix_embs,
        eager_prefix_pad_masks,
        eager_prefix_position_ids,
    )

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")
    language_attention_cls = PluginAttention

    compact_trt_language, compact_language_max_seq_len = compile_lm_trt_with_plugin(
        core,
        compact_prefix_embs,
        device=device,
        position_ids=compact_prefix_position_ids,
        attention_cls=language_attention_cls,
        settings=TRT_SETTINGS,
        compile_trt_module=compile_trt_module,
    )

    pi05_plugin_lm_smoke_check(
        core,
        compact_trt_language,
        compact_prefix_embs,
        max_seq_len=compact_language_max_seq_len,
        device=device,
        attention_mask=compact_prefix_attention_mask,
        position_ids=compact_prefix_position_ids,
        prefix_pad_masks=compact_prefix_pad_masks,
        max_logit_tokens=16,
    )

    compact_prefix_k, compact_prefix_v = run_pi05_plugin_language(
        compact_trt_language,
        core,
        compact_prefix_embs,
        max_seq_len=compact_language_max_seq_len,
        device=device,
        prefix_pad_masks=compact_prefix_pad_masks,
    )

    def compact_prefix_runner(prefix_embs, attention_mask=None, position_ids=None, return_hidden=False):
        return run_pi05_plugin_language(
            compact_trt_language,
            core,
            prefix_embs,
            max_seq_len=compact_language_max_seq_len,
            device=device,
            attention_mask=attention_mask,
            return_hidden=return_hidden,
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

    eager_actions = sample_actions_eager(policy, batch, noise, num_steps, device)
    
    compare_language(
        core.paligemma_with_expert.paligemma.model.language_model,
        compact_prefix_runner,
        compact_eager_prefix_embs,
        compact_eager_prefix_attention_mask,
        compact_eager_prefix_position_ids,
        compact_eager_prefix_pad_masks,
    )

    # -------------------------
    # Action engine
    # -------------------------
    print("compiling action")
    action_module = PI05StaticKVDiffusionStep(core).eval().to(device)

    sample_inputs = make_compile_inputs(
        core,
        action_module,
        batch_size=tokens.shape[0],
        prefix_len=compact_prefix_pad_masks.shape[1],
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
    
    print("direct action step metrics")
    timestep = torch.ones(tokens.shape[0], dtype=torch.float32, device=device)
    compare_action_step(
        core,
        action_module,
        trt_action,
        compact_prefix_pad_masks,
        compact_prefix_k,
        compact_prefix_v,
        noise,
        timestep,
        device=device,
    )

    print("full action metrics")
    compare_full_vla_to_eager_actions(
        policy,
        batch,
        trt_vision_model,
        compact_prefix_runner,
        trt_action,
        eager_actions,
        noise,
        num_steps,
        device=device,
        compact_prefix=True,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())