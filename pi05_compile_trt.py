# Test/pi05_compile_vla_trt.py
import copy

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt

from lerobot.policies.pi05 import PI05Policy
from trt.action_rollout import ActionRolloutContext, PI05ActionAdapter, sample_actions_raw
from trt.compile import compile_trt_module
from trt.measure import (
    compare_action_step,
    compare_full_vla_to_eager_actions,
    compare_language,
    compare_vision,
)
from trt.diffusion import PI05StaticKVDiffusionStep
from trt.utils import (
    load_policy, 
    build_packed_prefix_inputs,
    prepare_policy_inputs, 
    make_suffix_position_and_mask
)
from trt.data import make_batch
from trt.packing import compact_packed_language_inputs
from trt.vision import PI05VisualEmbed
from trt.language import (
    compile_language_trt_with_plugin,
    language_head_dim,
    make_prefill_kvcache_start_index,
    make_rope_rotary_cos_sin,
    make_plugin_lm_hidden_wrapper,
    pi05_plugin_lm_smoke_check,
    run_prefix_language_eager
)
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

def main() -> int:
    # -------------------------
    # Runtime setup
    # -------------------------
    # Disable TF32 so eager and TRT comparisons use stricter, more reproducible math.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Put every model and tensor on CUDA when available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained PI0.5 policy.
    policy = load_policy(PI05Policy, MODEL_ID, device, False).to(device).eval()
    # Build one representative batch for compilation and metric checks.
    batch = make_batch(policy, MODEL_ID, device, fill_missing=True)
    # The core model owns the vision, language, and action modules used below.
    core = policy.model.to(device).eval()

    # Prepare the four raw policy inputs: image pixels, image masks, language tokens, and token masks.
    images, img_masks, tokens, masks = prepare_policy_inputs(policy, batch, device)
    # Keep the first image stream as the representative compile input for the vision engine.
    pixel_values = images[0]

    # Load the custom TensorRT plugin library before compiling plugin-backed modules.
    load_plugin()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    # The PI0.5 image encoder is the SigLIP vision model inside PaliGemma.
    vision_model = core.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    # Infer the image-token sequence length so the patched attention plugin has the right shape.
    batch_size, seq_len = infer_siglip_seq_len(vision_model, pixel_values)
    # Wrap eager image embedding as a clean image -> patch-token module for TensorRT export.
    eager_model = PI05VisualEmbed(core).eval().to(device=device)

    # Temporarily replace SigLIP attention with the TensorRT plugin-friendly attention path.
    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP",
    )
    # Compile the vision/image-embedding module using one raw image tensor as the sample input.
    trt_vision_model = compile_trt_module(
        eager_model,
        (pixel_values,),
        TRT_SETTINGS,
    )

    # Restore the original eager attention modules after TensorRT export is complete.
    restore_attention(patched)

    # Compare TRT image patch embeddings against eager image patch embeddings for each image stream.
    compare_vision(core, images, trt_vision_model)

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")

    # -------------------------
    # Original eager prefix creation
    # -------------------------
    # Run the original eager vision tower to get image patch-token embeddings.
    eager_image_embs = [
        core.paligemma_with_expert.embed_image(image)
        for image in images
    ]
    # Combine eager image tokens with language-token embeddings and build masks/positions.
    eager_prefix = build_packed_prefix_inputs(
        core,
        eager_image_embs,
        img_masks,
        tokens,
        masks,
    )

    # Remove padded prefix slots so the eager reference uses the same dense sequence as TRT.
    compact_eager_prefix = compact_packed_language_inputs(eager_prefix)
    (
        compact_eager_prefix_embs,
        compact_eager_prefix_pad_masks,
        compact_eager_prefix_attention_mask,
        compact_eager_prefix_position_ids,
    ) = compact_eager_prefix.as_tuple()

    # Run the original language model over the compact eager prefix to produce reference hidden/KV tensors.
    eager_hidden, eager_prefix_k, eager_prefix_v = run_prefix_language_eager(
        core.paligemma_with_expert.paligemma.model.language_model,
        compact_eager_prefix_embs,
        compact_eager_prefix_attention_mask,
        compact_eager_prefix_position_ids,
    )

    # -------------------------
    # TRT prefix creation
    # -------------------------
    # Run the same raw images through the compiled vision module to get TRT image patch-token embeddings.
    trt_image_embs = [trt_vision_model(image) for image in images]

    # Combine TRT image tokens with eager language embeddings and build the same prefix metadata.
    prefix = build_packed_prefix_inputs(
        core,
        trt_image_embs,
        img_masks,
        tokens,
        masks,
    )
    # The plugin language model runs in fp16.
    prefix = prefix.with_inputs_embeds(prefix.inputs_embeds.to(torch.float16))

    # Remove padded prefix slots before compiling/running the TRT language prefill engine.
    compact_trt_prefix = compact_packed_language_inputs(prefix)
    (
        compact_trt_prefix_embs,
        compact_trt_prefix_pad_masks,
        compact_trt_prefix_attention_mask,
        compact_trt_prefix_position_ids,
    ) = compact_trt_prefix.as_tuple()

    # Compile the PaliGemma language stack with plugin attention for compact prefix prefill.
    lm = copy.deepcopy(
        core.paligemma_with_expert.paligemma.model.language_model
    ).to(device=device, dtype=torch.float16).eval()
    decoder = getattr(lm, "model", lm)
    cfg = lm.config
    plugin_language = make_plugin_lm_hidden_wrapper(
        decoder,
        cfg,
        max_seq_len=int(compact_trt_prefix_embs.shape[1]),
        device=device,
        position_ids=compact_trt_prefix_position_ids,
        return_prefix_kv=True,
    )
    trt_language_model, trt_max_seq_len = compile_language_trt_with_plugin(
        plugin_language,
        compact_trt_prefix_embs,
        num_layers=int(cfg.num_hidden_layers),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=language_head_dim(cfg),
        device=device,
        settings=TRT_SETTINGS,
    )

    # Run the compiled language model to produce TRT hidden states and the prefix KV cache.
    trt_prefix_embs = compact_trt_prefix_embs.to(device=device, dtype=torch.float16)
    trt_kv_caches = [
        torch.zeros(
            int(trt_prefix_embs.shape[0]),
            2,  # key + value
            int(cfg.num_key_value_heads),
            trt_max_seq_len,
            language_head_dim(cfg),
            device=device,
            dtype=trt_prefix_embs.dtype,
        )
        for _ in range(int(cfg.num_hidden_layers))
    ]
    trt_ctx_len = torch.full(
        (trt_prefix_embs.shape[0],),
        trt_prefix_embs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    trt_rope = make_rope_rotary_cos_sin(
        cfg,
        trt_max_seq_len,
        device,
        language_model=lm,
        position_ids=compact_trt_prefix_position_ids,
    )
    trt_kvcache_start_index = make_prefill_kvcache_start_index(device)
    trt_hidden, trt_prefix_k, trt_prefix_v = trt_language_model(
        trt_prefix_embs,
        trt_rope,
        trt_ctx_len,
        trt_kvcache_start_index,
        trt_kv_caches,
    )

    # Smoke-check the plugin language output with logits and KV-cache comparisons.
    pi05_plugin_lm_smoke_check(
        core,
        trt_language_model,
        compact_trt_prefix_embs,
        max_seq_len=trt_max_seq_len,
        device=device,
        attention_mask=compact_trt_prefix_attention_mask,
        position_ids=compact_trt_prefix_position_ids,
        prefix_pad_masks=compact_trt_prefix_pad_masks,
        max_logit_tokens=16,
    )

    # Compare eager and TRT language hidden states plus prefix K/V caches.
    compare_language(
        eager_hidden,
        eager_prefix_k,
        eager_prefix_v,
        trt_hidden,
        trt_prefix_k,
        trt_prefix_v,
        compact_trt_prefix_pad_masks,
    )

    # -------------------------
    # Eager baseline before action compile/offload
    # -------------------------
    # Use the policy's configured number of denoising steps for the action rollout.
    num_steps = core.config.num_inference_steps
    # Seed the diffusion noise so eager and TRT rollouts start from the same sample.
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # Sample the initial noisy action chunk in the model's full internal action dimension.
    noise = core.sample_noise(
        (tokens.shape[0], core.config.chunk_size, core.config.max_action_dim),
        device,
    )

    # The action module consumes prefix KV cache, noisy actions, and timestep to predict denoising velocity.
    action_module = PI05StaticKVDiffusionStep(core).eval().to(device)

    # Run the eager action module through the shared raw-tensor rollout loop.
    eager_action_context = ActionRolloutContext(
        noise=noise,
        device=device,
        prefix_k=eager_prefix_k,
        prefix_v=eager_prefix_v,
        prefix_pad_mask=compact_eager_prefix_pad_masks,
    )
    eager_actions = sample_actions_raw(
        action_module,
        eager_action_context,
        PI05ActionAdapter(core, num_steps),
    )

    # -------------------------
    # Action engine
    # -------------------------
    print("compiling action")

    # Build representative action-step inputs that match the compact prefix KV length.
    sample_inputs = make_compile_inputs(
        core,
        action_module,
        batch_size=tokens.shape[0],
        prefix_len=compact_trt_prefix_pad_masks.shape[1],
        device=device,
    )

    # Compile the static one-step denoising module.
    trt_action = compile_trt_module(
        action_module,
        sample_inputs,
        ACTION_TRT_SETTINGS,
    )

    # -------------------------
    # Metrics
    # -------------------------
    print("direct action step metrics")
    # Compare a single denoising step with identical prefix KV, noise, and timestep.
    timestep = torch.ones(tokens.shape[0], dtype=torch.float32, device=device)
    compare_action_step(
        core,
        action_module,
        trt_action,
        compact_trt_prefix_pad_masks,
        trt_prefix_k,
        trt_prefix_v,
        noise,
        timestep,
        device=device,
    )

    print("full action metrics")
    # Roll the whole TRT pipeline through every denoising step and compare final actions to eager.
    trt_actions = compare_full_vla_to_eager_actions(
        policy,
        batch,
        trt_prefix_k,
        trt_prefix_v,
        trt_vision_model,
        trt_action,
        eager_actions,
        noise,
        num_steps,
        device=device,
        compact_prefix=True,
    )

    # Signal successful script completion.
    return 0

if __name__ == "__main__":
    raise SystemExit(main())