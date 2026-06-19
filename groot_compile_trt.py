import torch
import copy

from transformers import AutoProcessor

from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.groot_n1 import DEFAULT_TOKENIZER_ASSETS_REPO
from lerobot.utils.constants import HF_LEROBOT_HOME

from trt.action_rollout import ActionRolloutContext, GROOTActionAdapter, sample_actions_raw
from trt.compile import compile_trt_module
from trt.diffusion import GrootStaticDiffusionStep
from trt.utils import (
    load_policy,
    compact_prefix_inputs,
    prepare_policy_inputs_groot,
)
from trt.helper import (
    get_processor
)
from trt.data import (
    load_test_data,
    prepare_model_inputs,
    make_batch,
    pack_state
)
from trt.packing import (
    MultimodalPromptProcessor,
    PackedLanguageInputs,
    PromptPackingSpec,
    PromptTensorInputs,
)
from trt.vision import GROOTVisualEmbed
from trt.language import (
    compile_language_trt_with_plugin,
    GROOTContextProjectionWrapper,
    GROOTLanguageContextWrapper,
    language_head_dim,
    make_plugin_lm_hidden_wrapper,
    make_prefill_kvcache_start_index,
)
from trt.rope import (
    make_dummy_rope_rotary_cos_sin,
    make_rope_rotary_cos_sin,
)
from trt.measure import (
    compare_full_groot_to_eager_actions,
    compare_groot_action_step,
    compare_vision,
    tensor_error_metrics,
)
from trt.plugin_utils import (
    register_plugin_op,
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

MODEL_ID = "nvidia/GR00T-N1.5-3B"
SEED = 42

GROOT_EMBODIMENT_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
    "so100": 2,
    "unitree_g1": 3,
}

def make_compile_inputs(action_step, vl_embs, state, embodiment_id, device):
    batch_size = vl_embs.shape[0]
    dtype = vl_embs.dtype

    action_horizon = action_step.action_horizon
    action_dim = action_step.action_decoder.layer2.b.shape[-1]

    actions = torch.randn(
        batch_size,
        action_horizon,
        action_dim,
        device=device,
        dtype=dtype,
    )

    timestep = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.long,
    )

    return (
        actions,
        timestep,
        vl_embs,
        state,
        embodiment_id,
    )

@torch.no_grad()
def build_groot_language_inputs(core, vit_embs, input_ids, attention_mask=None) -> PackedLanguageInputs:
    eagle = core.backbone.eagle_model
    image_token_index = getattr(
        eagle,
        "image_token_index",
        eagle.config.image_token_index,
    )

    processor = MultimodalPromptProcessor(
        PromptPackingSpec(
            style="chat_template_placeholder",
            token_embed_fn=eagle.language_model.get_input_embeddings(),
            image_token_id=image_token_index,
        )
    )

    return processor(
        PromptTensorInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embs=vit_embs,
        )
    )

@torch.no_grad()
def build_groot_context_inputs(core, vit_embs, input_ids, attention_mask):
    eagle = core.backbone.eagle_model
    packed = build_groot_language_inputs(
        core,
        vit_embs,
        input_ids,
        attention_mask,
    )

    out = eagle.language_model(
        inputs_embeds=packed.inputs_embeds,
        attention_mask=packed.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    context_embs = out.hidden_states[core.backbone.select_layer]
    context_embs = core.backbone.eagle_linear(context_embs)

    # Match action_head.process_backbone_output().
    vlln_weight = getattr(core.action_head.vlln, "weight", None)
    if vlln_weight is not None:
        context_embs = context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
    context_embs = core.action_head.vlln(context_embs)
    context_embs = core.action_head.vl_self_attention(context_embs)

    return (
        context_embs,
        packed.pad_mask,
        packed.attention_mask,
        packed.position_ids,
    )

def make_groot_context_masks(context_embs, attention_mask):
    context_pad_masks = attention_mask.to(device=context_embs.device, dtype=torch.bool)
    context_position_ids = torch.cumsum(context_pad_masks, dim=1) - 1

    return compact_prefix_inputs(
        context_embs,
        context_pad_masks,
        context_position_ids,
    )

@torch.no_grad()
def compare_groot_context(eager_context_embs, trt_context_embs, attention_mask, name="groot context"):

    compact_eager, _, _, _ = make_groot_context_masks(eager_context_embs, attention_mask)
    compact_trt, _, _, _ = make_groot_context_masks(trt_context_embs, attention_mask)

    tensor_error_metrics(name, compact_trt, compact_eager)

def _select_context_rows(context_embs, row_mask):
    row_mask = row_mask.to(device=context_embs.device, dtype=torch.bool)
    return torch.cat(
        [context_embs[b, row_mask[b], :] for b in range(context_embs.shape[0])],
        dim=0,
    )

@torch.no_grad()
def compare_groot_context_token_types(core, eager_context_embs, trt_context_embs, input_ids, attention_mask, name):
    eagle = core.backbone.eagle_model
    image_token_index = getattr(eagle, "image_token_index", eagle.config.image_token_index)

    valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    image_tokens = (input_ids == image_token_index) & valid
    text_tokens = (input_ids != image_token_index) & valid

    if int(image_tokens.sum().item()) > 0:
        tensor_error_metrics(
            f"{name} image tokens",
            _select_context_rows(trt_context_embs, image_tokens),
            _select_context_rows(eager_context_embs, image_tokens),
        )

    if int(text_tokens.sum().item()) > 0:
        tensor_error_metrics(
            f"{name} text tokens",
            _select_context_rows(trt_context_embs, text_tokens),
            _select_context_rows(eager_context_embs, text_tokens),
        )

def main() -> int:
    # Put every model and tensor on CUDA when available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained GROOT policy.
    policy = load_policy(GrootPolicy, MODEL_ID, device).to(device).eval()
    # The core model owns the vision, language/context, and action modules used below.
    model = policy._groot_model.to(device).eval()

    data, messages = load_test_data(
        dataset_id="lerobot/libero",
        episode_index=0,
        frame_index=0,
    )

    # create processor and get tokenizer ----
    cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
    processor = get_processor(str(cache_dir), 
        {
            'trust_remote_code': True, 
            'fix_mistral_regex': False
        })
    # ------
    
    model_inputs = prepare_model_inputs(
        processor,
        processor.process_vision_info,
        {"add_generation_prompt": True},
        {
            "images_kwargs": {
                "min_dynamic_tiles": 1,
                "max_dynamic_tiles": 1,
                "use_thumbnail": False,
            },
        },
        data,
        messages,
        device,
    )

    tokenized_data = model_inputs['tokenized_data']
    input_ids = tokenized_data['input_ids']

    # groot specifc inputs ------
    attention_mask = tokenized_data['attention_mask']
    state, state_mask = pack_state(
        data["state"],
        max_state_dim=policy.config.max_state_dim,
        device=device,
    )

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    embodiment_id = torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=torch.long,
        device=device,
    )

    # Keep the raw image pixels as a one-stream list so this mirrors the PI0.5 script.
    images = [tokenized_data["pixel_values"].to(
        device=device,
        dtype=torch.float16,
    )]
    pixel_values = images[0]
    # groot specifc inputs ------

    # Load the custom TensorRT plugin library before compiling plugin-backed modules.
    register_plugin_op()
    from trt import plugin_converter as _plugin_converter  # noqa: F401,E402

    load_plugin()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    # Wrap the original eager image encoder before patching attention; this is the true vision reference.
    eager_model = GROOTVisualEmbed(model).eval().to(device=device, dtype=torch.float16)
    with torch.no_grad():
        eager_image_embs = eager_model(pixel_values)

    # inner SigLIP transformer to infer batch size and seq_len
    vision_model = model.backbone.eagle_model.vision_model.vision_model
    batch_size, seq_len = infer_siglip_seq_len(vision_model, pixel_values)
    
    # Temporarily swap eager SigLIP attention for the plugin-friendly implementation.
    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP"
    )

    # The plugin op has a dummy eager implementation, so this patched module is only meaningful for TRT export.
    # Compile the patched vision path.
    trt_vision_model = compile_trt_module(
        eager_model,
        (pixel_values,),
        TRT_SETTINGS,
    )

    restore_attention(patched)

    # Compare TRT against restored original eager and keep the tensor for context compilation.
    compare_vision(model, images, trt_vision_model, eager_model)
    with torch.no_grad():
        trt_image_embs = trt_vision_model(pixel_values)
    tensor_error_metrics("groot TRT vs original vision embeddings", trt_image_embs, eager_image_embs)

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")
        
    # -------------------------
    # Original eager context creation
    # -------------------------
    eager_context_embs, _, _, _ = build_groot_context_inputs(
        model,
        eager_image_embs,
        input_ids,
        attention_mask,
    )

    # -------------------------
    # TensorRT/plugin language context creation
    # -------------------------

    # Build the actual LM input embeddings for the TRT vision path.
    trt_language_inputs = build_groot_language_inputs(
        model,
        trt_image_embs,
        input_ids,
        attention_mask,
    )

    # Build the same LM input embeddings with eager vision so we can isolate LM-plugin drift.
    eager_language_inputs = build_groot_language_inputs(
        model,
        eager_image_embs,
        input_ids,
        attention_mask,
    )

    # Compile the Eagle LM with PluginAttention.
    language_model = copy.deepcopy(model.backbone.eagle_model.language_model).to(
        device=device,
        dtype=torch.float16,
    ).eval()
    decoder = getattr(language_model, "model", language_model)
    hidden_lm_wrapper = make_plugin_lm_hidden_wrapper(
        decoder,
        language_model.config,
        max_seq_len=int(trt_language_inputs.inputs_embeds.shape[1]),
        device=device,
        position_ids=None,
        enable_bidirectional_prefill=0,
        return_prefix_kv=False,
        log_prefix="groot",
    )
    context_projection = GROOTContextProjectionWrapper(
        copy.deepcopy(model.backbone.eagle_linear).to(device=device, dtype=torch.float16).eval(),
        copy.deepcopy(model.action_head.vlln).to(device=device, dtype=torch.float16).eval(),
        copy.deepcopy(model.action_head.vl_self_attention).to(device=device, dtype=torch.float16).eval(),
    )
    plugin_language = GROOTLanguageContextWrapper(
        hidden_lm_wrapper,
        context_projection,
    ).eval()
    trt_language_model, trt_language_max_seq_len = compile_language_trt_with_plugin(
        plugin_language,
        trt_language_inputs.inputs_embeds,
        num_layers=len(decoder.layers),
        num_key_value_heads=int(language_model.config.num_key_value_heads),
        head_dim=language_head_dim(language_model.config),
        device=device,
        settings=TRT_SETTINGS,
    )
    lm_head_dim = language_head_dim(language_model.config)

    # Run plugin LM/context with eager vision embeddings.
    eager_lm_inputs = eager_language_inputs.inputs_embeds.to(device=device, dtype=torch.float16)
    eager_kv_caches = [
        torch.zeros(
            int(eager_lm_inputs.shape[0]),
            2,  # key + value
            int(language_model.config.num_key_value_heads),
            trt_language_max_seq_len,
            language_head_dim(language_model.config),
            device=device,
            dtype=eager_lm_inputs.dtype,
        )
        for _ in range(len(decoder.layers))
    ]
    eager_ctx_len = torch.full(
        (eager_lm_inputs.shape[0],),
        eager_lm_inputs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    eager_rope = make_rope_rotary_cos_sin(
        language_model.config,
        trt_language_max_seq_len,
        device,
        language_model=language_model,
        position_ids=eager_language_inputs.position_ids,
    )
    eager_kvcache_start_index = make_prefill_kvcache_start_index(device)
    trt_context_from_eager_vision = trt_language_model(
        eager_lm_inputs,
        eager_rope,
        eager_ctx_len,
        eager_kvcache_start_index,
        eager_kv_caches,
    )

    compare_groot_context(
        eager_context_embs,
        trt_context_from_eager_vision,
        attention_mask,
        name="groot plugin context with eager vision",
    )

    compare_groot_context_token_types(
        model,
        eager_context_embs,
        trt_context_from_eager_vision,
        input_ids,
        attention_mask,
        name="groot plugin context with eager vision",
    )

    # Run plugin LM/context with TRT vision embeddings.
    trt_lm_inputs = trt_language_inputs.inputs_embeds.to(device=device, dtype=torch.float16)
    trt_kv_caches = [
        torch.zeros(
            int(trt_lm_inputs.shape[0]),
            2,  # key + value
            int(language_model.config.num_key_value_heads),
            trt_language_max_seq_len,
            language_head_dim(language_model.config),
            device=device,
            dtype=trt_lm_inputs.dtype,
        )
        for _ in range(len(decoder.layers))
    ]
    trt_ctx_len = torch.full(
        (trt_lm_inputs.shape[0],),
        trt_lm_inputs.shape[1],
        device=device,
        dtype=torch.int32,
    )
    trt_rope = make_rope_rotary_cos_sin(
        language_model.config,
        trt_language_max_seq_len,
        device,
        language_model=language_model,
        position_ids=trt_language_inputs.position_ids,
    )
    trt_kvcache_start_index = make_prefill_kvcache_start_index(device)
    trt_context_embs = trt_language_model(
        trt_lm_inputs,
        trt_rope,
        trt_ctx_len,
        trt_kvcache_start_index,
        trt_kv_caches,
    )

    compare_groot_context(
        eager_context_embs,
        trt_context_embs,
        attention_mask,
        name="groot plugin context full TRT",
    )

    compare_groot_context_token_types(
        model,
        eager_context_embs,
        trt_context_embs,
        input_ids,
        attention_mask,
        name="groot plugin context full TRT",
    )

    # -------------------------
    # Eager baseline before action compile/offload
    # -------------------------
    # Keep the three action contexts explicit so the metric blocks show exactly what drift is isolated.
    eager_action_context = eager_context_embs.to(torch.float16)
    action_compile_context = trt_context_from_eager_vision.to(torch.float16)
    full_trt_action_context = trt_context_embs.to(torch.float16)

    # The action module consumes context embeddings, noisy actions, timestep, robot state, and embodiment id.
    action_module = GrootStaticDiffusionStep(model.action_head).eval().to(
        device=device,
        dtype=torch.float16,
    )

    # Seed the diffusion noise so eager and TRT rollouts start from the same sample.
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # Sample the initial noisy action chunk in GROOT's action horizon/dimension.
    noise = torch.randn(
        eager_action_context.shape[0],
        model.action_head.config.action_horizon,
        model.action_head.config.action_dim,
        device=device,
        dtype=eager_action_context.dtype,
    )

    # Run the eager action module through the shared raw-tensor rollout loop.
    eager_action_rollout_context = ActionRolloutContext(
        noise=noise,
        device=device,
        context_embs=eager_action_context,
        state=state,
        embodiment_id=embodiment_id,
    )
    eager_actions = sample_actions_raw(
        action_module,
        eager_action_rollout_context,
        GROOTActionAdapter(model.action_head),
    )

    # -------------------------
    # Action engine
    # -------------------------
    print("compiling action")

    # Build representative action-step inputs using the plugin LM context with eager vision.
    sample_inputs = make_compile_inputs(
        action_module,
        action_compile_context,
        state.to(dtype=torch.float16),
        embodiment_id,
        device,
    )

    # Compile the static one-step denoising module.
    trt_action = compile_trt_module(
        action_module,
        sample_inputs,
        ACTION_TRT_SETTINGS,
    )

    # ACTION_TRT_SETTINGS may offload the source eager module to CPU during compile.
    # Move it back before using it as the eager reference for action metrics.
    action_module = action_module.to(device=device, dtype=torch.float16).eval()

    # -------------------------
    # Metrics
    # -------------------------
    print("direct action step metrics")
    # Compare a single denoising step with identical context, state, noise, and timestep.
    compare_groot_action_step(
        action_module,
        trt_action,
        action_compile_context,
        state,
        embodiment_id,
        noise,
        device,
    )

    print("full action metrics")
    # Roll the whole TRT path: TRT vision, plugin LM/context, and TRT action engine.
    compare_full_groot_to_eager_actions(
        model,
        trt_action,
        full_trt_action_context,
        state,
        embodiment_id,
        eager_actions,
        noise,
        device=device,
        name="full action metrics with full TRT context",
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())