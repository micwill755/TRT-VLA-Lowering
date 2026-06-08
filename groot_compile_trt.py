import torch
import torch.nn as nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot import GrootPolicy
from lerobot.utils.constants import OBS_STATE

from trt.compile import compile_trt_module
from trt.diffusion import GrootStaticDiffusionStep
from trt.utils import (
    load_policy, 
    build_prefix_inputs,
    compact_prefix_inputs,
    sample_actions_eager, 
    prepare_policy_inputs_groot, 
    make_suffix_position_and_mask
)
from trt.data import make_batch
from trt.vision import GROOTVisualEmbed
from trt.language import (
    compile_lm_trt_with_plugin,
    pi05_plugin_lm_smoke_check,
    run_prefix_plugin_language,
    run_prefix_language_eager
)
from trt.measure import (
    compare_action_step,
    compare_full_vla_to_eager_actions,
    compare_language,
    compare_vision,
    tensor_error_metrics,
)
from trt.attention import ViTPluginAttention
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

MODEL_ID = "nvidia/GR00T-N1.5-3B"
SEED = 42

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
def build_groot_context_inputs(core, vit_embs, input_ids, attention_mask):
    eagle = core.backbone.eagle_model

    input_embs = eagle.language_model.get_input_embeddings()(input_ids)

    bsz, seq_len, hidden = input_embs.shape
    flat_embs = input_embs.reshape(bsz * seq_len, hidden)
    flat_ids = input_ids.reshape(bsz * seq_len)

    image_token_index = getattr(eagle, "image_token_index", eagle.config.image_token_index)
    selected = flat_ids == image_token_index

    flat_embs[selected] = vit_embs.reshape(-1, hidden).to(flat_embs.dtype)[: selected.sum()]
    input_embs = flat_embs.reshape(bsz, seq_len, hidden)

    out = eagle.language_model(
        inputs_embeds=input_embs,
        attention_mask=attention_mask,
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

    context_pad_masks = attention_mask.to(dtype=torch.bool, device=context_embs.device)
    context_attention_mask = attention_mask
    context_position_ids = torch.cumsum(context_pad_masks, dim=1) - 1

    return context_embs, context_pad_masks, context_attention_mask, context_position_ids

def make_groot_context_masks(context_embs, attention_mask):
    context_pad_masks = attention_mask.to(device=context_embs.device, dtype=torch.bool)
    context_position_ids = torch.cumsum(context_pad_masks, dim=1) - 1

    return compact_prefix_inputs(
        context_embs,
        context_pad_masks,
        context_position_ids,
    )

class GROOTContextEmbed(nn.Module):
    def __init__(self, core, input_ids):
        super().__init__()
        self.eagle = core.backbone.eagle_model
        self.eagle_linear = core.backbone.eagle_linear
        self.select_layer = core.backbone.select_layer
        self.vlln = core.action_head.vlln
        self.vl_self_attention = core.action_head.vl_self_attention

        image_token_index = getattr(
            self.eagle,
            "image_token_index",
            self.eagle.config.image_token_index,
        )
        image_token_indices = (input_ids.reshape(-1) == image_token_index).nonzero(as_tuple=False).flatten()
        self.register_buffer("image_token_indices", image_token_indices.to(torch.long), persistent=False)

    def forward(self, input_ids, attention_mask, vit_embs):
        input_embs = self.eagle.language_model.get_input_embeddings()(input_ids)

        bsz, seq_len, hidden = input_embs.shape
        flat_embs = input_embs.reshape(bsz * seq_len, hidden)
        vit_flat = vit_embs.reshape(-1, hidden).to(device=flat_embs.device, dtype=flat_embs.dtype)
        vit_flat = vit_flat[: self.image_token_indices.shape[0]]
        flat_embs = flat_embs.index_copy(0, self.image_token_indices, vit_flat)
        input_embs = flat_embs.reshape(bsz, seq_len, hidden)

        out = self.eagle.language_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        context_embs = out.hidden_states[self.select_layer]
        context_embs = self.eagle_linear(context_embs)

        vlln_weight = getattr(self.vlln, "weight", None)
        if vlln_weight is not None:
            context_embs = context_embs.to(device=vlln_weight.device, dtype=vlln_weight.dtype)
        context_embs = self.vlln(context_embs)
        context_embs = self.vl_self_attention(context_embs)
        return context_embs
        
def compile_groot_context_trt(core, input_ids, attention_mask, vit_embs, device, settings):
    context_model = GROOTContextEmbed(core, input_ids).eval().to(device=device, dtype=torch.float16)

    trt_context_model = compile_trt_module(
        context_model,
        (input_ids, attention_mask, vit_embs),
        settings,
    )

    return trt_context_model, input_ids.shape[1]

@torch.no_grad()
def run_groot_context_trt(trt_context_model, input_ids, attention_mask, vit_embs):
    return trt_context_model(input_ids, attention_mask, vit_embs)

@torch.no_grad()
def groot_context_smoke_check(eager_context_embs, trt_context_embs, attention_mask):
    compact_eager, _, _, _ = make_groot_context_masks(eager_context_embs, attention_mask)
    compact_trt, _, _, _ = make_groot_context_masks(trt_context_embs, attention_mask)

    tensor_error_metrics("groot context smoke", compact_trt, compact_eager)

@torch.no_grad()
def compare_groot_context(eager_context_embs, trt_context_embs, attention_mask):
    compact_eager, _, _, _ = make_groot_context_masks(eager_context_embs, attention_mask)
    compact_trt, _, _, _ = make_groot_context_masks(trt_context_embs, attention_mask)

    tensor_error_metrics("groot context", compact_trt, compact_eager)

def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_policy(GrootPolicy, MODEL_ID, device, True).to(device).eval()
    batch = make_batch(policy, None, device, fill_missing=False)
    core = policy._groot_model.to(device).eval()

    backbone_inputs, action_inputs = prepare_policy_inputs_groot(policy, batch, device)
    input_ids = backbone_inputs["eagle_input_ids"].to(device)
    attention_mask = backbone_inputs["eagle_attention_mask"].to(device)
    state = action_inputs["state"].to(device)
    embodiment_id = action_inputs["embodiment_id"].to(device)

    # images here are raw pixels
    images = [backbone_inputs["eagle_pixel_values"].to(device=device, dtype=torch.float16)]
    pixel_values = images[0]

    load_plugin()

    # -------------------------
    # Vision engine
    # -------------------------
    print("compiling vision")

    # inner SigLIP transformer to infer batch size and seq_len
    vision_model = core.backbone.eagle_model.vision_model.vision_model
    batch_size, seq_len = infer_siglip_seq_len(vision_model, pixel_values)
    
    patched = patch_vision_attention(
        vision_model,
        batch_size=batch_size,
        seq_len=seq_len,
        name="SigLIP"
    )

    eager_model = GROOTVisualEmbed(core).eval().to(device=device, dtype=torch.float16)
    trt_vision_model = compile_trt_module(
        eager_model,
        (pixel_values,),
        TRT_SETTINGS,
    )

    restore_attention(patched)

    compare_vision(core, images, trt_vision_model, eager_model)

    # -------------------------
    # Language/context engine
    # -------------------------
    print("compiling language")
        
    # -------------------------
    # Original eager context creation
    # -------------------------
    eager_image_embs = core.backbone.eagle_model.extract_feature(pixel_values)

    eager_context_embs, eager_context_pad_masks, _, eager_context_position_ids = build_groot_context_inputs(
        core,
        eager_image_embs,
        input_ids,
        attention_mask,
    )

    compact_eager_context_embs, compact_eager_context_pad_masks, _, _ = compact_prefix_inputs(
        eager_context_embs,
        eager_context_pad_masks,
        eager_context_position_ids,
    )

    # -------------------------
    # TensorRT context creation
    # -------------------------
    trt_image_embs = trt_vision_model(pixel_values)

    # compile TRT context
    trt_context_model, trt_context_max_seq_len = compile_groot_context_trt(
        core,
        input_ids,
        attention_mask,
        trt_image_embs,
        device,
        TRT_SETTINGS,
    )

    # run TRT context
    trt_context_embs = run_groot_context_trt(
        trt_context_model,
        input_ids,
        attention_mask,
        trt_image_embs,
    )

    # smoke
    groot_context_smoke_check(
        eager_context_embs,
        trt_context_embs,
        attention_mask,
    )

    # compare language/context
    compare_groot_context(
        eager_context_embs,
        trt_context_embs,
        attention_mask,
    )


    '''# -------------------------
    # Action engine
    # -------------------------
    print("compiling action")
    action_module = build_action_step(policy, device)

    sample_inputs = make_compile_inputs(
        action_module,
        vl_embs,
        state,
        embodiment_id,
        device,
    )

    trt_action = compile_trt_module(
        action_module,
        sample_inputs,
        TRT_SETTINGS,
    )

    print("metrics")
    eager_vit_embs, _ = compare_groot_vision(core, pixel_values, trt_visual)

    eager_vl_embs, _ = compare_groot_language(
        core,
        trt_language,
        input_ids,
        attention_mask,
        eager_vit_embs,
    )

    action_head = core.action_head
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    noise = torch.randn(
        eager_vl_embs.shape[0],
        action_head.config.action_horizon,
        action_head.config.action_dim,
        device=device,
        dtype=eager_vl_embs.dtype,
    )

    eager_actions = sample_actions_eager(
        policy,
        eager_vl_embs,
        state,
        embodiment_id,
        noise,
    )

    print("full action metrics")
    compare_full_groot_to_eager_actions(
        policy,
        batch,
        trt_visual,
        trt_language,
        trt_action,
        eager_actions,
        noise,
        device=device,
    )'''

    return 0

if __name__ == "__main__":
    raise SystemExit(main())