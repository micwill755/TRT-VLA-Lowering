import torch
import torch.nn as nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot import GrootPolicy
from lerobot.utils.constants import OBS_STATE

from trt.compile import compile_trt_module
from trt.diffusion import GrootStaticDiffusionStep
from trt.utils import load_policy, prepare_policy_inputs_groot
from trt.data import make_batch
from trt.vision import GROOTVisualEmbed
from trt.language import (
    compact_prefix_inputs,
    compile_pi05_lm_trt_with_plugin,
    pi05_plugin_lm_smoke_check,
    run_pi05_plugin_language,
    run_pi05_preprocessing,
)
from trt.measure import (
    compare_action_step,
    compare_full_vla_to_eager_actions,
    compare_language,
    compare_vision,
)
from trt.attention import ViTPluginAttention
from trt.plugin_utils import (
    load_plugin,
    patch_vision_attention,  
    restore_attention,
    infer_siglip_seq_len,
)

@torch.no_grad()
def compare_groot_language(core, language_runner, input_ids, attention_mask, vit_embs):
    eager = GROOTLanguageEmbed(core).eval().to(vit_embs.device)(
        input_ids,
        attention_mask,
        vit_embs,
    )
    trt = language_runner(input_ids, attention_mask, vit_embs)
    tensor_error_metrics("language/context", trt, eager)
    return eager, trt


@torch.no_grad()
def sample_actions_with_full_groot_trt(
    policy,
    batch,
    visual_runner,
    language_runner,
    action_runner,
    noise,
    device,
):
    core = policy._groot_model
    groot_inputs = filter_groot_inputs(batch)
    backbone_inputs, action_inputs = core.prepare_input(groot_inputs)

    pixel_values = backbone_inputs["eagle_pixel_values"].to(device)
    input_ids = backbone_inputs["eagle_input_ids"].to(device)
    attention_mask = backbone_inputs["eagle_attention_mask"].to(device)
    state = action_inputs["state"].to(device)
    embodiment_id = action_inputs["embodiment_id"].to(device)

    vit_embs = visual_runner(pixel_values)
    vl_embs = language_runner(input_ids, attention_mask, vit_embs)

    return sample_actions_with_runner(
        policy,
        action_runner,
        vl_embs,
        state,
        embodiment_id,
        noise,
    )


@torch.no_grad()
def compare_full_groot_to_eager_actions(
    policy,
    batch,
    visual_runner,
    language_runner,
    action_runner,
    eager_actions,
    noise,
    device,
):
    trt_actions = sample_actions_with_full_groot_trt(
        policy,
        batch,
        visual_runner,
        language_runner,
        action_runner,
        noise=noise.clone(),
        device=device,
    )

    action_dim = policy.config.output_features["action"].shape[0]
    eager_actions = eager_actions[:, :, :action_dim]
    trt_actions = trt_actions[:, :, :action_dim]

    diff = (eager_actions.float() - trt_actions.float()).abs()
    ade = compute_action_chunk_ade(trt_actions, eager_actions)

    print("action xyz ADE:", ade)
    print("action xyz minADE:", ade)
    print("Eager actions:", eager_actions.shape, eager_actions.dtype)
    print("TRT actions:", trt_actions.shape, trt_actions.dtype)
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())

MODEL_ID = "nvidia/GR00T-N1.5-3B"
DATASET_ID = "lerobot/libero"
FUTURE_STEPS = 5
SEED = 42

TRT_SETTINGS = {
    "disable_tf32": True,
    "use_explicit_typing": True,
    "use_fp32_acc": True,
    "truncate_double": True,
    "min_block_size": 1,
    "use_python_runtime": True,
    "decompose_attention": True,
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
def sample_actions_eager(policy, vl_embs, state, embodiment_id, noise):
    action_head = policy._groot_model.action_head
    num_steps = action_head.num_inference_timesteps
    dt = 1.0 / num_steps

    actions = noise.clone()

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)

        timestep = torch.full(
            (actions.shape[0],),
            t_discretized,
            device=actions.device,
            dtype=torch.long,
        )

        action_features = action_head.action_encoder(
            actions,
            timestep,
            embodiment_id,
        )

        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device,
            )
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        state_features = action_head.state_encoder(state, embodiment_id)
        future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0],
            -1,
            -1,
        )

        sa_embs = torch.cat(
            (state_features, future_tokens, action_features),
            dim=1,
        )

        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timestep,
        )

        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity

    return actions


@torch.no_grad()
def sample_actions_with_runner(policy, action_runner, vl_embs, state, embodiment_id, noise):
    action_head = policy._groot_model.action_head
    num_steps = action_head.num_inference_timesteps
    dt = 1.0 / num_steps

    actions = noise.clone()

    runner_dtype = vl_embs.dtype
    state = state.to(dtype=runner_dtype)
    actions = actions.to(dtype=runner_dtype)

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)

        timestep = torch.full(
            (actions.shape[0],),
            t_discretized,
            device=actions.device,
            dtype=torch.long,
        )

        pred_velocity = action_runner(
            actions,
            timestep,
            vl_embs,
            state,
            embodiment_id,
        ).float()

        actions = actions + dt * pred_velocity.to(actions.dtype)

    return actions


def compute_action_chunk_ade(pred, target):
    pred_xyz = pred[..., :3].float()
    target_xyz = target[..., :3].float()
    step_l2 = torch.linalg.vector_norm(pred_xyz - target_xyz, dim=-1)
    return step_l2.mean().item()


@torch.no_grad()
def compare_to_eager(policy, action_runner, vl_embs, state, embodiment_id, seed=SEED):
    action_head = policy._groot_model.action_head
    device = vl_embs.device

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    noise = torch.randn(
        vl_embs.shape[0],
        action_head.config.action_horizon,
        action_head.config.action_dim,
        device=device,
        dtype=vl_embs.dtype,
    )

    eager_actions = sample_actions_eager(
        policy,
        vl_embs,
        state,
        embodiment_id,
        noise,
    )

    runner_actions = sample_actions_with_runner(
        policy,
        action_runner,
        vl_embs,
        state,
        embodiment_id,
        noise,
    )

    action_dim = policy.config.output_features["action"].shape[0]
    eager_actions = eager_actions[:, :, :action_dim]
    runner_actions = runner_actions[:, :, :action_dim]

    diff = (eager_actions.float() - runner_actions.float()).abs()
    ade = compute_action_chunk_ade(runner_actions, eager_actions)

    print("action xyz ADE:", ade)
    print("action xyz minADE:", ade)
    print("Eager actions:", eager_actions.shape, eager_actions.dtype)
    print("Runner actions:", runner_actions.shape, runner_actions.dtype)
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())


@torch.no_grad()
def prepare_policy_inputs(policy, batch):
    groot = policy._groot_model
    action_head = groot.action_head

    groot_inputs = filter_groot_inputs(batch)
    backbone_inputs, action_inputs = groot.prepare_input(groot_inputs)

    backbone_outputs = groot.backbone(backbone_inputs)
    backbone_outputs = action_head.process_backbone_output(backbone_outputs)

    vl_embs = backbone_outputs.backbone_features
    state = action_inputs.state
    embodiment_id = action_inputs.embodiment_id

    return vl_embs, state, embodiment_id

def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_policy(GrootPolicy, MODEL_ID, device, True).to(device).eval()
    batch = make_batch(policy, None, device, fill_missing=False)
    core = policy._groot_model.to(device).eval()

    backbone_inputs, action_inputs = prepare_policy_inputs_groot(policy, batch, device)
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
        
    # image embeddings are the image patch tokens after vision tower forward
    image_embs = []
    for image in images:
        if trt_vision_model is None:
            image_embs.append(core.backbone.eagle_model.extract_feature(image))
        else:
            image_embs.append(trt_vision_model(image))

    prefix_embs, prefix_pad_masks, prefix_attention_mask, prefix_position_ids = build_prefix_inputs(
        core,
        image_embs,
        img_masks,
        tokens,
        masks,
        trt_vision=trt_vision_model
    )

    trt_language = compile_trt_module(
        GROOTLanguageEmbed(core).eval().to(device),
        (input_ids, attention_mask, vit_embs),
        TRT_SETTINGS,
    )

    vl_embs = trt_language(input_ids, attention_mask, vit_embs)
    state = action_inputs["state"].to(device=device)
    embodiment_id = action_inputs["embodiment_id"].to(device=device)

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