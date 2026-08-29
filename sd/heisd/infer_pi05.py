"""HeiSD-style Pi0.5 inference: stock vision/language engines, gated action denoise.

Vision + language prefill are unchanged. The action Euler loop may skip, early-exit,
or run the full step count based on the last chunk's kinematic score F and an
in-memory retrieval bank.

    python sd/heisd/infer_pi05.py --engine-dir /tmp/pi05_edge_llm
    python sd/specvla/infer_pi05.py --engine-dir /tmp/pi05_edge_llm   # same engines / frames
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_HEISD_DIR = Path(__file__).resolve().parent
_SD_DIR = _HEISD_DIR.parent
_TEST_ROOT = _SD_DIR.parent
for path in (_TEST_ROOT, _SD_DIR, _HEISD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kinematics import fused_metric
from retrieve import ActionBank
from pi05_runtime import (
    cuda_ms,
    denoise_range,
    load_pi05_engines,
    load_policy,
    pad_prefix_kv,
    prepare_frame,
    run_language,
    setup_cuda,
)


def accept(draft: torch.Tensor, verified: torch.Tensor, ade_thresh: float, gripper_dim: int | None) -> bool:
    delta = draft.float() - verified.float()
    if gripper_dim is None:
        path = float(delta.pow(2).mean().sqrt())
        return path < ade_thresh
    grip_ok = torch.equal(draft[..., gripper_dim].round(), verified[..., gripper_dim].round())
    path_delta = torch.cat([delta[..., :gripper_dim], delta[..., gripper_dim + 1 :]], dim=-1)
    path = float(path_delta.pow(2).mean().sqrt())
    return bool(grip_ok) and path < ade_thresh


def heisd_sample(
    action_engine,
    noise: torch.Tensor,
    *,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    core,
    lm_hidden: torch.Tensor,
    last_chunk: torch.Tensor | None,
    bank: ActionBank,
    dtype: torch.dtype,
    f_thresh: float,
    skip_sim: float,
    ade_thresh: float,
    draft_steps: int,
    full_steps: int,
    alpha: float,
    d_scale: float,
    xyz_dims: tuple[int, ...],
    gripper_dim: int | None,
    baseline: bool,
) -> tuple[torch.Tensor, str, int, float, float]:
    """Returns actions, mode, action-engine steps, F, retrieval similarity."""
    score_f = 0.0
    if last_chunk is not None:
        score_f = fused_metric(last_chunk[0], xyz_dims=xyz_dims, alpha=alpha, d_scale=d_scale)

    denoise_kw = dict(
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        prefix_pad_mask=prefix_pad_mask,
        core=core,
        full_steps=full_steps,
        dtype=dtype,
    )

    if baseline:
        actions = denoise_range(action_engine, noise, start_step=0, n_steps=full_steps, **denoise_kw)
        return actions, "full", full_steps, score_f, 0.0

    retrieved, sim = bank.query(lm_hidden)

    if score_f >= f_thresh and retrieved is not None and sim >= skip_sim:
        chunk = retrieved.to(device=noise.device, dtype=dtype)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        return chunk, "skip", 0, score_f, sim

    draft = denoise_range(action_engine, noise, start_step=0, n_steps=draft_steps, **denoise_kw)

    if score_f >= f_thresh and retrieved is not None:
        retrieved_dev = retrieved.to(device=noise.device, dtype=dtype)
        if retrieved_dev.ndim == 2:
            retrieved_dev = retrieved_dev.unsqueeze(0)
        if accept(draft, retrieved_dev, ade_thresh, gripper_dim):
            return retrieved_dev, "retrieve+verify", draft_steps, score_f, sim

    probe = denoise_range(action_engine, draft, start_step=draft_steps, n_steps=1, **denoise_kw)
    if accept(draft, probe, ade_thresh, gripper_dim):
        return draft, "early-exit", draft_steps + 1, score_f, sim

    remaining = full_steps - draft_steps - 1
    actions = denoise_range(
        action_engine,
        probe,
        start_step=draft_steps + 1,
        n_steps=remaining,
        **denoise_kw,
    )
    return actions, "full", full_steps, score_f, sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HeiSD Pi0.5 inference over exported TRT engines")
    parser.add_argument("--engine-dir", type=Path, default=Path("/tmp/pi05_edge_llm"))
    parser.add_argument("--model-id", type=str, default="lerobot/pi05_libero_base")
    parser.add_argument("--dataset", type=str, default="lerobot/libero")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f-thresh", type=float, default=0.5)
    parser.add_argument("--skip-sim", type=float, default=0.95)
    parser.add_argument("--ade-thresh", type=float, default=0.05)
    parser.add_argument("--draft-steps", type=int, default=2)
    parser.add_argument("--full-steps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--d-scale", type=float, default=1.0)
    parser.add_argument("--xyz-dims", type=str, default="0,1,2")
    parser.add_argument(
        "--gripper-dim",
        type=str,
        default="none",
        help="Action dim that must match exactly, or 'none' for ADE-only.",
    )
    parser.add_argument("--baseline", action="store_true", help="Always run the full denoise loop.")
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device, dtype = setup_cuda()
    xyz_dims = tuple(int(x) for x in args.xyz_dims.split(",") if x.strip() != "")
    gripper_dim = None if args.gripper_dim.lower() == "none" else int(args.gripper_dim)

    engine_dir = args.engine_dir
    print(f"Loading engines from {engine_dir}")
    vision, language, action = load_pi05_engines(engine_dir)

    print(f"Loading policy {args.model_id} on CPU (prefix embed + suffix masks)")
    policy, model, pre_processor = load_policy(args.model_id, engine_dir)
    cfg = policy.config
    full_steps = int(args.full_steps or cfg.num_inference_steps)
    chunk_size = int(cfg.chunk_size)
    action_dim = int(cfg.max_action_dim)

    bank = ActionBank()
    last_chunk = None
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    print(
        f"{'frame':>5}  {'mode':<16}  {'F':>6}  {'sim':>6}  {'steps':>5}  "
        f"{'vis_ms':>8}  {'lm_ms':>8}  {'act_ms':>8}"
    )

    for offset in range(args.num_frames):
        frame_index = args.frame_index + offset
        images, img_masks, tokens, masks, pixel_values = prepare_frame(
            policy,
            pre_processor,
            args.dataset,
            args.episode_index,
            frame_index,
            device,
            dtype,
        )

        def _vision():
            return vision(pixel_values)

        image_embs, vis_ms = cuda_ms(_vision, warmup=args.warmup if offset == 0 else 0)

        def _language():
            return run_language(
                language,
                model,
                images,
                img_masks,
                tokens,
                masks,
                image_embs,
                device,
                dtype,
            )

        lang_out, lm_ms = cuda_ms(_language, warmup=args.warmup if offset == 0 else 0)
        _, lm_hidden, prefix_k, prefix_v, prefix_pad_mask = lang_out
        prefix_seq_len = int(action.engine.config.get("prefix_seq_len", prefix_k.shape[-2]))
        prefix_k, prefix_v, prefix_pad_mask = pad_prefix_kv(
            prefix_k, prefix_v, prefix_pad_mask, prefix_seq_len
        )

        noise = torch.randn(
            1,
            chunk_size,
            action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        def _action():
            return heisd_sample(
                action,
                noise,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                prefix_pad_mask=prefix_pad_mask,
                core=model,
                lm_hidden=lm_hidden,
                last_chunk=last_chunk,
                bank=bank,
                dtype=dtype,
                f_thresh=args.f_thresh,
                skip_sim=args.skip_sim,
                ade_thresh=args.ade_thresh,
                draft_steps=args.draft_steps,
                full_steps=full_steps,
                alpha=args.alpha,
                d_scale=args.d_scale,
                xyz_dims=xyz_dims,
                gripper_dim=gripper_dim,
                baseline=args.baseline,
            )

        (actions, mode, steps, score_f, sim), act_ms = cuda_ms(
            _action, warmup=args.warmup if offset == 0 else 0
        )
        bank.add(lm_hidden, actions)
        last_chunk = actions.detach()

        print(
            f"{frame_index:5d}  {mode:<16}  {score_f:6.3f}  {sim:6.3f}  {steps:5d}  "
            f"{vis_ms:8.2f}  {lm_ms:8.2f}  {act_ms:8.2f}"
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
