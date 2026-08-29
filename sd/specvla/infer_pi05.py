"""Spec-VLA-style Pi0.5 inference: stock vision/language engines, relaxed-accept denoise.

No kinematic F, no retrieval bank. Draft ``k`` Euler steps, verify with +1 step, and
accept the draft if quantized action bins are within ``tau`` of the probe (Spec-VLA's
distance-sensitive rule, mapped onto Pi0.5's continuous chunk).

    python sd/specvla/infer_pi05.py --engine-dir /tmp/pi05_edge_llm
    python sd/heisd/infer_pi05.py --engine-dir /tmp/pi05_edge_llm   # same engines / frames
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_SPECVLA_DIR = Path(__file__).resolve().parent
_SD_DIR = _SPECVLA_DIR.parent
_TEST_ROOT = _SD_DIR.parent
for path in (_TEST_ROOT, _SD_DIR, _SPECVLA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from relax import accept_length, relaxed_accept, rmse


def specvla_sample(
    action_engine,
    noise: torch.Tensor,
    *,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    core,
    dtype: torch.dtype,
    draft_steps: int,
    full_steps: int,
    tau: int,
    n_bins: int,
    gripper_dim: int | None,
    baseline: bool,
) -> tuple[torch.Tensor, str, int, float, int]:
    """Returns actions, mode, action-engine steps, draft-vs-probe RMSE, accept length."""
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
        return actions, "full", full_steps, 0.0, 0

    draft = denoise_range(action_engine, noise, start_step=0, n_steps=draft_steps, **denoise_kw)
    probe = denoise_range(action_engine, draft, start_step=draft_steps, n_steps=1, **denoise_kw)
    error = rmse(draft, probe)
    accepted = accept_length(draft, probe, tau=tau, n_bins=n_bins)

    if relaxed_accept(draft, probe, tau=tau, n_bins=n_bins, gripper_dim=gripper_dim):
        return draft, "relaxed", draft_steps + 1, error, accepted

    remaining = full_steps - draft_steps - 1
    actions = denoise_range(
        action_engine,
        probe,
        start_step=draft_steps + 1,
        n_steps=remaining,
        **denoise_kw,
    )
    return actions, "full", full_steps, error, accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spec-VLA analogue for Pi0.5: relaxed-accept denoise over exported TRT engines"
    )
    parser.add_argument("--engine-dir", type=Path, default=Path("/tmp/pi05_edge_llm"))
    parser.add_argument("--model-id", type=str, default="lerobot/pi05_libero_base")
    parser.add_argument("--dataset", type=str, default="lerobot/libero")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-steps", type=int, default=2)
    parser.add_argument("--full-steps", type=int, default=None)
    parser.add_argument(
        "--relax-tau",
        type=int,
        default=1,
        help="Max quantized-bin distance for a dim to count as accepted (Spec-VLA tau).",
    )
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument(
        "--gripper-dim",
        type=str,
        default="none",
        help="Action dim that must match exactly, or 'none'.",
    )
    parser.add_argument("--baseline", action="store_true", help="Always run the full denoise loop.")
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device, dtype = setup_cuda()
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

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    print(
        f"{'frame':>5}  {'mode':<10}  {'tau':>4}  {'rmse':>8}  {'accept':>6}  {'steps':>5}  "
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
            return specvla_sample(
                action,
                noise,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                prefix_pad_mask=prefix_pad_mask,
                core=model,
                dtype=dtype,
                draft_steps=args.draft_steps,
                full_steps=full_steps,
                tau=args.relax_tau,
                n_bins=args.n_bins,
                gripper_dim=gripper_dim,
                baseline=args.baseline,
            )

        (actions, mode, steps, error, accepted), act_ms = cuda_ms(
            _action, warmup=args.warmup if offset == 0 else 0
        )
        del actions

        print(
            f"{frame_index:5d}  {mode:<10}  {args.relax_tau:4d}  {error:8.4f}  {accepted:6d}  {steps:5d}  "
            f"{vis_ms:8.2f}  {lm_ms:8.2f}  {act_ms:8.2f}"
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
