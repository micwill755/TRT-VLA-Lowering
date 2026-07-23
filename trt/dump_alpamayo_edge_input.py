# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump Physical AI AV clip data for TensorRT-Edge-LLM ``action_inference``.

Loads the same clip used by ``vla/test_vla_alpamayo_e2e.py``, writes:

- ``frames/frame_XX.png`` — 16 PNGs (4 cameras x 4 timesteps)
- ``input_action.json`` — Edge-compatible request JSON (with request ``id``)
- ``gt.json`` — real minADE ground truth for ``compute_minade.py``

Requires the Alpamayo Python 3.12 environment (``physical_ai_av`` + ``alpamayo_r1``).

Example:
    cd /path/to/Test
    PYTHONPATH=../alpamayo/src:. python trt/dump_alpamayo_edge_input.py \\
        --output-dir $HOME/tensorrt-edgellm-workspace/alpamayo_sample

For the VLA pytest harness (6 samples / clip), also pass::

    --num-traj-samples 6

Note: Edge ``action_inference`` uses one global ``--noiseSeed``, so duplicated
requests with a single seed are identical trajectories. For true minADE6, run
inference once per seed and merge responses that share the same request ``id``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000


def _save_chw_uint8_png(path: Path, chw) -> None:
    """Save a CHW uint8 tensor/array as an RGB PNG."""
    import numpy as np

    arr = np.asarray(chw)
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        hwc = arr
    if hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to write PNGs. Install pillow in the Alpamayo env."
        ) from exc

    Image.fromarray(hwc).save(path)


def _build_gt_entry(data: dict[str, Any], clip_id: str) -> dict[str, Any]:
    """Build one ``gt.json`` entry matching ``compute_minade.py`` / Alpamayo eval."""
    # Same tensors Alpamayo eval uses: future XY in t0-local frame + history pose.
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].detach().cpu().float()
    ego_history_xyz = data["ego_history_xyz"][0, 0].detach().cpu().float()
    ego_history_rot = data["ego_history_rot"][0, 0].detach().cpu().float()

    if tuple(gt_xy.shape) != (64, 2):
        raise ValueError(f"Expected gt_xy shape (64, 2), got {tuple(gt_xy.shape)}")
    if ego_history_xyz.ndim != 2 or ego_history_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected ego_history_xyz (T, 3), got {tuple(ego_history_xyz.shape)}"
        )
    if tuple(ego_history_rot.shape[-2:]) != (3, 3):
        raise ValueError(
            f"Expected ego_history_rot (..., 3, 3), got {tuple(ego_history_rot.shape)}"
        )

    return {
        clip_id: {
            "gt_xy": gt_xy.tolist(),
            "ego_history_xyz": ego_history_xyz.tolist(),
            "ego_history_rot": ego_history_rot.tolist(),
        }
    }


def dump_edge_input(
    output_dir: Path,
    *,
    clip_id: str = DEFAULT_CLIP_ID,
    t0_us: int = DEFAULT_T0_US,
    num_traj_samples: int = 1,
) -> tuple[Path, Path]:
    try:
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import alpamayo_r1 / physical_ai_av. "
            "Run this with the Alpamayo Python 3.12 environment and "
            "PYTHONPATH including alpamayo/src."
        ) from exc

    if num_traj_samples < 1:
        raise ValueError("--num-traj-samples must be >= 1")

    output_dir = output_dir.resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clip_id={clip_id} t0_us={t0_us} ...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    # (N_cameras, num_frames, 3, H, W) — same flatten order as test_vla_alpamayo_e2e.py
    image_frames = data["image_frames"].flatten(0, 1)
    trajectory = data["ego_history_xyz"][0, 0].detach().cpu().tolist()

    image_paths: list[str] = []
    for index, frame in enumerate(image_frames):
        path = frames_dir / f"frame_{index:02d}.png"
        _save_chw_uint8_png(path, frame.detach().cpu())
        image_paths.append(str(path))
        print(f"  wrote {path}")

    content = [{"type": "image", "image": p} for p in image_paths]
    content.append({"type": "trajectory", "trajectory": trajectory})
    content.append(
        {
            "type": "text",
            "text": (
                "output the chain-of-thought reasoning of the driving process, "
                "then output the future trajectory."
            ),
        }
    )

    request = {
        # compute_minade.py keys responses by request["id"] against gt.json.
        "id": clip_id,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a driving assistant that generates "
                            "safe and accurate actions."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ],
    }
    requests = [copy.deepcopy(request) for _ in range(num_traj_samples)]

    payload = {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": 128,
        "requests": requests,
        "meta": {
            "clip_id": clip_id,
            "t0_us": t0_us,
            "num_images": len(image_paths),
            "num_trajectory_points": len(trajectory),
            "num_traj_samples": num_traj_samples,
        },
    }

    input_json = output_dir / "input_action.json"
    input_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {input_json} ({num_traj_samples} request(s), id={clip_id})")

    gt = _build_gt_entry(data, clip_id)
    gt_json = output_dir / "gt.json"
    gt_json.write_text(json.dumps(gt) + "\n")
    hist_t = len(gt[clip_id]["ego_history_xyz"])
    print(
        f"Wrote {gt_json} "
        f"(gt_xy=64x2, ego_history_xyz={hist_t}x3, ego_history_rot={hist_t}x3x3)"
    )
    return input_json, gt_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dump Alpamayo Physical AI AV frames, trajectory, and real minADE "
            "gt.json for Edge-LLM action_inference."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "tensorrt-edgellm-workspace" / "alpamayo_sample",
        help="Directory for frames/, input_action.json, and gt.json",
    )
    parser.add_argument(
        "--clip-id",
        default=DEFAULT_CLIP_ID,
        help="Physical AI AV clip id (default matches test_vla_alpamayo_e2e.py)",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=DEFAULT_T0_US,
        help="Sample timestamp in microseconds (default matches e2e script)",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        help=(
            "Duplicate the request this many times (same id) so compute_minade "
            "default N=6 can score. Identical under a single --noiseSeed."
        ),
    )
    args = parser.parse_args(argv)

    try:
        dump_edge_input(
            args.output_dir,
            clip_id=args.clip_id,
            t0_us=args.t0_us,
            num_traj_samples=args.num_traj_samples,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
