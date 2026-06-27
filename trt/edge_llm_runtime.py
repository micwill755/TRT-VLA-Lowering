"""Helpers for TensorRT-Edge-LLM ``LLMInferenceRuntime`` smoke tests."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any

from PIL import Image

def _save_uint8_image(image: Image.Image, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path)


def write_llm_runtime_smoke_case(
    engine_root: str | pathlib.Path,
    *,
    task_text: str,
    images: list[Image.Image],
    max_generate_length: int = 8,
) -> pathlib.Path:
    """Write llm_inference JSON + PNGs for a VitRunner-compatible multimodal request."""
    engine_root = pathlib.Path(engine_root)
    smoke_dir = engine_root / "runtime_smoke"
    image_paths: list[pathlib.Path] = []

    for idx, image in enumerate(images):
        image_path = smoke_dir / f"camera_{idx}.png"
        _save_uint8_image(image, image_path)
        image_paths.append(image_path)

    payload = {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": int(max_generate_length),
        "requests": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *[
                                {
                                    "type": "image",
                                    "image": str(path.resolve()),
                                }
                                for path in image_paths
                            ],
                            {"type": "text", "text": task_text},
                        ],
                    }
                ],
            }
        ],
    }

    input_path = smoke_dir / "input.json"
    input_path.write_text(json.dumps(payload, indent=2) + "\n")
    return input_path


def run_llm_inference_runtime_smoke(
    *,
    engine_root: str | pathlib.Path,
    input_file: str | pathlib.Path,
    llm_inference_bin: str | pathlib.Path,
    output_file: str | pathlib.Path | None = None,
    max_generate_length: int | None = None,
    dump_output: bool = True,
    timeout_s: int = 600,
) -> subprocess.CompletedProcess[str]:
    """Run examples/llm/llm_inference against exported GROOT engines."""
    engine_root = pathlib.Path(engine_root)
    language_dir = engine_root / "language"
    input_file = pathlib.Path(input_file)
    llm_inference_bin = pathlib.Path(llm_inference_bin)

    if output_file is None:
        output_file = input_file.parent / "output.json"
    output_file = pathlib.Path(output_file)

    if not llm_inference_bin.exists():
        raise FileNotFoundError(f"llm_inference binary not found: {llm_inference_bin}")
    if not (language_dir / "language.engine").exists():
        raise FileNotFoundError(f"Missing language engine under {language_dir}")
    if not (engine_root / "visual" / "visual.engine").exists():
        raise FileNotFoundError(f"Missing visual engine under {engine_root / 'visual'}")

    env = os.environ.copy()

    plugin_path = os.environ.get("EDGELLM_PLUGIN_PATH") or os.environ.get("EDGE_LLM_PLUGIN_SO")
    if plugin_path:
        env["EDGELLM_PLUGIN_PATH"] = plugin_path

    trt_lib = os.environ.get("TRT_PACKAGE_DIR")
    if trt_lib:
        env["LD_LIBRARY_PATH"] = f"{env.get('LD_LIBRARY_PATH', '')}:{trt_lib}/lib"

    cmd = [
        str(llm_inference_bin),
        f"--engineDir={language_dir}",
        f"--multimodalEngineDir={engine_root}",
        f"--inputFile={input_file}",
        f"--outputFile={output_file}",
    ]
    if dump_output:
        cmd.append("--dumpOutput")
    if max_generate_length is not None:
        cmd.append(f"--maxGenerateLength={int(max_generate_length)}")

    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
    )
