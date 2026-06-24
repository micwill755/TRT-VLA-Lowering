"""Helpers for TensorRT-Edge-LLM ``LLMInferenceRuntime`` smoke tests."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any

from PIL import Image

# One placeholder per camera; VitRunner::textPreprocess expands each to builder_config.seq_len.
GROOT_VITRUNNER_IMAGE_FORMAT = "<img><IMG_CONTEXT></img>"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)


def build_groot_vitrunner_chat_template(tokenizer) -> dict[str, Any]:
    """Build processed_chat_template.json for VitRunner (single image placeholder per image)."""
    im_start = "<|im_start|>"
    im_end = tokenizer.eos_token
    if not im_end:
        raise ValueError("Tokenizer eos_token is required to build the GROOT chat template.")

    system_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": "SYS"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TEXTONLY"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    with_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TEXTONLY"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    system_prefix = system_only.split("SYS", 1)[0]
    system_suffix = "SYS" + system_only.split("SYS", 1)[1]

    user_prefix = user_only.split("TEXTONLY", 1)[0]
    user_suffix = "TEXTONLY" + user_only.split("TEXTONLY", 1)[1]

    assistant_only = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "TEXTONLY"},
            {"role": "assistant", "content": "ASSIST"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    assistant_prefix = assistant_only[len(user_only) :].split("ASSIST", 1)[0]
    assistant_suffix = "ASSIST" + assistant_only.split("ASSIST", 1)[1]

    generation_prompt = with_gen[len(user_only) :]

    return {
        "model_path": "groot-vitrunner",
        "roles": {
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "assistant": {"prefix": assistant_prefix, "suffix": assistant_suffix},
        },
        "content_types": {
            "image": {"format": GROOT_VITRUNNER_IMAGE_FORMAT},
        },
        "generation_prompt": generation_prompt,
        "default_system_prompt": "You are a helpful assistant.",
    }


def save_tokenizer_for_edge_llm(
    tokenizer_assets_dir: str | pathlib.Path,
    language_engine_dir: str | pathlib.Path,
    *,
    chat_template: dict[str, Any] | None = None,
    tokenizer=None,
) -> None:
    """Write HF tokenizer assets and processed_chat_template.json into language/."""
    dst = pathlib.Path(language_engine_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if tokenizer is not None:
        tokenizer.save_pretrained(dst)
    else:
        src = pathlib.Path(tokenizer_assets_dir)
        copied = False
        for name in TOKENIZER_FILES:
            src_file = src / name
            if src_file.exists():
                shutil.copy2(src_file, dst / name)
                copied = True
        if not (dst / "tokenizer.json").exists():
            raise FileNotFoundError(
                f"tokenizer.json not found under {src}; pass tokenizer= to save_pretrained()."
            )
        if not copied:
            raise FileNotFoundError(f"No tokenizer files copied from {src}")

    if chat_template is None:
        if tokenizer is None:
            raise ValueError("chat_template or tokenizer is required.")
        chat_template = build_groot_vitrunner_chat_template(tokenizer)

    (dst / "processed_chat_template.json").write_text(
        json.dumps(chat_template, indent=2) + "\n"
    )


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
