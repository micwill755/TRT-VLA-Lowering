Overview
========

Torch-TRT pipelines provides export, load, inference, and benchmark orchestration for
vision-language-action (VLA) models compiled with `TensorRT Edge-LLM
<https://nvidia.github.io/TensorRT-Edge-LLM/latest/>`_.

A single CLI entry point (``app.py``) drives compile, inference, and benchmark flows
through **profiles**, **pipelines**, and per-model **hooks**.
