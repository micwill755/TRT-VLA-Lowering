Overview
========

Torch-TRT pipelines provides export, inference, and benchmark orchestration for
vision-language-action (VLA) models compiled with `TensorRT Edge-LLM
<https://nvidia.github.io/TensorRT-Edge-LLM/latest/>`_.

A single CLI entry point (``app.py``) drives export, eager/TRT inference, and
backend parity checks through **profiles**, **pipelines**, and per-model **hooks**.
Serialized engines are loaded per stage during inference (no separate load pipeline).
