Overview
========

Torch-TRT pipelines provides export, inference, and benchmark orchestration for
vision-language-action (VLA) models compiled with `TensorRT Edge-LLM
<https://nvidia.github.io/TensorRT-Edge-LLM/latest/>`_.

A single CLI entry point (``app.py``) drives export, eager/TRT inference, and
backend parity checks through **profiles**, **pipelines**, and per-model **hooks**.
Serialized engines are loaded per stage during inference (no separate load pipeline).

Supported models
----------------

See :doc:`supported-models` for the full list. Quick reference:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Model
     - CLI
     - Entry point
   * - GR00T
     - ``--model gr00t``
     - ``app.py`` (see :doc:`../../examples/gr00t`)
   * - Pi0.5
     - ``--model pi05``
     - ``app.py`` (see :doc:`../../examples/pi05`)
   * - SmolVLA
     - ``--model smolvla``
     - ``app.py`` (see :doc:`../../examples/smolvla`)
   * - MolmoAct2
     - ``--model molmo2``
     - ``test_vla_molmo2_e2e.py`` (see :doc:`../../examples/molmo2`)
   * - Alpamayo
     - (not in ``app.py`` yet)
     - ``test_vla_alpamayo_e2e.py`` (see :doc:`../../examples/alpamayo`)
