Parity and Debugging
====================

Plugin export introduces a third runnable mode that is easy to misinterpret. Use the
right baseline for each question.


Three rungs (vision example)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 38 50

   * - Rung
     - What runs
     - Valid parity baseline?
   * - **A**
     - Unpatched eager PyTorch (standard SDPA)
     - **Yes** — reference for TRT
   * - **B**
     - Patched eager ``ViTPluginAttention`` (custom op stub)
     - **No** — attention output is zeros
   * - **C**
     - TensorRT engine compiled from patched graph
     - Compare against **A**


.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000'}}}%%
   graph TB
       A[A: unpatched eager] -->|meaningful| C[C: TRT engine]
       B[B: patched eager stub] -.->|not comparable| C
       B -.->|not comparable| A

       classDef good fill:#76B900,stroke:#5a8f00,color:#fff
       classDef bad fill:#f5f5f5,stroke:#999,color:#333

       class A,C good
       class B bad


The meaningful check is **A vs C**. Large errors between A and B or B and C are
expected because B does not implement attention in Python.


Typical metrics
---------------

Scripts such as ``test_vision.py`` report:

- ``mean_abs`` — mean absolute difference
- ``rel_l2`` — relative L2 error
- ``close%`` — fraction of elements within tolerance

Good vision parity after fp16 alignment is often ``rel_l2 ≈ 0.02`` (roughly 2%),
not bit-exact match.


In-memory vs serialized engines
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Path
     - Notes
   * - In-memory compile in test script
     - Compiles fresh from current patched module; good for isolating export issues
   * - Serialized ``.engine`` on disk
     - Must be re-exported after code or dtype fixes; stale engines show false regressions

If A vs in-memory C passes but A vs serialized C fails, suspect an old engine file
under ``engine_dir`` rather than converter logic.


Dtype alignment
---------------

Eager and TRT paths must agree on compute dtype (typically **fp16** for vision).
Mixed bf16 eager vs fp16 TRT produces large spurious gaps. Ensure export modules
and benchmark wrappers cast inputs/weights consistently before comparing.


Diagnostic checklist
--------------------

1. **Plugins loaded?** ``EDGE_LLM_PLUGIN_SO`` set; ``load_plugins_for_trt()`` called.
2. **Correct patch target?** Inner SigLIP ``vision_model`` for GR00T.
3. **Graph contains custom op?** Inspect exported graph or TRT layer names for
   ``ViTAttentionPlugin`` / ``AttentionPlugin``.
4. **Fresh engine?** Re-run export after changing ``attention.py``, converters, or dtypes.
5. **Comparing A vs C?** Do not treat patched eager as ground truth.


Torch-TensorRT logging
----------------------

Use ``torch_tensorrt.logging.set_level(logging.ERROR)`` (API name varies by build)
to quiet verbose converter registration. Import-order spam does not indicate a
functional problem.


Further reading
---------------

- :doc:`overview` — end-to-end plugin flow
- :doc:`custom-ops` — why eager stubs return zeros
- :doc:`architecture` — compile-shim layers
- ``test_vision.py`` — three-rung parity harness reference implementation
