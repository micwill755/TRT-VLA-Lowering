Export Modules Overview
=======================

Each GR00T engine stage compiles a small **export module** — a ``nn.Module`` whose
``forward`` matches the TensorRT / Edge-LLM binding contract. Model-specific
**hooks** under ``trt/executor/models/groot/export/`` build that module, patch
attention for plugins, call ``save_trt_engine_module``, and pass representative
tensors to the next stage.

The reusable wrappers live in ``trt/modules/export/``. Stage hooks only wire
subgraphs from ``ctx.model`` and handle I/O layout (NCHW vs HWC, flat KV bindings,
dummy downstream tensors).


Stage map
---------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph LR
       V[vision<br/>GridVisionExportModule]
       L[language<br/>CausalLMExportModule]
       AC[action_context<br/>ContextProjectionExportModule]
       A[action<br/>StaticActionVelocityStepExportModule]

       V --> L --> AC --> A

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       class V,L,AC,A nvNode

.. list-table::
   :header-rows: 1
   :widths: 18 28 22 32

   * - Stage
     - Export module
     - Engine file
     - Primary I/O
   * - Vision
     - :class:`GridVisionExportModule`
     - ``visual/visual.engine``
     - ``pixel_values`` HWC → ``visual_embeds`` flat
   * - Language
     - :class:`CausalLMExportModule`
     - ``language/language.engine``
     - flat prefill bindings → ``lm_hidden_states`` + prefix KV
   * - Action context
     - :class:`ContextProjectionExportModule`
     - ``action_context/context.engine``
     - ``lm_hidden_states`` → ``vl_embs``
   * - Action
     - :class:`StaticActionVelocityStepExportModule`
     - ``action/action.engine``
     - one denoising step → ``velocity``


Common export hook pattern
--------------------------

Every stage follows the same runner contract:

.. code-block:: text

   preprocess(ctx, merged_inputs)
       → build export module + trace tensors
   export(ctx, prepared)
       → patch attention → save_trt_engine_module → return engine_path + tensors
   save_artifacts?   (language only)
   postprocess(ctx, result)

**Upstream merge.** The pipeline passes ``{**pipeline_inputs, **upstream_stage}``
into each stage. Downstream ``preprocess`` reads upstream ``tensors`` (for example
``inputs["tensors"]["image_embs"]``).

**Dummy downstream tensors.** After compile, export does not always re-run the TRT
engine. Language and action_context return zero-filled tensors with the correct
shape/dtype so the next stage can trace without a full eager forward through the
just-compiled engine.


Related pages
-------------

- :doc:`../diffusion/overview` — flow-matching / diffusion step design.
- :doc:`vision-example` — SigLIP grid vision + projector.
- :doc:`language-example` — Edge-LLM causal LM prefill + vision splice.
- :doc:`action-context-example` — LM hidden → action-context embeddings.
- :doc:`../diffusion/groot-example` — one GR00T denoising step.
- :doc:`../diffusion/action-rollout` — multi-step loop outside the action engine.

*Files:* ``trt/modules/export/``, ``trt/executor/models/groot/export/``,
``trt/compile.py``, ``trt/runner/export.py``
