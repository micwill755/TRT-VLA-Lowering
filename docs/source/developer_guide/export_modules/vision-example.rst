GR00T Vision Export Example
===========================

This page walks through the **vision stage**: SigLIP tower + Eagle projector compiled
as ``visual/visual.engine`` for the Edge-LLM VitRunner.


Typical tensor shapes (GR00T N1.5, Libero sample)
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Tensor
     - Shape
     - Notes
   * - ``pixel_values`` (policy input)
     - ``[1, 3, 224, 224]`` NCHW fp16
     - LeRobot processor output on ``ctx.device``
   * - ``pixel_values`` (engine binding)
     - ``[1, 224, 224, 3]`` HWC fp16
     - ``nchw_to_hwc`` before TRT trace — VitRunner ABI
   * - ``visual_embeds`` (engine output)
     - ``[512, 2048]`` fp16
     - ``B * S_visual`` rows at LM hidden size (``1 * 512`` for default GR00T)
   * - ``image_embs`` (downstream)
     - ``[512, 2048]``
     - Same rows passed to language ``preprocess`` for image-token splice


Export module wiring
--------------------

The hook builds :class:`GridVisionExportModule` from the live Eagle backbone:

.. code-block:: python

   visual_module = GridVisionExportModule(
       vision_model=eagle.vision_model,
       projector=eagle.mlp1,
       sample_pixel_values=pixel_values,
       select_layer=eagle.select_layer,
       pixel_shuffle=eagle.use_pixel_shuffle,
       downsample_ratio=eagle.downsample_ratio,
       vision_kwargs={},
   ).eval().to(device=ctx.device, dtype=ctx.dtype)

Inside ``export``, the module is traced with **HWC** sample inputs and SigLIP
attention is patched for the ViT plugin:

.. code-block:: python

   images_hwc = nchw_to_hwc(pixel_values)  # engine binding layout

   patched = patch_vision_attention(vision.vision_model, batch_size, seq_len, name="SigLIP")
   try:
       engine_path = save_trt_engine_module(
           visual_module,
           (images_hwc,),
           ctx.engine_root / "visual",
           engine_file="visual.engine",
           input_names=["pixel_values"],
           output_names=["visual_embeds"],
           ...
       )
   finally:
       restore_attention(patched)


Putting it together — vision forward
------------------------------------

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       IN["pixel_values<br/>NCHW [B,3,H,W]<br/>or HWC at TRT boundary"]

       SIG["SigLIP vision_model<br/>patch embeddings + encoder"]
       SEL["select hidden state<br/>select_layer / last_hidden_state"]
       PS["optional pixel_shuffle<br/>downsample grid"]
       MLP["eagle.mlp1 projector<br/>ViT dim → LM dim"]
       OUT["visual_embeds<br/>[B*S, H_lm]"]

       IN --> SIG --> SEL --> PS --> MLP --> OUT

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333

       class SIG,MLP nvNode
       class IN,SEL,PS,OUT greyNode


Step-by-step
~~~~~~~~~~~~

**1. Layout normalization**

:class:`GridVisionExportModule` accepts NCHW or HWC pixels. Internally it calls
``hwc_to_nchw`` when the binding is HWC so HuggingFace SigLIP sees NCHW. Export
traces with HWC because the C++ VitRunner expects ``[B, H, W, C]``.

**2. Vision tower**

SigLIP runs with ``output_hidden_states=True`` when ``select_layer != -1``. GR00T
typically uses an intermediate layer plus optional **pixel shuffle** to reduce the
spatial grid before projection.

**3. Projector**

``eagle.mlp1`` maps ViT hidden size to language hidden size (2048 for N1.5).

**4. Flatten for Edge-LLM**

:meth:`ExportModule._finalize_output` reshapes ``[B, S, H]`` → ``[B*S, H]`` so
the C++ runtime can splice rows into image-placeholder token slots.


What the export hook returns
----------------------------

.. code-block:: python

   return {
       "engine_path": engine_path,
       "tensors": {"image_embs": image_embs},   # eager forward on NCHW pixels
       "metadata": {
           "image_token_id": ...,
           "seq_len": seq_len,                  # tokens per image after projector
           "vocab_size": ...,
       },
   }

``seq_len`` and ``image_token_id`` are written to ``visual/config.json`` and
consumed by the language stage and C++ embedding splice.


Parity tips
-----------

1. Compare **the same layout**: eager parity often uses NCHW policy pixels;
   serialized TRT uses HWC inside the wrapper — both should produce the same
   ``image_embs`` rows after the module normalizes layout.
2. **Plugin patching** must be active during compile only; ``restore_attention``
   after ``save_trt_engine_module``.
3. Vision parity tensor key in benchmark: ``vision:image_embs``.


*Files:* ``trt/modules/export/vision.py``, ``trt/executor/models/groot/export/vision.py``,
``trt/vision.py``, ``trt/plugin/plugin_utils.py`` (``patch_vision_attention``)
