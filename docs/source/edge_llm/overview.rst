Edge LLM Runtime Overview
=========================

The export pipelines in this project do not run the model at deployment time —
they produce TensorRT engines plus sidecar files that the **C++
``LLMInferenceRuntime``** (in ``TensorRT-Edge-LLM``) loads and drives. This
section explains how the engines you export sit on top of that runtime, and how
the export ``config.json`` files are the contract the runtime uses to allocate
GPU memory and bind tensors to engine inputs and outputs.

The runtime owns the runners
----------------------------

``LLMInferenceRuntime`` is constructed with two directories:

.. code-block:: cpp

   rt::LLMInferenceRuntime(engineDir, multimodalEngineDir, loraWeightsMap, stream);

- ``engineDir`` — the **language** engine directory. The runtime scans it for the
  first ``*.engine``, reads ``config.json`` next to it, loads
  ``embedding.safetensors``, and loads the tokenizer with ``loadFromHF(engineDir)``.
- ``multimodalEngineDir`` — the root that holds the other engines in fixed
  subdirectories: ``visual/``, ``audio/``, ``action_context/``, and ``action/``.

For the VLA models here, ``engineDir`` and ``multimodalEngineDir`` are usually the
**same** exported tree (e.g. ``/tmp/groot_edge_llm``): the language engine lives
in ``language/`` and the runtime is pointed at ``language/`` for ``engineDir`` and
the root for ``multimodalEngineDir``.

.. code-block:: bash

   llm_inference \
     --engineDir=/tmp/groot_edge_llm/language \
     --multimodalEngineDir=/tmp/groot_edge_llm \
     --inputFile=requests.json \
     --outputFile=responses.json

From those two paths the runtime builds up to five runners:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Runner
     - Loaded from
     - Present when
   * - ``LLMEngineRunner``
     - ``engineDir/*.engine`` + ``config.json``
     - Always (the backbone).
   * - ``VisionRunner`` (VitRunner)
     - ``multimodalEngineDir/visual/``
     - Request has images and a visual engine exists.
   * - ``AudioRunner``
     - ``multimodalEngineDir/audio/``
     - Request has audio and an audio engine exists.
   * - ``ActionContextRunner``
     - ``multimodalEngineDir/action_context/``
     - Only when an ``action_context`` engine was exported (GR00T).
   * - ``ActionRunner``
     - ``multimodalEngineDir/action/``
     - Whenever an ``action`` engine exists.

.. mermaid::

   %%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%
   graph TB
       subgraph EXPORT ["Export (Python)"]
           direction TB
           VE["visual/visual.engine<br/>+ config.json"]
           LE["language/language.engine<br/>+ config.json + embedding.safetensors<br/>+ tokenizer"]
           ACE["action_context/context.engine<br/>+ config.json"]
           AE["action/action.engine<br/>+ config.json"]
       end

       subgraph RT ["LLMInferenceRuntime (C++)"]
           direction TB
           VR["mVisionRunner"]
           LR["mLLMEngineRunner"]
           ACR["mActionContextRunner"]
           AR["mActionRunner"]
       end

       VE --> VR
       LE --> LR
       ACE --> ACR
       AE --> AR

       classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
       classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
       class LE,LR nvNode
       class VE,ACE,AE,VR,ACR,AR greyNode

Why config.json matters
-----------------------

The runtime never inspects the PyTorch model. Every buffer it allocates and every
engine binding it wires comes from two sources:

1. The **``config.json``** written next to each engine by
   :func:`save_trt_engine_module` (see :doc:`bindings`).
2. The **binding names** baked into the TRT engine at export, which must match the
   constants in ``common/bindingNames.h`` on the C++ side.

If either drifts, the runtime either fails validation at load or binds the wrong
GPU pointer. The next pages break down the runners and the exact binding contract.

.. toctree::
   :maxdepth: 1
   :hidden:

   runners
   bindings
