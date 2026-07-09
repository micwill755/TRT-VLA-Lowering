Torch-TRT pipelines Documentation
=================================

Welcome to the Torch-TRT pipelines documentation. This project provides export,
inference, and benchmark orchestration for VLA models on top of
`TensorRT Edge-LLM <https://nvidia.github.io/TensorRT-Edge-LLM/latest/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   user_guide/getting_started/overview
   user_guide/getting_started/supported-models
   user_guide/getting_started/installation

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/gr00t
   examples/pi05
   examples/smolvla
   examples/molmo2
   examples/alpamayo

.. toctree::
   :maxdepth: 2
   :caption: Edge LLM

   edge_llm/overview
   edge_llm/runners
   edge_llm/bindings

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   developer_guide/architecture/overview
   developer_guide/architecture/pipelines-and-stages
   developer_guide/architecture/runners
   developer_guide/architecture/hooks
   developer_guide/architecture/export-pipeline
   developer_guide/architecture/inference-pipeline
   developer_guide/architecture/load-pipeline
   developer_guide/architecture/benchmark-pipeline

.. toctree::
   :maxdepth: 2
   :caption: Plugins

   developer_guide/plugins/overview
   developer_guide/plugins/architecture
   developer_guide/plugins/registration
   developer_guide/plugins/custom-ops
   developer_guide/plugins/converters
   developer_guide/plugins/attention-patching
   developer_guide/plugins/parity-and-debugging

.. toctree::
   :maxdepth: 2
   :caption: Export Modules

   developer_guide/export_modules/overview
   developer_guide/export_modules/vision-example
   developer_guide/export_modules/language-example
   developer_guide/export_modules/action-context-example
   developer_guide/diffusion/overview
   developer_guide/diffusion/components
   developer_guide/diffusion/groot-example
   developer_guide/diffusion/action-rollout

.. toctree::
   :maxdepth: 2
   :caption: Customization

   developer_guide/customization/customization-guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
