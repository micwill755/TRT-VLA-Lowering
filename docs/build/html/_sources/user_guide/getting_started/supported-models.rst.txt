Supported Models
================

The following models are supported (or in progress) in Torch-TRT pipelines.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Model
     - CLI alias
     - Notes
   * - :doc:`models/gr00t`
     - ``gr00t``, ``groot``
     - Export, load, inference, benchmark
   * - :doc:`models/pi05`
     - ``pi05``
     - Load registered; export varies by release
   * - :doc:`models/molmo2`
     - ``molmo2``
     - Export and inference
   * - :doc:`models/smolvla`
     - ``smolvla``
     - Load registered
   * - :doc:`models/alpamayo`
     - ``alpamayo``
     - VLA pipeline (Edge-LLM layout)

.. toctree::
   :maxdepth: 1
   :hidden:

   models/gr00t
   models/pi05
   models/molmo2
   models/smolvla
   models/alpamayo
