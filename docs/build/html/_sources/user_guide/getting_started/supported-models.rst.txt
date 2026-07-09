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
     - Export, inference, benchmark via ``app.py``; see :doc:`../../examples/gr00t`
   * - :doc:`models/pi05`
     - ``pi05``
     - Export, inference, benchmark via ``app.py``; see :doc:`../../examples/pi05`
   * - :doc:`models/smolvla`
     - ``smolvla``
     - Export, inference, benchmark via ``app.py``; see :doc:`../../examples/smolvla`
   * - :doc:`models/molmo2`
     - ``molmo2``
     - E2E parity via ``test_vla_molmo2_e2e.py``; see :doc:`../../examples/molmo2`
   * - :doc:`models/alpamayo`
     - ``alpamayo``
     - E2E parity via ``test_vla_alpamayo_e2e.py``; see :doc:`../../examples/alpamayo`

.. toctree::
   :maxdepth: 1
   :hidden:

   models/gr00t
   models/pi05
   models/molmo2
   models/smolvla
   models/alpamayo
