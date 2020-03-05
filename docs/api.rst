Python API
==========

Pipeline
--------
.. autoclass:: nvidia.dali.pipeline.Pipeline
   :members:

Tensor
------

TensorCPU
^^^^^^^^^
.. autoclass:: nvidia.dali.backend.TensorCPU
   :members:
   :undoc-members:

TensorGPU
^^^^^^^^^
.. autoclass:: nvidia.dali.backend.TensorGPU
   :members:
   :undoc-members:

TensorList
----------

TensorListCPU
^^^^^^^^^^^^^
.. autoclass:: nvidia.dali.backend.TensorListCPU
   :members:
   :special-members: __getitem__

TensorListGPU
^^^^^^^^^^^^^
.. autoclass:: nvidia.dali.backend.TensorListGPU
   :members:
   :special-members: __getitem__


.. _layout_str_doc:
------------
.. include:: data_layout.rst

-----

DALIDataType
^^^^^^^^^^^^
.. autoclass:: nvidia.dali.types.DALIDataType
   :members:
   :member-order: bysource
   :undoc-members:

DALIIterpType
^^^^^^^^^^^^^
.. autoclass:: nvidia.dali.types.DALIInterpType
   :members:
   :member-order: bysource
   :undoc-members:

DALIImageType
^^^^^^^^^^^^^
.. autoclass:: nvidia.dali.types.DALIImageType
   :members:
   :member-order: bysource
   :undoc-members:

PipelineAPIType
^^^^^^^^^^^^^^^
.. autoclass:: nvidia.dali.types.PipelineAPIType
   :members:
   :undoc-members:
   :member-order: bysource
