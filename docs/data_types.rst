Types
=====

.. _TensorList:

TensorList
----------
.. currentmodule:: nvidia.dali.pipeline

TensorList represents a batch of tensors. TensorLists are the return values of `Pipeline.run`
or `Pipeline.share_outputs`

.. currentmodule:: nvidia.dali.backend


TensorListCPU
^^^^^^^^^^^^^
.. autoclass:: TensorListCPU
   :members:
   :special-members: __getitem__, __init__

TensorListGPU
^^^^^^^^^^^^^
.. autoclass:: TensorListGPU
   :members:
   :special-members: __getitem__, __init__

Tensor
------

TensorCPU
^^^^^^^^^
.. autoclass:: TensorCPU
   :members:
   :undoc-members:
   :special-members: __init__

TensorGPU
^^^^^^^^^
.. autoclass:: TensorGPU
   :members:
   :undoc-members:
   :special-members: __init__


.. _layout_str_doc:

Data Layouts
------------
.. include:: data_layout.rst


Constant wrapper
----------------
.. currentmodule:: nvidia.dali.types

Constant
^^^^^^^^
.. autofunction:: Constant
.. autoclass:: ScalarConstant
   :members:

Enums
-----

DALIDataType
^^^^^^^^^^^^
.. autoclass:: DALIDataType
   :members:
   :undoc-members:

DALIIterpType
^^^^^^^^^^^^^
.. autoclass:: DALIInterpType
   :members:
   :undoc-members:

DALIImageType
^^^^^^^^^^^^^
.. autoclass:: DALIImageType
   :members:
   :undoc-members:

TensorLayout
^^^^^^^^^^^^
.. autoclass:: TensorLayout
   :members:
   :undoc-members:

PipelineAPIType
^^^^^^^^^^^^^^^
.. autoclass:: PipelineAPIType
   :members:
   :undoc-members:

