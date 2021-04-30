Types
=====

.. _TensorList:

TensorList
----------
.. currentmodule:: nvidia.dali

TensorList represents a batch of tensors. TensorLists are the return values of :meth:`Pipeline.run`,
:meth:`Pipeline.outputs` or :meth:`Pipeline.share_outputs`.

Subsequent invocations of the mentioned functions (or :meth:`Pipeline.release_outputs`) invalidate
the TensorList (as well as any DALI :ref:`Tensors<Tensor>` obtained from it) and indicate to DALI
that the memory can be used for something else.

TensorList wraps the outputs of current iteration and is valid only for the duration of the
iteration. Using the TensorList after moving to the next iteration is not allowed.
If you wish to retain the data you need to copy it before indicating DALI that you released it.

For typicall use-cases, for example when DALI is used through :ref:`DL Framework Plugins`,
no additionall memory bookkeeping is necessary.

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
   :special-members: __init__, __array_interface__

TensorGPU
^^^^^^^^^
.. autoclass:: TensorGPU
   :members:
   :undoc-members:
   :special-members: __init__, __cuda_array_interface__


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
.. autoenum:: DALIDataType
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name

DALIIterpType
^^^^^^^^^^^^^
.. autoenum:: DALIInterpType
   :members:
   :undoc-members:
   :exclude-members: name

DALIImageType
^^^^^^^^^^^^^
.. autoenum:: DALIImageType
   :members:
   :undoc-members:
   :exclude-members: name


SampleInfo
^^^^^^^^^^
.. autoclass:: SampleInfo
   :members:

TensorLayout
^^^^^^^^^^^^
.. autoclass:: TensorLayout
   :members:
   :undoc-members:

PipelineAPIType
^^^^^^^^^^^^^^^
.. autoenum:: PipelineAPIType
   :members:
   :undoc-members:

