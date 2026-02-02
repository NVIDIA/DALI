API Reference
=============

.. currentmodule:: nvidia.dali.experimental.dynamic
    
This page documents the public API of DALI Dynamic.
For the list of all available operators, see :doc:`Operation Reference <ops_reference>`.

Tensor and Batch objects
------------------------

Batch
^^^^^
.. autoclass:: Batch
   :members:

Tensor
^^^^^^
.. autoclass:: Tensor
   :members:

tensor
^^^^^^
.. autofunction:: tensor

as_tensor
^^^^^^^^^
.. autofunction:: as_tensor

batch
^^^^^
.. autofunction:: batch

as_batch
^^^^^^^^
.. autofunction:: as_batch

Data types
----------

Those are the data type objects that DALI Dynamic uses to indicate the type of elements of Tensors and Batches. 
They can typically be passed as `dtype` argument to request specific output type of the operator. 
There are also several DALI-specific types, representing DALI enums.

All of the types below are instances of the :py:class:`DType` class.

.. include:: operations/types_table

Type conversion functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dtype

.. autofunction:: type_id

DType class
^^^^^^^^^^^

.. currentmodule:: nvidia.dali.experimental.dynamic

.. autoclass:: DType
   :members:

Execution context
-----------------

Device
^^^^^^
.. autoclass:: Device
   :members:

EvalContext
^^^^^^^^^^^
.. autoclass:: EvalContext
   :members:

EvalMode
^^^^^^^^
.. autoclass:: EvalMode
   :members:

Random state
------------

.. currentmodule:: nvidia.dali.experimental.dynamic.random

RNG
^^^

.. autoclass:: RNG
   :members:
