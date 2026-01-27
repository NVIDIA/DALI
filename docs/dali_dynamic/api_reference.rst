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

DType class
^^^^^^^^^^^

.. currentmodule:: nvidia.dali.experimental.dynamic

.. autoclass:: DType
   :members:

Type conversion functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dtype

.. autofunction:: type_id

DALI Dynamic Mode data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: operations/types_table

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
