Operations
==========

.. currentmodule:: nvidia.dali

Operations functions are used to define the data processing graph within a DALI :ref:`Pipeline <pipeline>`.
They accept as inputs and return as outputs :class:`~nvidia.dali.pipeline.DataNode` instances, which represent batches of Tensors. 
It is worth noting that those operation functions can not be used to process data directly.

The following table lists all available operations available in DALI:

.. include:: fn_table

Modules
^^^^^^^

.. include:: fn_autodoc
