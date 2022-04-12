.. _operation reference:

Operation Reference
===================

.. currentmodule:: nvidia.dali

The data processing graph within a DALI :ref:`Pipeline <pipeline>` is defined by calling operation
functions. They accept and return instances of :class:`~nvidia.dali.pipeline.DataNode`,
which are *symbolic* representations of batches of Tensors.
The operation functions cannot be used to process data directly.

The constraints for defining the processing pipeline can be found in
:ref:`this section <processing_graph_structure>` of Pipeline documentation.

The following table lists all operations available in DALI:

.. include:: operations/fn_table

.. include:: operations/fn_autodoc
