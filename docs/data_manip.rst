Data Manipulation
=================

.. currentmodule:: nvidia.dali.pipeline

DALI follows a graph-based approach to data processing. This graph is defined by
calling operator functions on data nodes (see :class:`DataNode`). The data nodes also support
Python-style indexing and can be incorporated in mathematical expressions.

.. toctree::
   :maxdepth: 2

   indexing
   math
   supported_ops
