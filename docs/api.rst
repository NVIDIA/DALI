Python API
==========

DALI functionality is exposed through Python API for ease of use and interoperability with common
deep learning frameworks. This part of the documentation contains the detailed description of this
API.

.. currentmodule:: nvidia.dali.pipeline

`Pipeline <pipeline.rst>`_ section describes the :class:`Pipeline` class - the central and most
important part of every program using DALI.

`Types <data_types.rst>`_ section describes types used to construct and returned by DALI pipelines.

`Functional API <functional_api.rst>`_ (Experimental!) section describes a psuedo-imperative API
which can be used to define DALI pipelines with less verbosity.

.. toctree::
   :maxdepth: 2
   :caption:
        API documentation

   pipeline
   data_types
   functional_api
   math
