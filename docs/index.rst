.. NVIDIA DALI documentation main file, created by
   sphinx-quickstart on Fri Jul  6 10:39:47 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NVIDIA DALI documentation
=========================

.. ifconfig:: "dev" in release

   .. warning::
      You are currently viewing unstable developer preview of the documentation.
      To see the documentation for the latest stable release, refer to:

      * `Release Notes <https://docs.nvidia.com/deeplearning/dali/release-notes/index.html>`_
      * `Developer Guide <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>`_ (stable version of this page)

.. include:: ../README.rst
   :start-after: overview-begin-marker-do-not-remove
   :end-before: overview-end-marker-do-not-remove

This library is open sourced and it is available in the `NVIDIA GitHub repository <https://github.com/NVIDIA/DALI>`_.

.. toctree::
   :hidden:

   Documentation home <self>

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   Platform Support <support_matrix>
   Getting Started Tutorial <examples/getting started.ipynb>

.. toctree::
   :maxdepth: 2
   :caption: Python API Documentation

   pipeline
   data_types
   supported_ops
   math
   framework_plugins
   supported_ops_legacy

.. toctree::
   :maxdepth: 2
   :caption: Examples and Tutorials

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   advanced_topics
   compilation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   Release Notes <https://docs.nvidia.com/deeplearning/dali/release-notes/index.html>
   GitHub <https://github.com/NVIDIA/DALI>

Indices and tables
==================

* :ref:`genindex`
