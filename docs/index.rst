.. NVIDIA DALI documentation main file, created by
   sphinx-quickstart on Fri Jul  6 10:39:47 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NVIDIA DALI Documentation
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

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   Platform Support <support_matrix>
   Getting Started Tutorial <examples/getting_started.ipynb>
   Reporting vulnerabilities <security>

.. toctree::
   :hidden:
   :caption: Python API Documentation

   pipeline
   data_types
   math
   indexing
   supported_ops
   auto_aug/auto_aug
   supported_ops_legacy
   framework_plugins

.. toctree::
   :hidden:
   :caption: Examples and Tutorials

   examples/index

.. toctree::
   :hidden:
   :caption: Advanced

   advanced_topics_performance_tuning
   advanced_topics_sharding
   advanced_topics_pipe_run
   advanced_topics_checkpointing
   advanced_topics_experimental
   compilation
   env_vars

.. toctree::
   :hidden:
   :caption: Frequently Asked Questions

   FAQ

.. toctree::
   :hidden:
   :caption: Reference

   Release Notes <https://docs.nvidia.com/deeplearning/dali/release-notes/index.html>
   GitHub <https://github.com/NVIDIA/DALI>
   Roadmap <https://github.com/NVIDIA/DALI/issues/5320>
