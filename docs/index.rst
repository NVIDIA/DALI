.. NVIDIA DALI documentation master file, created by
   sphinx-quickstart on Fri Jul  6 10:39:47 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NVIDIA DALI documentation
=========================

Today’s deep learning applications include complex, multi-stage pre-processing data pipelines that include compute-intensive steps mainly carried out on the CPU. For instance, steps such as load data from disk, decode, crop, random resize, color and spatial augmentations and format conversions are carried out on the CPUs, limiting the performance and scalability of training and inference tasks. In addition, the deep learning frameworks today have multiple data pre-processing implementations, resulting in challenges such as portability of training and inference workflows and code maintainability.

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks and an execution engine to accelerate input data pre-processing for deep learning applications. DALI provides both performance and flexibility of accelerating different data pipelines, as a single library, that can be easily integrated into different deep learning training and inference applications.

Key highlights of DALI include:

* Full data pipeline accelerated from reading from disk to getting ready for training/inference
* Flexibility through configurable graphs and custom operators
* Support for image classification and segmentation workloads
* Ease of integration through direct framework plugins and open source bindings
* Portable training workflows with multiple input formats - JPEG, LMDB, RecordIO, TFRecord
* Extensible for user specific needs through open source license

.. warning::
   You are currently viewing unstable developer preview of the documentation. To see the documentation for the latest stable release click `here <https://docs.nvidia.com/deeplearning/sdk/index.html#data-loading>`_

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   examples/getting started.ipynb
   examples/index
   framework_plugins
   api
   supported_ops


Indices and tables
==================

* :ref:`genindex`
