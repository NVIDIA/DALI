|License|

NVIDIA DALI
===========

.. overview-begin-marker-do-not-remove

Today’s deep learning applications include complex, multi-stage pre-processing data pipelines that include compute-intensive steps mainly carried out on the CPU. For instance, steps such as load data from disk, decode, crop, random resize, color and spatial augmentations and format conversions are carried out on the CPUs, limiting the performance and scalability of training and inference tasks. In addition, the deep learning frameworks today have multiple data pre-processing implementations, resulting in challenges such as portability of training and inference workflows and code maintainability.

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks and an execution engine to accelerate input data pre-processing for deep learning applications. DALI provides both performance and flexibility of accelerating different data pipelines, as a single library, that can be easily integrated into different deep learning training and inference applications.

Key highlights of DALI include:

* Full data pipeline accelerated from reading from disk to getting ready for training/inference
* Flexibility through configurable graphs and custom operators
* Support for image classification and segmentation workloads
* Ease of integration through direct framework plugins and open source bindings
* Portable training workflows with multiple input formats - JPEG, LMDB, RecordIO, TFRecord
* Extensible for user specific needs through open source license

.. overview-end-marker-do-not-remove

.. installation-begin-marker-do-not-remove

DALI and NGC
------------

DALI is preinstalled in the `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_ TensorFlow, PyTorch, and MXNet containers in versions 18.07 and later.

Installing prebuilt DALI packages
---------------------------------

Prerequisities
^^^^^^^^^^^^^^

.. |driver link| replace:: **NVIDIA Driver**
.. _driver link: https://www.nvidia.com/drivers
.. |cuda link| replace:: **NVIDIA CUDA 9.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |mxnet link| replace:: **MXNet 1.3 beta**
.. _mxnet link: http://mxnet.incubator.apache.org
.. |pytorch link| replace:: **pyTorch 0.4**
.. _pytorch link: https://pytorch.org
.. |tf link| replace:: **TensorFlow 1.7**
.. _tf link: https://www.tensorflow.org

-  **Linux x64**
-  |driver link|_ supporting `CUDA 9.0 <https://developer.nvidia.com/cuda-downloads>`__ or later (i.e., 384.xx or later driver releases)
-  One or more of the following Deep Learning frameworks:

   -  |mxnet link|_ ``mxnet-cu90==1.3.0b20180612`` or later
   -  |pytorch link|_
   -  |tf link|_ or later

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali

Compiling DALI from source
--------------------------

Prerequisities
^^^^^^^^^^^^^^

.. |nvjpeg link| replace:: **nvJPEG library**
.. _nvjpeg link: https://developer.nvidia.com/nvjpeg
.. |protobuf link| replace:: **protobuf**
.. _protobuf link: https://github.com/google/protobuf
.. |cmake link| replace:: **CMake 3.5**
.. _cmake link: https://cmake.org
.. |jpegturbo link| replace:: **libjpeg-turbo 1.5.x**
.. _jpegturbo link: https://github.com/libjpeg-turbo/libjpeg-turbo
.. |opencv link| replace:: **OpenCV 3**
.. _opencv link: https://opencv.org
.. |lmdb link| replace:: **liblmdb 0.9.x**
.. _lmdb link: https://github.com/LMDB/lmdb

-  **Linux x64**
-  |cuda link|_
   *(CUDA 8.0 compatibility is provided unofficially)*
-  |nvjpeg link|_
   *(This can be unofficially disabled; see below)*
-  |protobuf link|_ version 2 or later (version 3 or later is required for TensorFlow TFRecord file format support)
-  |cmake link|_ or later
-  |jpegturbo link|_ or later
   *(This can be unofficially disabled; see below)*
-  |opencv link|_ or later
   *(OpenCV 2.x compatibility is provided unofficially)*
-  **(Optional)** |lmdb link|_ or later
-  One or more of the following Deep Learning frameworks:

   -  |mxnet link|_ ``mxnet-cu90==1.3.0b20180612`` or later
   -  |pytorch link|_
   -  |tf link|_ or later

.. note::

   TensorFlow installation is required to build the TensorFlow plugin for DALI

.. note::

   Items marked *"unofficial"* are community contributions that are
   believed to work but not officially tested or maintained by NVIDIA.

Get the DALI source
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone --recursive https://github.com/NVIDIA/dali
   cd dali

Make the build directory
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir build
   cd build

Compile DALI
^^^^^^^^^^^^

To build DALI without LMDB support:

.. code-block:: bash

   cmake ..
   make -j"$(nproc)"

To build DALI with LMDB support:

.. code-block:: bash

   cmake -DBUILD_LMDB=ON ..
   make -j"$(nproc)"

Optional CMake build parameters:

-  ``BUILD_PYTHON`` - build Python bindings (default: ON)
-  ``BUILD_TEST`` - include building test suite (default: ON)
-  ``BUILD_BENCHMARK`` - include building benchmarks (default: ON)
-  ``BUILD_LMDB`` - build with support for LMDB (default: OFF)
-  ``BUILD_NVTX`` - build with NVTX profiling enabled (default: OFF)
-  ``BUILD_TENSORFLOW`` - build TensorFlow plugin (default: OFF)
-  *(Unofficial)* ``BUILD_JPEG_TURBO`` - build with libjpeg-turbo (default: ON)
-  *(Unofficial)* ``BUILD_NVJPEG`` - build with nvJPEG (default: ON)

Install Python bindings
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install dali/python

.. installation-end-marker-do-not-remove

Getting started
---------------

.. |examples link| replace:: ``docs/examples``
.. _examples link: docs/examples

|examples link|_ directory contains a series of examples (in the form of Jupyter notebooks) of different features of DALI. It also contains examples of how to use DALI to interface with DL frameworks.

Documentation for the latest stable release is available `here <https://docs.nvidia.com/deeplearning/sdk/index.html#data-loading>`_. Nightly version of the documentation that stays in sync with the master branch is available `here <https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/index.html>`_.

Additional resources
--------------------

- GPU Technology Conference 2018 presentation about DALI, T. Gale, S. Layton and P. Tredak: `slides <http://on-demand.gputechconf.com/gtc/2018/presentation/s8906-fast-data-pipelines-for-deep-learning-training.pdf>`_, `recording <http://on-demand.gputechconf.com/gtc/2018/video/S8906/>`_.

Contributing to DALI
--------------------

Contributions to DALI are more than welcome. To make the pull request process smooth, please follow these `guidelines <CONTRIBUTING.md>`_.

Contributors
------------

DALI was built with major contributions from Trevor Gale, Przemek Tredak, Simon Layton, Andrei Ivanov, Serge Panev

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
