|License|  |Documentation|

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

Prerequisites
^^^^^^^^^^^^^

.. |driver link| replace:: **NVIDIA Driver**
.. _driver link: https://www.nvidia.com/drivers
.. |cuda link| replace:: **NVIDIA CUDA 9.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |mxnet link| replace:: **MXNet 1.3**
.. _mxnet link: http://mxnet.incubator.apache.org
.. |pytorch link| replace:: **PyTorch 0.4**
.. _pytorch link: https://pytorch.org
.. |tf link| replace:: **TensorFlow 1.7**
.. _tf link: https://www.tensorflow.org

-  **Linux x64**
-  |driver link|_ supporting `CUDA 9.0 <https://developer.nvidia.com/cuda-downloads>`__ or later (i.e., 384.xx or later driver releases)
-  One or more of the following deep learning frameworks:

   -  |mxnet link|_ ``mxnet-cu90`` or later
   -  |pytorch link|_
   -  |tf link|_ or later

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali

.. note::
   nvidia-dali package contains prebuilt versions of the dali tensorflow plugin for several versions of tensorflow.
   Since release 0.6.1 there is also a possibility to install dali tensorflow plugin for the currently installed version of tensorflow, thus allowing forward compatibility:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin

.. note::
Installing this package will install nvidia-dali and its dependencies if not already installed. The package tensorflow-gpu must be installed before attempting to install nvidia-dali-tf-plugin.

.. note::
The package nvidia-dali-tf-plugin has a strict requirement with nvidia-dali as its exact same version. Thus, installing nvidia-dali-tf-plugin at its latest version will replace any older nvidia-dali versions already installed with the latest. To work with older versions of DALI, please provide the version explicitely to the pip install command.

.. code-block:: bash

   OLDER_VERSION=0.6.1
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin==$OLDER_VERSION

Compiling DALI from source
--------------------------

Prerequisites
^^^^^^^^^^^^^

.. |nvjpeg link| replace:: **nvJPEG library**
.. _nvjpeg link: https://developer.nvidia.com/nvjpeg
.. |protobuf link| replace:: **protobuf**
.. _protobuf link: https://github.com/google/protobuf
.. |cmake link| replace:: **CMake 3.5**
.. _cmake link: https://cmake.org
.. |jpegturbo link| replace:: **libjpeg-turbo 1.5.x**
.. _jpegturbo link: https://github.com/libjpeg-turbo/libjpeg-turbo
.. |ffmpeg link| replace:: **FFmpeg 3.4.2**
.. _ffmpeg link: https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-3.4.2.tar.bz2
.. |opencv link| replace:: **OpenCV 3**
.. _opencv link: https://opencv.org
.. |lmdb link| replace:: **liblmdb 0.9.x**
.. _lmdb link: https://github.com/LMDB/lmdb
.. |gcc link| replace:: **GCC 4.9.2**
.. _gcc link: https://www.gnu.org/software/gcc/

.. table::
   :align: center

   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | **Linux x64**                          |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |gcc link|_ or later                   |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |cuda link|_                           | *CUDA 8.0 compatibility is provided unofficially*                                           |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |nvjpeg link|_                         | *This can be unofficially disabled. See below*                                              |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |protobuf link|_                       | | version 2 or later                                                                        |
   |                                        | | (version 3 or later is required for TensorFlow TFRecord file format support)              |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |cmake link|_ or later                 |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |jpegturbo link|_ or later             | *This can be unofficially disabled. See below*                                              |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |ffmpeg link|_ or later                | We recommend using version 3.4.2 compiled following the *instructions below*.               |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |opencv link|_ or later                | | We recommend using version 3.4+, however previous versions are also compatible.           |
   |                                        | | *OpenCV 2.x compatibility is provided unofficially*                                       |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | **(Optional)** |lmdb link|_ or later   |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | One or more of the following Deep Learning frameworks:                                                                               |
   |      -  |mxnet link|_ ``mxnet-cu90`` or later                                                                                        |
   |      -  |pytorch link|_                                                                                                              |
   |      -  |tf link|_ or later                                                                                                          |
   +----------------------------------------+---------------------------------------------------------------------------------------------+

.. note::

   TensorFlow installation is required to build the TensorFlow plugin for DALI

.. note::

   Items marked *"unofficial"* are community contributions that are
   believed to work but not officially tested or maintained by NVIDIA.

.. note::

   This software uses code of FFmpeg licensed under the LGPLv2.1 and its source can be downloaded https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-3.4.2.tar.bz2

   FFmpeg was compiled using the following command line:

.. code-block:: bash

   ./configure \
     --prefix=/usr/local \
     --disable-static \
     --disable-all \
     --disable-autodetect \
     --disable-iconv \
     --enable-shared \
     --enable-avformat \
     --enable-avcodec \
     --enable-avfilter \
     --enable-protocol=file \
     --enable-demuxer=mov,matroska \
     --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb && \
     make

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

To build DALI using Clang (experimental):

.. note::

   This build is experimental and it is not maintained and tested
   like the default configuration. It is not guaranteed to work.
   We recommend using GCC for production builds.

.. code-block:: bash

   cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang  ..
   make -j"$(nproc)"

Optional CMake build parameters:

-  ``BUILD_PYTHON`` - build Python bindings (default: ON)
-  ``BUILD_TEST`` - include building test suite (default: ON)
-  ``BUILD_BENCHMARK`` - include building benchmarks (default: ON)
-  ``BUILD_LMDB`` - build with support for LMDB (default: OFF)
-  ``BUILD_NVTX`` - build with NVTX profiling enabled (default: OFF)
-  ``BUILD_TENSORFLOW`` - build TensorFlow plugin (default: OFF)
-  ``WERROR`` - treat all build warnings as errors (default: OFF)
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

The |examples link|_ directory contains a series of examples (in the form of Jupyter notebooks) highlighting different features of DALI. It also contains examples of how to use DALI to interface with deep learning frameworks.

Documentation for the latest stable release is available `here <https://docs.nvidia.com/deeplearning/sdk/index.html#data-loading>`_. Nightly version of the documentation that stays in sync with the master branch is available `here <https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/index.html>`_.

Additional resources
--------------------

- GPU Technology Conference 2018 presentation about DALI, T. Gale, S. Layton and P. Tredak: `slides <http://on-demand.gputechconf.com/gtc/2018/presentation/s8906-fast-data-pipelines-for-deep-learning-training.pdf>`_, `recording <http://on-demand.gputechconf.com/gtc/2018/video/S8906/>`_.

Contributing to DALI
--------------------

Contributions to DALI are more than welcome. To contribute to DALI and make pull requests, follow the guidelines outlined in the `Contributing <CONTRIBUTING.md>`_ document.

Reporting problems, asking questions
-----------------------------------

We appreciate any feedback, questions or bug reporting regarding this project. When help with code is needed, follow the process outlined in the Stack Overflow (https://stackoverflow.com/help/mcve) document. Ensure posted examples are:
- minimal – use as little code as possible that still produces the same problem
- complete – provide all parts needed to reproduce the problem. Check if you can strip external dependency and still show the problem. The less time we spend on reproducing problems the more time we have to fix it
- verifiable – test the code you're about to provide to make sure it reproduces the problem. Remove all other problems that are not related to your request/question.

Contributors
------------

DALI was built with major contributions from Trevor Gale, Przemek Tredak, Simon Layton, Andrei Ivanov, Serge Panev

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. |Documentation| image:: https://img.shields.io/badge/Nvidia%20DALI-documentation-brightgreen.svg?longCache=true
   :target: https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/
