Installation
============

Installing prebuilt DALI packages
---------------------------------

Prerequisities
^^^^^^^^^^^^^^

Ensure you meet the following minimum requirements:

* Linux x64
* `NVIDIA Driver <http://www.nvidia.com/Download/index.aspx>`_ supporting CUDA 9.0 or later

  * This corresponds to 384.xx and later driver releases.

* DALI can work with any of the following Deep Learning frameworks:

  * `MXNet <http://mxnet.incubator.apache.org>`_

    * Version 1.3 beta is required, `mxnet-cu90==1.3.0b20180612` or later

  * `pyTorch <https://pytorch.org>`_

    * Version 0.4

  * `TensorFlow <https://www.tensorflow.org>`_

    * Version 1.7 or newer

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali


Compiling DALI from source
--------------------------

Prerequisities
^^^^^^^^^^^^^^

* Linux
* `NVIDIA CUDA 9.0 or later <https://developer.nvidia.com/cuda-downloads>`_
* `nvJPEG library <https://developer.nvidia.com/nvjpeg>`_
* `protobuf <https://github.com/google/protobuf>`_ version 2 or above (version 3 or above is required for TensorFlow TFRecord file format support)
* `CMake <https://cmake.org>`_ version 3.5 or above
* `libjpeg-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_ version 1.5.x or above
* `OpenCV <https://opencv.org>`_ version 3 or above
* (Optional) `liblmdb <https://github.com/LMDB/lmdb>`_ version 0.9.x or above
* DALI can work with any of the following Deep Learning frameworks:

  * `MXNet <http://mxnet.incubator.apache.org>`_

    * Version 1.3 beta is required, `mxnet-cu90==1.3.0b20180612` or later

  * `pyTorch <https://pytorch.org>`_

    * Version 0.4

  * `TensorFlow <https://www.tensorflow.org>`_

    * Version 1.7 or newer

.. note::

   Installing TensorFlow is required to build the TensorFlow plugin for DALI


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
   make -j"$(nproc)" install

To build DALI with LMDB support:

.. code-block:: bash

   cmake -DBUILD_LMDB=ON ..
   make -j"$(nproc)" install

Optional CMake build parameters:

- `BUILD_PYTHON` - build Python bindings (default: ON)
- `BUILD_TEST` - include building test suite (default: ON)
- `BUILD_BENCHMARK` - include building benchmarks (default: ON)
- `BUILD_LMDB` - build with support for LMDB (default: OFF)
- `BUILD_NVTX` - build with NVTX profiling enabled (default: OFF)
- `BUILD_TENSORFLOW` - build TensorFlow plugin (default: OFF)

Install Python bindings
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install dali/python

